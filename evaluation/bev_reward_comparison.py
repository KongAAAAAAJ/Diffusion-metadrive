"""Compare Stage 1 and GRPO checkpoints using only per-step joint rewards."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from evaluation.bev_four_model_evaluator import (
    _configure_deterministic_inference,
    _initial_scene_sha256,
    _initial_state_signature,
    _load_policy,
    _sync,
    _validate_common_reward_binding,
)
from evaluation.bev_model_manifest import ModelSpec, file_sha256
from expert_dataset.collect_joint_bev import (
    JointBEVSampleBuilder,
    SensorlessJointBEVPlatoonEnv,
    simulator_decision_dt_s,
)
from models.bev_planner.joint_reward import (
    JointRewardConfig,
    JointRewardResult,
    JointTrajectoryProxyReward,
    joint_reward_config_sha256,
)
from models.bev_planner.trajectory_optimizer import (
    KinematicTrajectoryOptimizer,
    TrajectoryOptimizationError,
)
from models.decisioner.rule_decisioner import (
    LaneChangeCommitmentError,
    diffusion_mode_feedback_actions,
    hard_valid_modes_by_rule_action,
    joint_proposal_actions,
    make_rule_maker,
    match_joint_action_proposal,
)
from scenarios.bev_round13_contract import (
    BEVScenarioContractError,
    HOLDOUT_SEEDS,
    PRIMARY_S5_S9_SCENARIOS,
    deterministic_initial_speed_km_h,
    primary_scenario_contract,
)
from train.train_bev_diffusion_stage1 import planner_forward_from_batch
from train.train_bev_joint_grpo_online import (
    AGENT_IDS,
    constant_velocity_actions,
    episode_has_ended,
    execution_mode_valid_mask,
    joint_trajectory_action,
    model_inputs_to_batch,
    optimize_safe_stop_trajectories,
    optimize_selected_model_trajectories,
)


MODEL_IDS = ("stage1_a", "grpo_open")
REWARD_COLUMNS = (
    "total_reward",
    "progress_reward",
    "formation_reward",
    "gap_reward",
    "ttc_reward",
    "road_reward",
    "comfort_reward",
    "collision_reward",
    "out_of_drivable_reward",
)
CSV_COLUMNS = (
    "model",
    "scenario",
    "route",
    "seed",
    "step",
    *REWARD_COLUMNS,
)


class RewardComparisonError(RuntimeError):
    """Raised when a reward-only comparison violates its frozen contract."""


@dataclass(frozen=True)
class RewardComparisonConfig:
    device: Literal["cpu", "cuda"] = "cuda"
    seeds: tuple[int, ...] = HOLDOUT_SEEDS
    scenarios: tuple[tuple[str, str], ...] = PRIMARY_S5_S9_SCENARIOS
    max_steps: int = 800

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise RewardComparisonError("device must be cpu or cuda")
        if not self.seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in self.seeds
        ):
            raise RewardComparisonError("seeds must be a non-empty integer tuple")
        if isinstance(self.max_steps, bool) or self.max_steps <= 0:
            raise RewardComparisonError("max_steps must be a positive integer")
        try:
            primary_scenario_contract(self.scenarios)
        except BEVScenarioContractError as exc:
            raise RewardComparisonError(str(exc)) from exc


def _model_specs(stage1_checkpoint: Path, grpo_checkpoint: Path) -> tuple[ModelSpec, ...]:
    stage1 = Path(stage1_checkpoint).expanduser().resolve()
    grpo = Path(grpo_checkpoint).expanduser().resolve()
    stage1_sha = file_sha256(stage1)
    return (
        ModelSpec(
            model_id="stage1_a",
            kind="stage1",
            variant="A",
            checkpoint=stage1,
            checkpoint_sha256=stage1_sha,
        ),
        ModelSpec(
            model_id="grpo_open",
            kind="grpo",
            variant="A",
            checkpoint=grpo,
            checkpoint_sha256=file_sha256(grpo),
            reward_domain="tau_d",
            source_checkpoint=stage1,
            source_checkpoint_sha256=stage1_sha,
        ),
    )


def _load_models(
    stage1_checkpoint: Path,
    grpo_checkpoint: Path,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    reward_bindings: dict[str, tuple[str, str, str]] = {}
    models = {
        spec.model_id: _load_policy(
            spec,
            device=device,
            formal=False,
            reward_bindings=reward_bindings,
        )
        for spec in _model_specs(stage1_checkpoint, grpo_checkpoint)
    }
    binding = _validate_common_reward_binding(reward_bindings)
    expected_config_sha = joint_reward_config_sha256(JointRewardConfig())
    if binding is None or binding["reward_config_sha256"] != expected_config_sha:
        raise RewardComparisonError(
            "the GRPO checkpoint is not bound to the active default joint reward config"
        )
    return models


def _new_env(scenario: tuple[str, str], seed: int) -> object:
    env = SensorlessJointBEVPlatoonEnv(
        {
            "num_agents": 3,
            "traffic_density": 0.0,
            "initial_speed_km_h": deterministic_initial_speed_km_h(
                scenario[0], int(seed)
            ),
            "start_seed": int(seed),
            "num_scenarios": 1,
        }
    )
    env.set_runtime_scenario_route(*scenario)
    spawn_manager = getattr(getattr(env, "engine", None), "spawn_manager", None)
    set_spawn_seed = getattr(spawn_manager, "set_episode_spawn_seed", None)
    if callable(set_spawn_seed):
        set_spawn_seed(int(seed))
    env.reset(seed=int(seed))
    return env


def _mean_component(result: JointRewardResult, name: str) -> float:
    values = np.asarray(result.components[name], dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise RewardComparisonError(f"reward component {name} is invalid")
    return float(values.mean())


def reward_values(
    result: JointRewardResult,
    config: JointRewardConfig,
) -> dict[str, float]:
    """Return signed weighted terms whose sum is exactly the joint reward."""

    rewards = np.asarray(result.rewards, dtype=np.float64)
    if rewards.ndim != 1 or rewards.size == 0 or not np.isfinite(rewards).all():
        raise RewardComparisonError("joint total reward is invalid")
    values = {
        "total_reward": float(rewards.mean()),
        "progress_reward": config.progress_weight
        * _mean_component(result, "progress_score"),
        "formation_reward": -config.formation_weight
        * _mean_component(result, "formation_penalty"),
        "gap_reward": -config.gap_weight
        * _mean_component(result, "gap_penalty"),
        "ttc_reward": -config.ttc_weight
        * _mean_component(result, "ttc_penalty"),
        "road_reward": -config.road_weight
        * _mean_component(result, "road_penalty"),
        "comfort_reward": -config.comfort_weight
        * _mean_component(result, "comfort_penalty"),
        "collision_reward": -config.collision_penalty
        * float(np.asarray(result.collision, dtype=np.float64).mean()),
        "out_of_drivable_reward": -config.out_of_drivable_penalty
        * float(np.asarray(result.out_of_drivable, dtype=np.float64).mean()),
    }
    component_sum = sum(values[name] for name in REWARD_COLUMNS[1:])
    if not math.isclose(
        component_sum,
        values["total_reward"],
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    ):
        raise RewardComparisonError(
            "signed reward components do not sum to total_reward"
        )
    return values


def _reward_row(
    *,
    model_id: str,
    scenario: tuple[str, str],
    seed: int,
    step: int,
    result: JointRewardResult,
    config: JointRewardConfig,
) -> dict[str, object]:
    return {
        "model": model_id,
        "scenario": scenario[0],
        "route": scenario[1],
        "seed": int(seed),
        "step": int(step),
        **reward_values(result, config),
    }


def _write_reward_csv(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    if not rows:
        raise RewardComparisonError("evaluation produced no model planning rewards")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            if set(row) != set(CSV_COLUMNS):
                raise RewardComparisonError("reward row does not match the CSV contract")
            writer.writerow({name: row[name] for name in CSV_COLUMNS})


def _evaluate_model(
    model_id: str,
    planner: torch.nn.Module,
    *,
    device: torch.device,
    config: RewardComparisonConfig,
    reward_backend: JointTrajectoryProxyReward,
    reward_config: JointRewardConfig,
    reference_initial_states: dict[tuple[str, str, int], np.ndarray],
    reference_initial_scenes: dict[tuple[str, str, int], str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in config.scenarios:
        for seed in config.seeds:
            env = _new_env(scenario, int(seed))
            builder = JointBEVSampleBuilder(AGENT_IDS)
            builder.reset()
            rule_maker = make_rule_maker(dict(env.config))
            rule_maker.reset(env, list(AGENT_IDS))
            trajectory_optimizer = KinematicTrajectoryOptimizer()
            committed_execution_id: int | None = None
            committed_plan_actions: dict[str, int] | None = None
            dt_s = simulator_decision_dt_s(env)
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            initial_key = (scenario[0], scenario[1], int(seed))
            initial_state = _initial_state_signature(env)
            reference_state = reference_initial_states.setdefault(
                initial_key, initial_state.copy()
            )
            if not np.allclose(initial_state, reference_state, rtol=0.0, atol=1.0e-8):
                raise RewardComparisonError(
                    f"models received different reset state for {scenario[0]} seed={seed}"
                )
            initial_scene = _initial_scene_sha256(env)
            reference_scene = reference_initial_scenes.setdefault(
                initial_key, initial_scene
            )
            if initial_scene != reference_scene:
                raise RewardComparisonError(
                    f"models received different initial scene for {scenario[0]} seed={seed}"
                )
            try:
                for step_index in range(config.max_steps):
                    builder.capture_state(env, step_index * dt_s)
                    if not builder.history_ready():
                        action = constant_velocity_actions(env)
                    else:
                        values = builder.build_model_inputs(env)
                        rule_batch = None
                        rule_condition: dict[str, int] | None = None
                        rule_condition_is_commitment = False
                        hard_modes = hard_valid_modes_by_rule_action(
                            AGENT_IDS, values.mode_valid_mask
                        )
                        if rule_maker.has_active_lane_change_commitments:
                            if (
                                committed_execution_id is None
                                or committed_plan_actions is None
                            ):
                                raise RewardComparisonError(
                                    "RuleMaker commitment has no execution state"
                                )
                            try:
                                rule_maker.advance_committed_execution(
                                    env,
                                    list(AGENT_IDS),
                                    committed_execution_id,
                                )
                            except LaneChangeCommitmentError as exc:
                                raise RewardComparisonError(
                                    f"RuleMaker commitment failed: {exc}"
                                ) from exc
                            if rule_maker.has_active_lane_change_commitments:
                                rule_condition = (
                                    rule_maker.committed_execution_rule_actions(
                                        env, committed_plan_actions
                                    )
                                )
                                rule_condition_is_commitment = True
                            else:
                                committed_execution_id = None
                                committed_plan_actions = None
                        if rule_condition is None:
                            try:
                                rule_batch = rule_maker.propose_joint_actions(
                                    env,
                                    list(AGENT_IDS),
                                    getattr(env, "_last_planner_batch", None) or {},
                                    hard_valid_modes_by_action=hard_modes,
                                )
                            except LaneChangeCommitmentError as exc:
                                raise RewardComparisonError(
                                    f"RuleMaker proposal failed: {exc}"
                                ) from exc
                            rule_condition = (
                                joint_proposal_actions(
                                    rule_batch.proposals[0], AGENT_IDS
                                )
                                if rule_batch.proposals
                                else {agent_id: 0 for agent_id in AGENT_IDS}
                            )
                        values = builder.augment_v2_model_inputs(
                            env,
                            values,
                            rule_action_condition=rule_condition,
                            rule_formation_state=rule_maker.is_formation_locked,
                        )
                        try:
                            execution_mask = execution_mode_valid_mask(
                                values, optimizer=trajectory_optimizer
                            )
                        except TrajectoryOptimizationError:
                            break
                        batch = model_inputs_to_batch(
                            values,
                            device,
                            mode_valid_mask=execution_mask,
                        )
                        noise = torch.randn(
                            (1, 3, 10, 8, 2),
                            dtype=torch.float32,
                            device=device,
                            generator=generator,
                        )
                        _sync(device)
                        output = planner_forward_from_batch(
                            planner, batch, diffusion_noise=noise
                        )
                        _sync(device)
                        raw_trajectories = (
                            output["selected_trajectory"][0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                        selected_modes = (
                            output["selected_mode"][0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.int64, copy=False)
                        )
                        if raw_trajectories.shape != (3, 8, 3):
                            raise RewardComparisonError(
                                "selected joint trajectory must have shape [3,8,3]"
                            )
                        result = reward_backend.score(
                            env, values, raw_trajectories[None]
                        )
                        rows.append(
                            _reward_row(
                                model_id=model_id,
                                scenario=scenario,
                                seed=int(seed),
                                step=step_index,
                                result=result,
                                config=reward_config,
                            )
                        )

                        forced_safe_stop = False
                        try:
                            optimization = optimize_selected_model_trajectories(
                                values,
                                raw_trajectories,
                                selected_modes,
                                optimizer=trajectory_optimizer,
                            )
                        except TrajectoryOptimizationError:
                            optimization = optimize_safe_stop_trajectories(
                                values, optimizer=trajectory_optimizer
                            )
                            selected_modes = np.full((3,), 9, dtype=np.int64)
                            forced_safe_stop = True

                        matched = None
                        if not forced_safe_stop:
                            try:
                                _, feedback_actions, _ = (
                                    diffusion_mode_feedback_actions(
                                        selected_modes.tolist(),
                                        AGENT_IDS,
                                        scenario_id=scenario[0],
                                        local_route=scenario[1],
                                    )
                                )
                            except LaneChangeCommitmentError as exc:
                                raise RewardComparisonError(
                                    f"RuleMaker feedback failed: {exc}"
                                ) from exc
                            if rule_condition_is_commitment:
                                compatible = all(
                                    feedback_actions[agent_id]
                                    == int(rule_condition[agent_id])
                                    for agent_id in AGENT_IDS
                                )
                                if not compatible:
                                    optimization = optimize_safe_stop_trajectories(
                                        values, optimizer=trajectory_optimizer
                                    )
                                    selected_modes = np.full(
                                        (3,), 9, dtype=np.int64
                                    )
                                    forced_safe_stop = True
                            else:
                                matched = (
                                    None
                                    if rule_batch is None
                                    else match_joint_action_proposal(
                                        rule_batch, feedback_actions, AGENT_IDS
                                    )
                                )
                                if matched is None:
                                    optimization = optimize_safe_stop_trajectories(
                                        values, optimizer=trajectory_optimizer
                                    )
                                    selected_modes = np.full(
                                        (3,), 9, dtype=np.int64
                                    )
                                    forced_safe_stop = True

                        if not forced_safe_stop and matched is not None:
                            assert rule_batch is not None
                            try:
                                rule_maker.accept_joint_action(
                                    rule_batch.batch_id, matched.proposal_id
                                )
                            except LaneChangeCommitmentError as exc:
                                raise RewardComparisonError(
                                    f"RuleMaker acceptance failed: {exc}"
                                ) from exc
                            if rule_maker.has_active_lane_change_commitments:
                                committed_execution_id = int(rule_batch.batch_id)
                                committed_plan_actions = joint_proposal_actions(
                                    matched, AGENT_IDS
                                )
                        action = joint_trajectory_action(
                            optimization.optimized_trajectories
                        )

                    _, _, terminated, truncated, info = env.step(action)
                    if episode_has_ended(terminated, truncated, info):
                        break
            finally:
                env.close()
    return rows


@torch.no_grad()
def evaluate_reward_comparison(
    stage1_checkpoint: Path,
    grpo_checkpoint: Path,
    output_csv: Path,
    config: RewardComparisonConfig | None = None,
) -> list[dict[str, object]]:
    """Run both checkpoints and persist only one unified per-step reward CSV."""

    cfg = config or RewardComparisonConfig()
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RewardComparisonError("CUDA evaluation requested but unavailable")
    device = torch.device(cfg.device)
    _configure_deterministic_inference(device)
    models = _load_models(stage1_checkpoint, grpo_checkpoint, device)
    reward_config = JointRewardConfig()
    reward_backend = JointTrajectoryProxyReward(reward_config)
    reference_initial_states: dict[tuple[str, str, int], np.ndarray] = {}
    reference_initial_scenes: dict[tuple[str, str, int], str] = {}
    rows: list[dict[str, object]] = []
    for model_id in MODEL_IDS:
        rows.extend(
            _evaluate_model(
                model_id,
                models[model_id],
                device=device,
                config=cfg,
                reward_backend=reward_backend,
                reward_config=reward_config,
                reference_initial_states=reference_initial_states,
                reference_initial_scenes=reference_initial_scenes,
            )
        )
    if {str(row["model"]) for row in rows} != set(MODEL_IDS):
        raise RewardComparisonError("both checkpoints must produce reward rows")
    _write_reward_csv(rows, output_csv)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--grpo-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-steps", type=int, default=800)
    arguments = parser.parse_args(argv)
    try:
        rows = evaluate_reward_comparison(
            arguments.stage1_checkpoint,
            arguments.grpo_checkpoint,
            arguments.output,
            RewardComparisonConfig(
                device=arguments.device,
                max_steps=arguments.max_steps,
            ),
        )
    except RewardComparisonError as exc:
        parser.exit(2, f"reward comparison error: {exc}\n")
    counts = {
        model_id: sum(str(row["model"]) == model_id for row in rows)
        for model_id in MODEL_IDS
    }
    print(f"reward_csv={arguments.output}")
    print(f"reward_rows={len(rows)} model_rows={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CSV_COLUMNS",
    "MODEL_IDS",
    "REWARD_COLUMNS",
    "RewardComparisonConfig",
    "RewardComparisonError",
    "evaluate_reward_comparison",
    "reward_values",
]
