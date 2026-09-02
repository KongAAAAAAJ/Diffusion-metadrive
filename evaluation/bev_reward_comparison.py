"""Compare Stage 1 and GRPO checkpoints using only per-step joint rewards."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from evaluation.bev_model_manifest import ModelSpec, file_sha256
from expert_dataset.collect_joint_bev import (
    JointBEVSampleBuilder,
    SensorlessJointBEVPlatoonEnv,
    simulator_decision_dt_s,
)
from models.bev_planner.joint_reward import (
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256,
    JOINT_REWARD_CONTRACT,
    JOINT_REWARD_CONTRACT_SHA256,
    JointRewardConfig,
    JointRewardError,
    JointRewardResult,
    JointTrajectoryProxyReward,
    joint_reward_config_sha256,
)
from models.bev_planner.trajectory_optimizer import (
    KinematicTrajectoryOptimizer,
    KinematicTrajectoryOptimizerConfig,
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
from train.bev_joint_grpo import (
    GRPO_CHECKPOINT_FORMAT,
    GRPO_CHECKPOINT_SCHEMA_VERSION,
    LEGACY_GRPO_CHECKPOINT_FORMAT,
    LEGACY_GRPO_CHECKPOINT_SCHEMA_VERSION,
    load_grpo_a_checkpoint_for_evaluation,
    load_grpo_a_config_for_evaluation,
    load_stage1_a_for_grpo,
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
_LEGACY_GRPO_OPEN_REWARD_APPLICATION_CONTRACT = {
    "version": "stage2_grpo_open_application_v1",
    "policy_sample_domain": "tau_d",
    "policy_probability_domain": "tau_d",
    "reward_input_domain": "tau_d",
    "candidate_selection_domain": "tau_d",
    "execution_input_domain": "tau_cmd",
    "execution_transform": "KinematicTrajectoryOptimizer(selected_tau_d)",
    "optimize_only_selected_candidate": True,
    "optimizer_must_succeed_before_policy_update": True,
    "tracking_expansion_enabled": False,
    "calibration_required": False,
    "best_checkpoint_metric": "validation/raw_proxy_reward_mean",
    "simulator_validation_role": "diagnostic_only",
}
_LEGACY_GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        _LEGACY_GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _configure_deterministic_inference(device: torch.device) -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RewardComparisonError(
            "reward comparison requires PYTHONHASHSEED=0 at process startup"
        )
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _surrounding_vehicles(env: object) -> tuple[object, ...]:
    seen: set[int] = set()
    vehicles: list[object] = []
    agents = getattr(env, "agents", {}) or {}
    agent_values = agents.values() if isinstance(agents, Mapping) else agents
    engine = getattr(env, "engine", None)
    traffic_manager = getattr(engine, "traffic_manager", None)
    traffic = getattr(traffic_manager, "traffic_vehicles", None)
    if traffic is None:
        traffic = getattr(traffic_manager, "_traffic_vehicles", ()) or ()
    try:
        traffic_values = tuple(traffic)
    except TypeError:
        traffic_values = ()
    for vehicle in (*tuple(agent_values), *traffic_values):
        if vehicle is None or id(vehicle) in seen:
            continue
        seen.add(id(vehicle))
        vehicles.append(vehicle)
    return tuple(vehicles)


def _initial_state_signature(env: object) -> np.ndarray:
    rows = []
    for agent_id in AGENT_IDS:
        if agent_id not in env.agents:
            raise RewardComparisonError(
                f"initial state is missing required agent {agent_id}"
            )
        vehicle = env.agents[agent_id]
        position = np.asarray(vehicle.position, dtype=np.float64).reshape(-1)
        rows.append(
            (
                float(position[0]),
                float(position[1]),
                float(vehicle.heading_theta),
                float(vehicle.speed_km_h),
            )
        )
    signature = np.asarray(rows, dtype=np.float64)
    if signature.shape != (3, 4) or not np.isfinite(signature).all():
        raise RewardComparisonError("initial vehicle state is invalid")
    return signature


def _initial_scene_sha256(env: object) -> str:
    """Hash platoon and background actors without unstable object UUIDs."""

    agent_objects = {
        id(vehicle): agent_id
        for agent_id, vehicle in (getattr(env, "agents", {}) or {}).items()
    }
    rows = []
    engine = getattr(env, "engine", None)
    get_policy = getattr(engine, "get_policy", None)
    for vehicle in _surrounding_vehicles(env):
        position = np.asarray(getattr(vehicle, "position", ()), dtype=np.float64)
        if position.shape[0] < 2 or not np.isfinite(position[:2]).all():
            raise RewardComparisonError("initial scene vehicle position is invalid")
        lane = getattr(vehicle, "lane", None)
        lane_index = getattr(lane, "index", getattr(vehicle, "lane_index", None))
        policy = (
            get_policy(getattr(vehicle, "name", ""))
            if callable(get_policy)
            else None
        )
        rows.append(
            {
                "role": agent_objects.get(id(vehicle), "background"),
                "vehicle_class": type(vehicle).__name__,
                "position": [float(position[0]), float(position[1])],
                "heading": float(getattr(vehicle, "heading_theta", 0.0)),
                "speed_km_h": float(getattr(vehicle, "speed_km_h", 0.0)),
                "lane_index": repr(lane_index),
                "policy_class": type(policy).__name__ if policy is not None else None,
            }
        )
    rows.sort(
        key=lambda row: (
            row["role"],
            row["vehicle_class"],
            row["position"][0],
            row["position"][1],
            row["lane_index"],
        )
    )
    encoded = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_grpo_checkpoint_contract(
    payload: Mapping[str, object],
) -> tuple[str, str, str]:
    required_fields = {
        "schema_version",
        "format",
        "run_mode",
        "reward_contract_version",
        "reward_contract_sha256",
        "reward_config",
        "reward_config_sha256",
        "reward_application_contract",
        "reward_application_contract_sha256",
        "reward_input_domain",
        "candidate_selection_domain",
        "execution_input_domain",
        "best_checkpoint_metric",
        "tracking_expansion_enabled",
        "calibration_required",
        "diagnostic_only",
        "eligible_for_formal_training",
        "scenario_seeds",
        "environment_steps",
        "scenario_contract_sha256",
        "trajectory_optimizer_config",
        "trajectory_optimizer_sha256",
    }
    missing = sorted(required_fields.difference(payload))
    if missing:
        raise RewardComparisonError(
            f"grpo_open checkpoint is missing contract fields: {missing}"
        )
    expected_diagnostic = {
        "run_mode": "smoke",
        "diagnostic_only": True,
        "eligible_for_formal_training": False,
        "calibration_required": False,
    }
    for field, expected in expected_diagnostic.items():
        if payload.get(field) != expected:
            raise RewardComparisonError(
                f"grpo_open diagnostic checkpoint {field} mismatch"
            )

    raw_application = payload.get("reward_application_contract")
    application_sha = payload.get("reward_application_contract_sha256")
    if not isinstance(raw_application, Mapping) or not isinstance(
        application_sha, str
    ):
        raise RewardComparisonError("grpo_open reward application contract is missing")
    canonical_application_sha = hashlib.sha256(
        json.dumps(
            dict(raw_application),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    checkpoint_identity = (
        payload.get("schema_version"),
        payload.get("format"),
    )
    if checkpoint_identity == (
        GRPO_CHECKPOINT_SCHEMA_VERSION,
        GRPO_CHECKPOINT_FORMAT,
    ):
        expected_application = GRPO_OPEN_REWARD_APPLICATION_CONTRACT
        expected_application_sha = GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
    elif checkpoint_identity == (
        LEGACY_GRPO_CHECKPOINT_SCHEMA_VERSION,
        LEGACY_GRPO_CHECKPOINT_FORMAT,
    ):
        expected_application = _LEGACY_GRPO_OPEN_REWARD_APPLICATION_CONTRACT
        expected_application_sha = (
            _LEGACY_GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
        )
    else:
        raise RewardComparisonError("grpo_open checkpoint identity is unsupported")
    if (
        application_sha != canonical_application_sha
        or dict(raw_application) != expected_application
        or application_sha != expected_application_sha
    ):
        raise RewardComparisonError(
            "grpo_open must use the frozen raw-tau_d application contract"
        )
    for field in (
        "reward_input_domain",
        "candidate_selection_domain",
        "execution_input_domain",
        "best_checkpoint_metric",
        "tracking_expansion_enabled",
        "calibration_required",
    ):
        if payload.get(field) != raw_application.get(field):
            raise RewardComparisonError(
                f"grpo_open reward application metadata mismatch: {field}"
            )

    raw_reward_config = payload.get("reward_config")
    try:
        parsed_reward_config = (
            JointRewardConfig(**dict(raw_reward_config))
            if isinstance(raw_reward_config, Mapping)
            else None
        )
    except (TypeError, ValueError, JointRewardError) as exc:
        raise RewardComparisonError("grpo_open reward config is invalid") from exc
    expected_reward_config = JointRewardConfig()
    expected_reward_version = JOINT_REWARD_CONTRACT["version"]
    if (
        parsed_reward_config is None
        or dict(raw_reward_config) != asdict(parsed_reward_config)
        or parsed_reward_config != expected_reward_config
        or payload.get("reward_contract_version") != expected_reward_version
        or payload.get("reward_contract_sha256") != JOINT_REWARD_CONTRACT_SHA256
        or payload.get("reward_config_sha256")
        != joint_reward_config_sha256(parsed_reward_config)
    ):
        raise RewardComparisonError(
            "grpo_open is not bound to the active frozen joint reward config"
        )

    optimizer_config = KinematicTrajectoryOptimizerConfig()
    if (
        payload.get("trajectory_optimizer_config") != asdict(optimizer_config)
        or payload.get("trajectory_optimizer_sha256") != optimizer_config.sha256()
    ):
        raise RewardComparisonError(
            "grpo_open trajectory optimizer contract mismatch"
        )
    if (
        payload.get("scenario_contract_sha256")
        != primary_scenario_contract()["sha256"]
    ):
        raise RewardComparisonError("grpo_open S5-S9 scenario contract mismatch")
    return (
        str(expected_reward_version),
        JOINT_REWARD_CONTRACT_SHA256,
        joint_reward_config_sha256(parsed_reward_config),
    )


def _load_policy(
    spec: ModelSpec,
    *,
    device: torch.device,
    reward_bindings: dict[str, tuple[str, str, str]],
) -> torch.nn.Module:
    if spec.model_id == "stage1_a":
        if spec.kind != "stage1" or spec.variant != "A":
            raise RewardComparisonError("stage1_a model spec is invalid")
        trainer, _, source_sha = load_stage1_a_for_grpo(
            spec.checkpoint,
            device=device,
            allow_diagnostic_source=True,
        )
        if source_sha != spec.checkpoint_sha256:
            raise RewardComparisonError("stage1_a checkpoint SHA mismatch")
    elif spec.model_id == "grpo_open":
        if (
            spec.kind != "grpo"
            or spec.variant != "A"
            or spec.reward_domain != "tau_d"
            or spec.source_checkpoint is None
            or spec.source_checkpoint_sha256 is None
        ):
            raise RewardComparisonError("grpo_open model spec is invalid")
        grpo_config = load_grpo_a_config_for_evaluation(spec.checkpoint)
        trainer, _, source_sha = load_stage1_a_for_grpo(
            spec.source_checkpoint,
            device=device,
            config=grpo_config,
            allow_diagnostic_source=True,
        )
        if source_sha != spec.source_checkpoint_sha256:
            raise RewardComparisonError("grpo_open source Stage1 SHA mismatch")
        payload = load_grpo_a_checkpoint_for_evaluation(
            spec.checkpoint,
            trainer,
            expected_source_stage1_sha256=source_sha,
        )
        reward_bindings[spec.model_id] = _validate_grpo_checkpoint_contract(payload)
    else:
        raise RewardComparisonError(
            "reward comparison supports only stage1_a and grpo_open"
        )
    trainer.planner.eval()
    return trainer.planner


def _validate_common_reward_binding(
    bindings: Mapping[str, tuple[str, str, str]],
) -> dict[str, str] | None:
    if not bindings:
        return None
    if set(bindings) != {"grpo_open"}:
        raise RewardComparisonError(
            "reward comparison requires exactly one grpo_open reward binding"
        )
    version, contract_sha, config_sha = bindings["grpo_open"]
    return {
        "reward_contract_version": version,
        "reward_contract_sha256": contract_sha,
        "reward_config_sha256": config_sha,
    }


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
