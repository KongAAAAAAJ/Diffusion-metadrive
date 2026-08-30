"""Online raw-diffusion proxy-reward joint GRPO training for Variants A/B."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from evaluation.plot_grpo import (
    ADVANTAGE_VECTOR_TAG,
    REWARD_CURVE_TAGS,
    VALIDATION_REWARD_CURVE_TAGS,
    generate_grpo_plots,
)
from evaluation.joint_simulator_branch import (
    JointEpisodeSpec,
    JointSimulatorBranchEvaluator,
    capture_joint_pose_global,
)
from expert_dataset.collect_joint_bev import (
    JointBEVSampleBuilder,
    SensorlessJointBEVPlatoonEnv,
    simulator_decision_dt_s,
)
from models.bev_planner import (
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256,
    JOINT_REWARD_CONTRACT,
    JOINT_REWARD_CONTRACT_SHA256,
    JointGRPOConfig,
    JointRewardConfig,
    JointRewardError,
    JointTrajectoryProxyReward,
    KinematicTrajectoryOptimizer,
    KinematicTrajectoryOptimizerConfig,
    TrajectoryOptimizationError,
    TrajectoryOptimizationResult,
    joint_reward_config_sha256,
)
from models.bev_planner.mode_contract import ModeIndex
from models.decisioner.rule_decisioner import (
    LaneChangeCommitmentError,
    diffusion_mode_feedback_actions,
    hard_valid_modes_by_rule_action,
    joint_proposal_actions,
    make_rule_maker,
    match_joint_action_proposal,
)
from train.bev_joint_grpo import (
    grpo_b_checkpoint_payload,
    grpo_checkpoint_payload,
    load_grpo_b_checkpoint,
    load_grpo_checkpoint,
    load_stage1_a_for_grpo,
    load_stage1_b_for_grpo,
    save_grpo_checkpoint,
)
from train.train_bev_diffusion_stage1 import planner_forward_from_batch
from scenarios.bev_round13_contract import (
    DEVELOPMENT_SEEDS,
    HOLDOUT_SEEDS,
    PRIMARY_S5_S9_SCENARIOS,
    BEVScenarioContractError,
    deterministic_initial_speed_km_h,
    primary_scenario_contract,
    validate_primary_scenario_contract,
)


AGENT_IDS = ("agent0", "agent1", "agent2")


class OnlineGRPOError(RuntimeError):
    """Raised when online raw-domain GRPO violates its contract."""


@dataclass(frozen=True)
class JointGRPOOnlineConfig:
    device: str = "cuda"
    seed: int = 17
    group_size: int = 4
    total_optimizer_steps: int = 5000
    resume_checkpoint: Path | None = None
    scenarios: tuple[tuple[str, str], ...] = PRIMARY_S5_S9_SCENARIOS
    scenario_seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    environment_steps_per_episode: int = 100
    validation_interval_steps: int = 100
    advantage_vector_log_interval_steps: int = 100

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise OnlineGRPOError("online GRPO device must be cpu or cuda")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise OnlineGRPOError("online GRPO seed must be an integer")
        if (
            isinstance(self.group_size, bool)
            or not isinstance(self.group_size, int)
            or self.group_size < 2
        ):
            raise OnlineGRPOError(
                "group_size must be an integer greater than or equal to 2"
            )
        for name in (
            "total_optimizer_steps",
            "environment_steps_per_episode",
            "validation_interval_steps",
            "advantage_vector_log_interval_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise OnlineGRPOError(f"{name} must be a positive integer")
        if not self.scenarios or any(
            len(value) != 2 or not value[0] or not value[1]
            for value in self.scenarios
        ):
            raise OnlineGRPOError("at least one scenario/route pair is required")
        try:
            primary_scenario_contract(self.scenarios)
        except BEVScenarioContractError as exc:
            raise OnlineGRPOError(str(exc)) from exc
        if not self.scenario_seeds or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.scenario_seeds
        ):
            raise OnlineGRPOError("scenario_seeds must contain integers")
        if self.resume_checkpoint is not None:
            object.__setattr__(
                self, "resume_checkpoint", Path(self.resume_checkpoint)
            )


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise OnlineGRPOError("CUDA was requested but is unavailable")
    return torch.device(name)


def model_inputs_to_batch(
    values: object,
    device: torch.device,
    *,
    mode_valid_mask: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    if not hasattr(values, "as_dict"):
        raise OnlineGRPOError("online model inputs must expose as_dict()")
    result = {}
    for name, value in values.as_dict().items():
        array = np.asarray(
            mode_valid_mask
            if name == "mode_valid_mask" and mode_valid_mask is not None
            else value
        )
        if name == "mode_valid_mask" and (
            array.shape != (3, 10) or array.dtype != np.bool_
        ):
            raise OnlineGRPOError(
                "execution mode_valid_mask must be bool [3,10]"
            )
        result[name] = torch.from_numpy(np.array(array, copy=True)).unsqueeze(0).to(
            device
        )
    return result


def execution_mode_valid_mask(
    model_inputs: object,
    *,
    optimizer: KinematicTrajectoryOptimizer | None = None,
) -> np.ndarray:
    """Build the calibrated execution subset of the physical hard mask."""

    for name in ("coarse_trajectories", "ego_state", "mode_valid_mask"):
        if not hasattr(model_inputs, name):
            raise OnlineGRPOError(f"online model inputs are missing {name}")
    transform = optimizer or KinematicTrajectoryOptimizer()
    return transform.execution_mode_valid_mask(
        np.asarray(model_inputs.coarse_trajectories),
        np.asarray(model_inputs.ego_state)[:, 0],
        np.asarray(model_inputs.mode_valid_mask),
    )


def constant_velocity_actions(env: object) -> dict[str, np.ndarray]:
    actions = {}
    for agent_id in AGENT_IDS:
        vehicle = env.agents[agent_id]
        speed = max(0.0, float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6)
        trajectory = np.zeros((8, 3), dtype=np.float32)
        trajectory[:, 0] = speed * np.arange(1, 9, dtype=np.float32) * 0.5
        actions[agent_id] = trajectory
    return actions


def route_following_warmup_actions(
    env: object, builder: JointBEVSampleBuilder
) -> dict[str, np.ndarray]:
    """Label-free route-following warm-up for curved S5--S9 approaches."""
    if not builder.history_ready():
        return constant_velocity_actions(env)
    values = builder.build_model_inputs(env)
    actions = {}
    for role, agent_id in enumerate(AGENT_IDS):
        valid = np.asarray(values.mode_valid_mask[role], dtype=np.bool_)
        preferred = [
            int(ModeIndex.KEEP_HIGH),
            int(ModeIndex.KEEP_MEDIUM),
            int(ModeIndex.KEEP_LOW),
        ]
        mode = next((index for index in preferred if valid[index]), None)
        if mode is None:
            raise OnlineGRPOError(
                f"{agent_id} has no valid KEEP anchor during route warm-up"
            )
        actions[agent_id] = np.array(
            values.coarse_trajectories[role, mode],
            dtype=np.float32,
            copy=True,
        )
    return actions


def joint_trajectory_action(trajectories: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(trajectories)
    if value.shape != (3, 8, 3) or not np.isfinite(value).all():
        raise OnlineGRPOError("selected online action must be finite [3,8,3]")
    return {
        agent_id: np.array(value[role], dtype=np.float32, copy=True, order="C")
        for role, agent_id in enumerate(AGENT_IDS)
    }


def optimize_selected_model_trajectories(
    model_inputs: object,
    raw_trajectories: np.ndarray,
    selected_modes: np.ndarray,
    *,
    optimizer: KinematicTrajectoryOptimizer | None = None,
) -> TrajectoryOptimizationResult:
    """Apply the execution transform after policy sampling.

    The caller retains ``raw_trajectories`` in the GRPO rollout, so DDIM
    replay/log-prob remains defined on the unmodified diffusion action.
    """

    for name in ("coarse_trajectories", "ego_state"):
        if not hasattr(model_inputs, name):
            raise OnlineGRPOError(f"online model inputs are missing {name}")
    transform = optimizer or KinematicTrajectoryOptimizer()
    return transform.optimize(
        raw_trajectories,
        np.asarray(model_inputs.coarse_trajectories),
        np.asarray(model_inputs.ego_state)[:, 0],
        selected_modes,
    )


def optimize_safe_stop_trajectories(
    model_inputs: object,
    *,
    optimizer: KinematicTrajectoryOptimizer | None = None,
) -> TrajectoryOptimizationResult:
    """Project the always-valid fixed STOP anchors for fail-closed execution."""

    if not hasattr(model_inputs, "coarse_trajectories"):
        raise OnlineGRPOError("online model inputs are missing coarse_trajectories")
    coarse = np.asarray(model_inputs.coarse_trajectories)
    raw_stop = np.asarray(coarse[:, int(ModeIndex.STOP)], dtype=np.float32)
    return optimize_selected_model_trajectories(
        model_inputs,
        raw_stop,
        np.full((3,), int(ModeIndex.STOP), dtype=np.int64),
        optimizer=optimizer,
    )


def episode_has_ended(
    terminated: Mapping[str, object],
    truncated: Mapping[str, object],
    info: Mapping[str, object],
) -> bool:
    if bool(terminated.get("__all__", False)) or bool(
        truncated.get("__all__", False)
    ):
        return True
    for agent_id in AGENT_IDS:
        value = info.get(agent_id)
        if not isinstance(value, Mapping):
            continue
        if any(
            bool(value.get(name, False))
            for name in (
                "crash",
                "crash_vehicle",
                "crash_object",
                "crash_building",
                "crash_human",
                "out_of_road",
                "out_of_route",
            )
        ):
            return True
    return False


def _new_env(scenario: tuple[str, str], seed: int) -> object:
    env = SensorlessJointBEVPlatoonEnv(
        {
            "num_agents": 3,
            "traffic_density": 0.0,
            "initial_speed_km_h": deterministic_initial_speed_km_h(
                scenario[0], seed
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


def _new_online_rule_maker(planner: object, env: object) -> object:
    planner_config = getattr(planner, "config", None)
    if getattr(planner_config, "model_version", None) != "v2":
        raise OnlineGRPOError("online GRPO requires a v2 planner")
    rule_maker = make_rule_maker(dict(env.config))
    rule_maker.reset(env, list(AGENT_IDS))
    return rule_maker


def _condition_online_model_inputs(
    rule_maker: object,
    env: object,
    builder: JointBEVSampleBuilder,
    values: object,
) -> tuple[object, object]:
    try:
        hard_modes = hard_valid_modes_by_rule_action(
            AGENT_IDS, np.asarray(values.mode_valid_mask)
        )
        proposal_batch = rule_maker.propose_joint_actions(
            env,
            list(AGENT_IDS),
            getattr(env, "_last_planner_batch", None) or {},
            hard_valid_modes_by_action=hard_modes,
        )
        rule_condition = (
            joint_proposal_actions(proposal_batch.proposals[0], AGENT_IDS)
            if proposal_batch.proposals
            else {agent_id: 0 for agent_id in AGENT_IDS}
        )
    except LaneChangeCommitmentError as exc:
        raise OnlineGRPOError(f"online RuleMaker proposal failed: {exc}") from exc
    return (
        builder.augment_v2_model_inputs(
            env,
            values,
            rule_action_condition=rule_condition,
            rule_formation_state=rule_maker.is_formation_locked,
        ),
        proposal_batch,
    )


def _validate_online_trajectory_controls(
    env: object, trajectories: np.ndarray
) -> None:
    action = joint_trajectory_action(trajectories)
    for agent_id in AGENT_IDS:
        try:
            control = np.asarray(
                env.trajectory_to_control(agent_id, action[agent_id])
            )
        except Exception as exc:
            raise OnlineGRPOError(
                f"trajectory control failed for {agent_id}: {exc}"
            ) from exc
        if not np.isfinite(control).all():
            raise OnlineGRPOError(
                f"trajectory control is non-finite for {agent_id}"
            )


def _finalize_online_rule_action(
    rule_maker: object,
    proposal_batch: object,
    *,
    env: object,
    scenario: tuple[str, str],
    values: object,
    selected_modes: np.ndarray,
    optimization: TrajectoryOptimizationResult,
    optimizer: KinematicTrajectoryOptimizer,
) -> tuple[TrajectoryOptimizationResult, dict[str, int]]:
    diagnostics = {
        "conditioned_rollouts": 0,
        "proposal_match_attempts": 0,
        "proposal_matches": 0,
        "condition_failures": 0,
        "forced_safe_stops": 0,
        "s7_feedback_exception_hits": 0,
    }
    diagnostics["conditioned_rollouts"] = 1
    diagnostics["proposal_match_attempts"] = 1
    try:
        _, feedback_actions, s7_exceptions = diffusion_mode_feedback_actions(
            np.asarray(selected_modes, dtype=np.int64).tolist(),
            AGENT_IDS,
            scenario_id=scenario[0],
            local_route=scenario[1],
        )
        matched = match_joint_action_proposal(
            proposal_batch, feedback_actions, AGENT_IDS
        )
    except LaneChangeCommitmentError as exc:
        raise OnlineGRPOError(
            f"online RuleMaker action matching failed: {exc}"
        ) from exc
    diagnostics["s7_feedback_exception_hits"] = sum(
        int(value) for value in s7_exceptions.values()
    )

    if matched is None:
        diagnostics["condition_failures"] = 1
        diagnostics["forced_safe_stops"] = 1
        coarse = np.asarray(values.coarse_trajectories)
        raw_stop = np.asarray(
            coarse[:, int(ModeIndex.STOP)], dtype=np.float32
        )[None]
        resolved = optimize_selected_model_trajectories(
            values,
            raw_stop,
            np.full((1, 3), int(ModeIndex.STOP), dtype=np.int64),
            optimizer=optimizer,
        )
    else:
        diagnostics["proposal_matches"] = 1
        resolved = optimization

    _validate_online_trajectory_controls(
        env, resolved.optimized_trajectories[0]
    )
    if matched is not None:
        try:
            rule_maker.accept_joint_action(
                proposal_batch.batch_id, matched.proposal_id
            )
        except LaneChangeCommitmentError as exc:
            raise OnlineGRPOError(
                f"online RuleMaker proposal acceptance failed: {exc}"
            ) from exc
    return resolved, diagnostics


def _load_trainer(
    variant: str,
    source_checkpoint: Path,
    device: torch.device,
    *,
    grpo_config: JointGRPOConfig,
    allow_diagnostic_source: bool,
):
    loader = (
        load_stage1_a_for_grpo if variant == "A" else load_stage1_b_for_grpo
    )
    return loader(
        source_checkpoint,
        device=device,
        config=grpo_config,
        allow_diagnostic_source=allow_diagnostic_source,
    )


def _checkpoint_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise OnlineGRPOError(f"unable to read checkpoint: {path}") from exc
    return digest.hexdigest()


def _reward_contract_version() -> str:
    version = JOINT_REWARD_CONTRACT.get("version")
    if not isinstance(version, str) or not version:
        raise OnlineGRPOError("joint reward contract version is invalid")
    return version


def _joint_rewards_are_informative(
    rewards: np.ndarray,
    *,
    group_size: int,
    minimum_span: float = 1e-6,
) -> bool:
    """Whether a sampled group can carry non-zero signed GRPO credit."""

    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size < 2
    ):
        raise OnlineGRPOError(
            "group_size must be an integer greater than or equal to 2"
        )
    values = np.asarray(rewards)
    if (
        values.shape != (group_size,)
        or values.dtype not in (np.float32, np.float64)
    ):
        raise OnlineGRPOError(f"joint proxy rewards must be float [{group_size}]")
    if not np.isfinite(values).all():
        raise OnlineGRPOError("joint proxy rewards must be finite")
    if not math.isfinite(minimum_span) or minimum_span < 0.0:
        raise OnlineGRPOError(
            "minimum reward span must be finite and non-negative"
        )
    return float(np.ptp(values.astype(np.float64, copy=False))) > minimum_span


def _should_record_advantage_vector(
    optimizer_step: int,
    target_steps: int,
    interval_steps: int,
) -> bool:
    for name, value in (
        ("optimizer_step", optimizer_step),
        ("target_steps", target_steps),
        ("interval_steps", interval_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise OnlineGRPOError(f"{name} must be a positive integer")
    if optimizer_step > target_steps:
        raise OnlineGRPOError("optimizer_step cannot exceed target_steps")
    return optimizer_step % interval_steps == 0 or optimizer_step == target_steps


def _write_advantage_vector_summary(
    writer: SummaryWriter,
    advantages: torch.Tensor,
    optimizer_step: int,
    target_steps: int,
    interval_steps: int,
    *,
    group_size: int,
) -> bool:
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size < 2
    ):
        raise OnlineGRPOError(
            "group_size must be an integer greater than or equal to 2"
        )
    if not _should_record_advantage_vector(
        optimizer_step, target_steps, interval_steps
    ):
        return False
    vector = advantages.detach().cpu()
    if (
        vector.dtype != torch.float32
        or tuple(vector.shape) != (1, group_size)
    ):
        raise OnlineGRPOError(
            f"advantage vector must be float32 with shape [1,{group_size}]"
        )
    if not bool(torch.isfinite(vector).all()):
        raise OnlineGRPOError("advantage vector must be finite")
    writer.add_tensor(ADVANTAGE_VECTOR_TAG, vector, optimizer_step)
    return True


def _round_robin_training_buckets(
    scenarios: Sequence[tuple[str, str]], seeds: Sequence[int]
) -> tuple[tuple[tuple[str, str], int], ...]:
    buckets = tuple(
        ((str(scenario[0]), str(scenario[1])), int(seed))
        for scenario in scenarios
        for seed in seeds
    )
    if not buckets:
        raise OnlineGRPOError("online GRPO training schedule is empty")
    return buckets


def _scenario_summary(env: object) -> dict[str, object]:
    orchestrator = getattr(env, "_scenario_orchestrator", None)
    getter = getattr(orchestrator, "get_episode_summary", None)
    if not callable(getter):
        raise OnlineGRPOError("environment has no scenario realization summary")
    value = getter()
    if not isinstance(value, Mapping):
        raise OnlineGRPOError("scenario realization summary is invalid")
    return dict(value)


def _scenario_ready_for_primary_sampling(env: object) -> bool:
    summary = _scenario_summary(env)
    if not bool(summary.get("scenario_realized", False)):
        return False
    scenario_id = str(summary.get("scenario_id", ""))
    if scenario_id == "S5_hard_brake_lead":
        # The adjacent-lane support recipe is realized at reset, before the
        # actual three-second hard-brake event.  It must not make S5 eligible
        # for calibration on its own: require the non-support trigger and the
        # concrete brake profile installed by the hard-brake handler.
        notes = summary.get("scenario_notes", ())
        return bool(summary.get("scenario_triggered", False)) and (
            isinstance(notes, (list, tuple))
            and "lead_brake_profile" in notes
        )
    # S6/S8 background traffic is itself the evaluated hazard.  Sampling an
    # intermediate recipe state would let a new actor appear inside a 4-second
    # simulator branch although the proxy snapshot cannot contain it.  Other
    # primary scenarios contain optional/background coverage recipes whose
    # completion is not part of their hazard realization contract.
    if scenario_id in {
        "S6_background_merge_in",
        "S8_ego_exit_to_ramp",
    }:
        return bool(summary.get("scenario_recipes_complete", False))
    return True


def _next_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    indices = []
    for child in output_root.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            try:
                indices.append(int(child.name[4:]))
            except ValueError:
                continue
    path = output_root / f"run_{max(indices, default=0) + 1}"
    path.mkdir()
    (path / "checkpoints").mkdir()
    return path


def _application_contract_version() -> str:
    version = GRPO_OPEN_REWARD_APPLICATION_CONTRACT.get("version")
    if not isinstance(version, str) or not version:
        raise OnlineGRPOError("GRPO-Open application contract version is invalid")
    return version


def _validate_raw_reward_config(config: JointRewardConfig) -> None:
    """Require the frozen zero-expansion V2 config for raw tau_d scoring."""

    if not isinstance(config, JointRewardConfig):
        raise OnlineGRPOError("raw reward config must be JointRewardConfig")
    if any(
        float(getattr(config, name)) != 0.0
        for name in (
            "tracking_longitudinal_margin_m",
            "tracking_lateral_margin_m",
            "tracking_heading_margin_rad",
        )
    ):
        raise OnlineGRPOError(
            "GRPO-Open tau_d reward requires zero tracking margins"
        )
    default = JointRewardConfig()
    if dataclasses.asdict(config) != dataclasses.asdict(default):
        raise OnlineGRPOError(
            "GRPO-Open tau_d reward config must match the frozen base config"
        )


def _checkpoint_payload(
    *,
    variant: str,
    trainer: object,
    source_sha: str,
    source_payload: Mapping[str, object],
    metrics: Mapping[str, float],
    diagnostic_only: bool,
    run_mode: str,
    reward_config: JointRewardConfig,
    scenario_contract_sha: str,
    scenario_seeds: Sequence[int],
    environment_steps: int,
    best_validation_reward: float,
    best_checkpoint_sha256: str | None,
) -> dict[str, object]:
    _validate_raw_reward_config(reward_config)
    if not math.isfinite(float(best_validation_reward)):
        raise OnlineGRPOError("best validation reward must be finite")
    if best_checkpoint_sha256 is not None:
        if (
            len(best_checkpoint_sha256) != 64
            or best_checkpoint_sha256 != best_checkpoint_sha256.lower()
        ):
            raise OnlineGRPOError("best checkpoint SHA256 is invalid")
        try:
            int(best_checkpoint_sha256, 16)
        except ValueError as exc:
            raise OnlineGRPOError("best checkpoint SHA256 is invalid") from exc
    builder = (
        grpo_checkpoint_payload if variant == "A" else grpo_b_checkpoint_payload
    )
    payload = builder(
        trainer=trainer,
        source_stage1_sha256=source_sha,
        source_stage1_payload=source_payload,
        metrics=metrics,
        diagnostic_only=diagnostic_only,
    )
    payload.update(
        {
            "run_mode": run_mode,
            "reward_contract_version": _reward_contract_version(),
            "reward_contract_sha256": JOINT_REWARD_CONTRACT_SHA256,
            "reward_config": dataclasses.asdict(reward_config),
            "reward_config_sha256": joint_reward_config_sha256(reward_config),
            "reward_application_contract": GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
            "reward_application_contract_sha256": (
                GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
            ),
            "reward_input_domain": "tau_d",
            "candidate_selection_domain": "tau_d",
            "execution_input_domain": "tau_cmd",
            "best_checkpoint_metric": "validation/raw_proxy_reward_mean",
            "tracking_expansion_enabled": False,
            "calibration_required": False,
            "scenario_contract_sha256": scenario_contract_sha,
            "scenario_seeds": [int(value) for value in scenario_seeds],
            "environment_steps": int(environment_steps),
            "best_validation_reward": float(best_validation_reward),
            "best_checkpoint_sha256": best_checkpoint_sha256,
            "trajectory_optimizer_config": dataclasses.asdict(
                KinematicTrajectoryOptimizerConfig()
            ),
            "trajectory_optimizer_sha256": (
                KinematicTrajectoryOptimizerConfig().sha256()
            ),
        }
    )
    return payload


def _validate_online_checkpoint_metadata(
    payload: Mapping[str, object],
    *,
    run_mode: str,
    reward_config: JointRewardConfig,
    scenario_contract_sha: str,
    scenario_seeds: Sequence[int],
) -> None:
    _validate_raw_reward_config(reward_config)
    legacy_fields = (
        "calibration_report_sha256",
        "calibration_gate_bypassed",
        "calibration_report_passed",
        "calibration_blockers",
    )
    if any(name in payload for name in legacy_fields):
        raise OnlineGRPOError(
            "online GRPO checkpoint contains legacy calibration semantics"
        )
    expected = {
        "run_mode": run_mode,
        "diagnostic_only": run_mode != "formal",
        "eligible_for_formal_training": run_mode == "formal",
        "reward_contract_version": _reward_contract_version(),
        "reward_contract_sha256": JOINT_REWARD_CONTRACT_SHA256,
        "reward_config": dataclasses.asdict(reward_config),
        "reward_config_sha256": joint_reward_config_sha256(reward_config),
        "reward_application_contract": GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
        "reward_application_contract_sha256": (
            GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
        ),
        "reward_input_domain": "tau_d",
        "candidate_selection_domain": "tau_d",
        "execution_input_domain": "tau_cmd",
        "best_checkpoint_metric": "validation/raw_proxy_reward_mean",
        "tracking_expansion_enabled": False,
        "calibration_required": False,
        "scenario_contract_sha256": scenario_contract_sha,
        "scenario_seeds": [int(value) for value in scenario_seeds],
        "trajectory_optimizer_config": dataclasses.asdict(
            KinematicTrajectoryOptimizerConfig()
        ),
        "trajectory_optimizer_sha256": (
            KinematicTrajectoryOptimizerConfig().sha256()
        ),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise OnlineGRPOError(f"online GRPO checkpoint {name} mismatch")
    environment_steps = payload.get("environment_steps")
    if (
        isinstance(environment_steps, bool)
        or not isinstance(environment_steps, int)
        or environment_steps < 0
    ):
        raise OnlineGRPOError(
            "online GRPO checkpoint environment_steps is invalid"
        )
    best_reward = payload.get("best_validation_reward")
    if (
        isinstance(best_reward, bool)
        or not isinstance(best_reward, (int, float))
        or not math.isfinite(float(best_reward))
    ):
        raise OnlineGRPOError(
            "online GRPO checkpoint best_validation_reward is invalid"
        )
    best_sha = payload.get("best_checkpoint_sha256")
    if best_sha is not None:
        if (
            not isinstance(best_sha, str)
            or len(best_sha) != 64
            or best_sha != best_sha.lower()
        ):
            raise OnlineGRPOError(
                "online GRPO checkpoint best_checkpoint_sha256 is invalid"
            )
        try:
            int(best_sha, 16)
        except ValueError as exc:
            raise OnlineGRPOError(
                "online GRPO checkpoint best_checkpoint_sha256 is invalid"
            ) from exc


@torch.no_grad()
def _fixed_raw_proxy_and_simulator_validation(
    planner: torch.nn.Module,
    *,
    device: torch.device,
    reward_config: JointRewardConfig,
    scenarios: Sequence[tuple[str, str]],
    seeds: Sequence[int],
) -> tuple[dict[str, float], tuple[dict[str, object], ...]]:
    """Evaluate raw proxy reward and best-effort closed-loop diagnostics."""

    _validate_raw_reward_config(reward_config)
    proxy_backend = JointTrajectoryProxyReward(reward_config)
    evaluator = JointSimulatorBranchEvaluator(reward_config)
    trajectory_optimizer = KinematicTrajectoryOptimizer()
    raw_rewards: list[float] = []
    raw_unsafe_count = 0
    raw_collision_count = 0
    raw_out_count = 0
    simulator_rewards: list[float] = []
    simulator_unsafe_count = 0
    simulator_collision_count = 0
    simulator_out_count = 0
    simulator_errors: list[dict[str, object]] = []
    for scenario in scenarios:
        for seed in seeds:
            env = _new_env(tuple(scenario), int(seed))
            builder = JointBEVSampleBuilder(AGENT_IDS)
            builder.reset()
            rule_maker = _new_online_rule_maker(planner, env)
            prefix: list[dict[str, np.ndarray]] = []
            dt_s = simulator_decision_dt_s(env)
            try:
                for step_index in range(100):
                    builder.capture_state(env, step_index * dt_s)
                    if (
                        builder.history_ready()
                        and _scenario_ready_for_primary_sampling(env)
                    ):
                        break
                    action = route_following_warmup_actions(env, builder)
                    prefix.append(
                        {
                            name: np.array(value, copy=True)
                            for name, value in action.items()
                        }
                    )
                    _, _, terminated, truncated, info = env.step(action)
                    if episode_has_ended(terminated, truncated, info):
                        raise OnlineGRPOError(
                            "fixed validation ended during history warm-up"
                        )
                else:
                    raise OnlineGRPOError(
                        "fixed validation never reached a realized S5--S9 state"
                    )
                values = builder.build_model_inputs(env)
                values, _ = _condition_online_model_inputs(
                    rule_maker, env, builder, values
                )
                execution_mask = execution_mode_valid_mask(
                    values, optimizer=trajectory_optimizer
                )
                batch = model_inputs_to_batch(
                    values,
                    device,
                    mode_valid_mask=execution_mask,
                )
                generator = torch.Generator(device=device)
                generator.manual_seed(int(seed))
                noise = torch.randn(
                    (1, 3, 10, 8, 2),
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )
                output = planner_forward_from_batch(
                    planner, batch, diffusion_noise=noise
                )
                raw_candidate = (
                    output["selected_trajectory"][0]
                    .detach()
                    .cpu()
                    .numpy()[None]
                    .astype(np.float32, copy=False)
                )
                selected_modes = (
                    output["selected_mode"][0]
                    .detach()
                    .cpu()
                    .numpy()[None]
                    .astype(np.int64, copy=False)
                )
                raw_proxy = proxy_backend.score(env, values, raw_candidate)
                raw_rewards.append(float(raw_proxy.rewards[0]))
                raw_unsafe_count += int(raw_proxy.unsafe[0])
                raw_collision_count += int(raw_proxy.collision[0])
                raw_out_count += int(raw_proxy.out_of_drivable[0])
                command_candidate = optimize_selected_model_trajectories(
                    values,
                    raw_candidate,
                    selected_modes,
                    optimizer=trajectory_optimizer,
                ).optimized_trajectories
                spec = JointEpisodeSpec(
                    scenario_id=str(scenario[0]),
                    local_route=str(scenario[1]),
                    seed=int(seed),
                    reference_pose_global=capture_joint_pose_global(env),
                )
            finally:
                env.close()
            try:
                branch = evaluator.evaluate(spec, prefix, command_candidate)
            except Exception as exc:
                simulator_errors.append(
                    {
                        "scenario": str(scenario[0]),
                        "route": str(scenario[1]),
                        "seed": int(seed),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            simulator_rewards.append(float(branch.reward.rewards[0]))
            simulator_unsafe_count += int(branch.reward.unsafe[0])
            simulator_collision_count += int(branch.reward.collision[0])
            simulator_out_count += int(branch.reward.out_of_drivable[0])
    if not raw_rewards:
        raise OnlineGRPOError("fixed validation produced no raw proxy rewards")
    metrics = {
        "validation/raw_proxy_reward_mean": float(np.mean(raw_rewards)),
        "validation/raw_proxy_unsafe_count": float(raw_unsafe_count),
        "validation/raw_proxy_collision_count": float(raw_collision_count),
        "validation/raw_proxy_out_of_road_count": float(raw_out_count),
        "validation/simulator_available": float(not simulator_errors),
        "validation/simulator_success_count": float(len(simulator_rewards)),
        "validation/simulator_failure_count": float(len(simulator_errors)),
    }
    if simulator_rewards:
        metrics.update(
            {
                "validation/simulator_reward_mean": float(
                    np.mean(simulator_rewards)
                ),
                "validation/unsafe_count": float(simulator_unsafe_count),
                "validation/collision_count": float(simulator_collision_count),
                "validation/out_of_road_count": float(simulator_out_count),
            }
        )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise OnlineGRPOError("fixed validation metrics must be finite")
    return metrics, tuple(simulator_errors)


def _validation_reward_comparison_metrics(
    current_validation: Mapping[str, object],
    pretrain_reward: object,
) -> dict[str, float]:
    reward_tag = "validation/raw_proxy_reward_mean"
    if reward_tag not in current_validation:
        raise OnlineGRPOError(
            f"fixed validation is missing required metric {reward_tag}"
        )
    try:
        current_reward = float(current_validation[reward_tag])
        baseline_reward = float(pretrain_reward)
    except (TypeError, ValueError) as exc:
        raise OnlineGRPOError(
            "validation current and pretrain rewards must be finite scalars"
        ) from exc
    if not math.isfinite(current_reward) or not math.isfinite(baseline_reward):
        raise OnlineGRPOError(
            "validation current and pretrain rewards must be finite scalars"
        )
    return {
        "validation/pretrain_reward": baseline_reward,
        "validation/reward_gain": current_reward - baseline_reward,
    }


def _validation_raw_proxy_reward(validation: Mapping[str, object]) -> float:
    """Return the sole best-checkpoint objective as a finite scalar."""

    reward_tag = "validation/raw_proxy_reward_mean"
    if reward_tag not in validation:
        raise OnlineGRPOError(
            f"fixed validation is missing required metric {reward_tag}"
        )
    try:
        reward = float(validation[reward_tag])
    except (TypeError, ValueError) as exc:
        raise OnlineGRPOError(
            "validation raw proxy reward must be a finite scalar"
        ) from exc
    if not math.isfinite(reward):
        raise OnlineGRPOError(
            "validation raw proxy reward must be a finite scalar"
        )
    return reward


def _resume_best_checkpoint_anchor(
    resume_checkpoint: Path,
    resume_payload: Mapping[str, object],
) -> tuple[Path, float]:
    """Resolve and verify the historical raw-reward best checkpoint."""

    best_reward = float(resume_payload["best_validation_reward"])
    best_sha = resume_payload.get("best_checkpoint_sha256")
    if best_sha is None:
        if _validation_raw_proxy_reward(
            resume_payload.get("metrics", {})
        ) != best_reward:
            raise OnlineGRPOError(
                "self-contained best checkpoint reward binding mismatch"
            )
        return Path(resume_checkpoint), best_reward

    best_path = Path(resume_checkpoint).with_name("best.pt")
    if _checkpoint_file_sha256(best_path) != best_sha:
        raise OnlineGRPOError("resume best checkpoint SHA256 mismatch")
    try:
        best_payload = torch.load(
            best_path, map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise OnlineGRPOError(
            f"unable to load resume best checkpoint: {best_path}"
        ) from exc
    if not isinstance(best_payload, Mapping):
        raise OnlineGRPOError("resume best checkpoint must be a mapping")
    binding_fields = (
        "schema_version",
        "format",
        "variant",
        "predecessor_condition",
        "grpo_config",
        "source_stage1_sha256",
        "run_mode",
        "reward_contract_version",
        "reward_contract_sha256",
        "reward_config_sha256",
        "reward_application_contract_sha256",
        "reward_input_domain",
        "candidate_selection_domain",
        "execution_input_domain",
        "best_checkpoint_metric",
        "tracking_expansion_enabled",
        "calibration_required",
        "scenario_contract_sha256",
        "trajectory_optimizer_sha256",
    )
    if any(
        best_payload.get(field) != resume_payload.get(field)
        for field in binding_fields
    ):
        raise OnlineGRPOError("resume best checkpoint contract binding mismatch")
    raw_best_reward = best_payload.get("best_validation_reward")
    if (
        isinstance(raw_best_reward, bool)
        or not isinstance(raw_best_reward, (int, float))
        or not math.isfinite(float(raw_best_reward))
    ):
        raise OnlineGRPOError("resume best checkpoint reward binding mismatch")
    if (
        best_payload.get("best_checkpoint_sha256") is not None
        or float(raw_best_reward) != best_reward
        or _validation_raw_proxy_reward(best_payload.get("metrics", {}))
        != best_reward
    ):
        raise OnlineGRPOError("resume best checkpoint reward binding mismatch")
    return best_path, best_reward


def _score_select_and_optimize_raw_candidates(
    env: object,
    model_inputs: object,
    raw_candidates: np.ndarray,
    sampled_modes: np.ndarray,
    *,
    proxy_backend: JointTrajectoryProxyReward,
    trajectory_optimizer: KinematicTrajectoryOptimizer,
) -> tuple[object, int, TrajectoryOptimizationResult]:
    """Score every tau_d, then optimize only its raw-reward argmax."""

    raw = np.asarray(raw_candidates)
    modes = np.asarray(sampled_modes)
    proxy = proxy_backend.score(env, model_inputs, raw)
    selected_index = int(np.argmax(proxy.rewards))
    optimization = optimize_selected_model_trajectories(
        model_inputs,
        raw[selected_index : selected_index + 1],
        modes[selected_index : selected_index + 1],
        optimizer=trajectory_optimizer,
    )
    return proxy, selected_index, optimization


def run_joint_grpo_training(
    config: JointGRPOOnlineConfig,
    *,
    variant: Literal["A", "B"],
    run_mode: Literal["formal", "smoke"],
    source_checkpoint: Path,
    output_root: Path,
    max_optimizer_steps: int | None = None,
) -> dict[str, object]:
    if variant not in ("A", "B"):
        raise OnlineGRPOError("online GRPO variant must be A or B")
    if run_mode == "smoke":
        if (
            isinstance(max_optimizer_steps, bool)
            or not isinstance(max_optimizer_steps, int)
            or max_optimizer_steps <= 0
        ):
            raise OnlineGRPOError(
                "smoke mode requires a positive max_optimizer_steps"
            )
        target_steps = max_optimizer_steps
    elif run_mode == "formal":
        if max_optimizer_steps is not None:
            raise OnlineGRPOError("formal mode forbids a diagnostic step cap")
        target_steps = config.total_optimizer_steps
    else:
        raise OnlineGRPOError("run_mode must be formal or smoke")

    reward_config = JointRewardConfig()
    _validate_raw_reward_config(reward_config)
    try:
        scenario_contract_sha = validate_primary_scenario_contract(
            primary_scenario_contract(config.scenarios)
        )
    except BEVScenarioContractError as exc:
        raise OnlineGRPOError(str(exc)) from exc
    torch_device = _device(config.device)
    grpo_config = JointGRPOConfig(group_size=config.group_size)
    trainer, source_payload, source_sha = _load_trainer(
        variant,
        Path(source_checkpoint),
        torch_device,
        grpo_config=grpo_config,
        allow_diagnostic_source=run_mode == "smoke",
    )
    if run_mode == "formal" and source_payload.get(
        "eligible_for_formal_training"
    ) is not True:
        raise OnlineGRPOError("formal GRPO requires an eligible Stage 1 source")

    pretrain_validation, pretrain_simulator_errors = (
        _fixed_raw_proxy_and_simulator_validation(
            trainer.planner,
            device=torch_device,
            reward_config=reward_config,
            scenarios=config.scenarios,
            seeds=HOLDOUT_SEEDS,
        )
    )
    pretrain_reward = _validation_raw_proxy_reward(pretrain_validation)
    pretrain_simulator_reward = pretrain_validation.get(
        "validation/simulator_reward_mean"
    )

    checkpoint_loader = (
        load_grpo_checkpoint if variant == "A" else load_grpo_b_checkpoint
    )
    environment_steps = 0
    sampled_rollouts = 0
    uninformative_rollouts = 0
    advantage_vector_record_count = 0
    rule_diagnostics = {
        "conditioned_rollouts": 0,
        "proposal_match_attempts": 0,
        "proposal_matches": 0,
        "condition_failures": 0,
        "forced_safe_stops": 0,
        "s7_feedback_exception_hits": 0,
    }
    last_metrics: dict[str, float] = {}
    resume_best_path: Path | None = None
    best_reward: float | None = None
    if config.resume_checkpoint is not None:
        resume_payload = checkpoint_loader(
            config.resume_checkpoint,
            trainer,
            expected_source_stage1_sha256=source_sha,
        )
        _validate_online_checkpoint_metadata(
            resume_payload,
            run_mode=run_mode,
            reward_config=reward_config,
            scenario_contract_sha=scenario_contract_sha,
            scenario_seeds=config.scenario_seeds,
        )
        environment_steps = int(resume_payload["environment_steps"])
        last_metrics = {
            str(name): float(value)
            for name, value in resume_payload["metrics"].items()
        }
        resume_best_path, best_reward = _resume_best_checkpoint_anchor(
            Path(config.resume_checkpoint), resume_payload
        )
        if trainer.optimizer_step >= target_steps:
            raise OnlineGRPOError(
                "resume checkpoint already reached requested optimizer steps"
            )

    run_start_optimizer_step = trainer.optimizer_step
    run_dir = _next_run_directory(Path(output_root))
    online_config = dataclasses.asdict(config)
    online_config["resume_checkpoint"] = (
        str(config.resume_checkpoint)
        if config.resume_checkpoint is not None
        else None
    )
    frozen = {
        "format": "bev_joint_grpo_online_config_v3",
        "variant": variant,
        "run_mode": run_mode,
        "diagnostic_only": run_mode != "formal",
        "eligible_for_formal_training": run_mode == "formal",
        "online_config": online_config,
        "reward_contract_version": _reward_contract_version(),
        "reward_contract": JOINT_REWARD_CONTRACT,
        "reward_contract_sha256": JOINT_REWARD_CONTRACT_SHA256,
        "reward_config": dataclasses.asdict(reward_config),
        "reward_config_sha256": joint_reward_config_sha256(reward_config),
        "reward_application_contract": GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
        "reward_application_contract_sha256": (
            GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
        ),
        "reward_input_domain": "tau_d",
        "candidate_selection_domain": "tau_d",
        "execution_input_domain": "tau_cmd",
        "best_checkpoint_metric": "validation/raw_proxy_reward_mean",
        "tracking_expansion_enabled": False,
        "calibration_required": False,
        "trajectory_optimizer_config": dataclasses.asdict(
            KinematicTrajectoryOptimizerConfig()
        ),
        "trajectory_optimizer_sha256": (
            KinematicTrajectoryOptimizerConfig().sha256()
        ),
        "scenario_contract": primary_scenario_contract(config.scenarios),
        "scenario_contract_sha256": scenario_contract_sha,
    }
    (run_dir / "config.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics_path = run_dir / "metrics.jsonl"
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    proxy_backend = JointTrajectoryProxyReward(reward_config)
    trajectory_optimizer = KinematicTrajectoryOptimizer()
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(config.seed)
    training_buckets = _round_robin_training_buckets(
        config.scenarios, config.scenario_seeds
    )
    bucket_sample_counts = [0 for _ in training_buckets]
    bucket_update_counts = [0 for _ in training_buckets]
    consecutive_empty_episodes = 0
    best_path = run_dir / "checkpoints" / "best.pt"
    last_path = run_dir / "checkpoints" / "last.pt"
    best_checkpoint_sha256: str | None = None
    if resume_best_path is not None:
        shutil.copyfile(resume_best_path, best_path)
        best_checkpoint_sha256 = _checkpoint_file_sha256(best_path)
    last_validated_step = -1
    simulator_diagnostic_errors: list[dict[str, object]] = [
        {"optimizer_step": 0, "stage1_baseline": True, **dict(value)}
        for value in pretrain_simulator_errors
    ]

    try:
        while trainer.optimizer_step < target_steps:
            bucket_index = sampled_rollouts % len(training_buckets)
            scenario, seed = training_buckets[bucket_index]
            samples_at_episode_start = sampled_rollouts
            env = _new_env(scenario, seed)
            builder = JointBEVSampleBuilder(AGENT_IDS)
            builder.reset()
            rule_maker = _new_online_rule_maker(trainer.planner, env)
            dt_s = simulator_decision_dt_s(env)
            episode_step = 0
            try:
                while (
                    trainer.optimizer_step < target_steps
                    and sampled_rollouts == samples_at_episode_start
                    and episode_step < config.environment_steps_per_episode
                ):
                    builder.capture_state(env, episode_step * dt_s)
                    if not (
                        builder.history_ready()
                        and _scenario_ready_for_primary_sampling(env)
                    ):
                        action = route_following_warmup_actions(env, builder)
                    else:
                        values = builder.build_model_inputs(env)
                        values, proposal_batch = _condition_online_model_inputs(
                            rule_maker, env, builder, values
                        )
                        execution_mask = execution_mode_valid_mask(
                            values, optimizer=trajectory_optimizer
                        )
                        batch = model_inputs_to_batch(
                            values,
                            torch_device,
                            mode_valid_mask=execution_mask,
                        )
                        rollout = trainer.sample_groups(
                            batch, generator=generator
                        )
                        raw_candidates = (
                            rollout.selected_trajectories[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                        sampled_modes = (
                            rollout.sampled_modes[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.int64, copy=False)
                        )
                        try:
                            proxy, selected_index, optimization = (
                                _score_select_and_optimize_raw_candidates(
                                    env,
                                    values,
                                    raw_candidates,
                                    sampled_modes,
                                    proxy_backend=proxy_backend,
                                    trajectory_optimizer=trajectory_optimizer,
                                )
                            )
                        except TrajectoryOptimizationError:
                            np.savez_compressed(
                                run_dir / "trajectory_optimizer_failure.npz",
                                raw_trajectories=raw_candidates,
                                sampled_modes=sampled_modes,
                                coarse_trajectories=values.coarse_trajectories,
                                current_speeds_mps=values.ego_state[:, 0],
                                mode_valid_mask=values.mode_valid_mask,
                                environment_steps=np.asarray(
                                    [environment_steps], dtype=np.int64
                                ),
                                episode_steps=np.asarray(
                                    [episode_step], dtype=np.int64
                                ),
                            )
                            raise
                        optimization, rule_event = _finalize_online_rule_action(
                            rule_maker,
                            proposal_batch,
                            env=env,
                            scenario=scenario,
                            values=values,
                            selected_modes=sampled_modes[selected_index],
                            optimization=optimization,
                            optimizer=trajectory_optimizer,
                        )
                        for name, value in rule_event.items():
                            rule_diagnostics[name] += int(value)
                        sampled_rollouts += 1
                        bucket_sample_counts[bucket_index] += 1
                        informative = _joint_rewards_are_informative(
                            proxy.rewards,
                            group_size=trainer.config.group_size,
                        )
                        if informative:
                            rewards = torch.from_numpy(
                                proxy.rewards.reshape(1, -1)
                            ).to(torch_device)
                            update = trainer.update(rollout, rewards)
                            last_metrics = update.loss.scalar_metrics()
                            last_metrics.update(
                                {
                                    "optimizer_step": float(trainer.optimizer_step),
                                    "environment_steps": float(environment_steps),
                                    "sampled_rollouts": float(sampled_rollouts),
                                    "uninformative_rollouts": float(
                                        uninformative_rollouts
                                    ),
                                    "training_bucket_index": float(bucket_index),
                                    "raw_proxy_reward_mean": float(
                                        proxy.rewards.mean()
                                    ),
                                    "raw_proxy_reward_max": float(
                                        proxy.rewards.max()
                                    ),
                                    "raw_proxy_unsafe_rate": float(
                                        proxy.unsafe.mean()
                                    ),
                                    "selected_raw_candidate_index": float(
                                        selected_index
                                    ),
                                    "gradient_total": float(
                                        update.total_gradient_norm
                                    ),
                                    "selected_trajectory_optimizer_ms": float(
                                        optimization.elapsed_ms
                                    ),
                                    "selected_trajectory_intervention_ade_m": float(
                                        optimization.intervention_ade_m.mean()
                                    ),
                                    "selected_trajectory_intervention_fde_m": float(
                                        optimization.intervention_fde_m.mean()
                                    ),
                                    "selected_raw_trajectory_valid_rate": float(
                                        optimization.raw_valid.mean()
                                    ),
                                    "selected_trajectory_retained_raw_fraction": float(
                                        optimization.retained_raw_fraction.mean()
                                    ),
                                    "selected_optimized_trajectory_valid_rate": float(
                                        optimization.optimized_valid.mean()
                                    ),
                                    **{
                                        f"diagnostic/v2_rule_{name}": float(value)
                                        for name, value in rule_diagnostics.items()
                                    },
                                }
                            )
                            for name, value in update.gradient_norms.items():
                                last_metrics[f"gradient/{name}"] = float(value)
                            bucket_update_counts[bucket_index] += 1
                            if _write_advantage_vector_summary(
                                writer,
                                update.loss.advantages,
                                trainer.optimizer_step,
                                target_steps,
                                config.advantage_vector_log_interval_steps,
                                group_size=trainer.config.group_size,
                            ):
                                advantage_vector_record_count += 1
                            for metric_name, metric_value in last_metrics.items():
                                writer.add_scalar(
                                    metric_name,
                                    metric_value,
                                    trainer.optimizer_step,
                                )
                            with metrics_path.open("a", encoding="utf-8") as stream:
                                stream.write(
                                    json.dumps(last_metrics, sort_keys=True) + "\n"
                                )
                        else:
                            uninformative_rollouts += 1
                            writer.add_scalar(
                                "diagnostic/uninformative_rollouts",
                                float(uninformative_rollouts),
                                environment_steps,
                            )
                        action = joint_trajectory_action(
                            optimization.optimized_trajectories[0]
                        )

                    _, _, terminated, truncated, info = env.step(action)
                    environment_steps += 1
                    episode_step += 1
                    if episode_has_ended(terminated, truncated, info):
                        break
            finally:
                env.close()
            if sampled_rollouts == samples_at_episode_start:
                consecutive_empty_episodes += 1
                if consecutive_empty_episodes >= 3:
                    raise OnlineGRPOError(
                        "three consecutive episodes produced no online state "
                        f"rollout for training bucket {bucket_index}: "
                        f"{scenario[0]}/{scenario[1]} seed={seed}"
                    )
            else:
                consecutive_empty_episodes = 0
            if trainer.optimizer_step != last_validated_step and (
                trainer.optimizer_step == target_steps
                or (
                    trainer.optimizer_step > 0
                    and trainer.optimizer_step
                    % config.validation_interval_steps
                    == 0
                )
            ):
                validation, simulator_errors = (
                    _fixed_raw_proxy_and_simulator_validation(
                        trainer.planner,
                        device=torch_device,
                        reward_config=reward_config,
                        scenarios=config.scenarios,
                        seeds=HOLDOUT_SEEDS,
                    )
                )
                validation.update(
                    _validation_reward_comparison_metrics(
                        validation, pretrain_reward
                    )
                )
                if (
                    pretrain_simulator_reward is not None
                    and "validation/simulator_reward_mean" in validation
                ):
                    simulator_reward = float(
                        validation["validation/simulator_reward_mean"]
                    )
                    baseline = float(pretrain_simulator_reward)
                    if math.isfinite(simulator_reward) and math.isfinite(baseline):
                        validation.update(
                            {
                                "validation/simulator_pretrain_reward": baseline,
                                "validation/simulator_reward_gain": (
                                    simulator_reward - baseline
                                ),
                            }
                        )
                for value in simulator_errors:
                    error_record = {
                        "optimizer_step": int(trainer.optimizer_step),
                        "stage1_baseline": False,
                        **dict(value),
                    }
                    simulator_diagnostic_errors.append(error_record)
                    writer.add_text(
                        "validation/simulator_error",
                        json.dumps(error_record, sort_keys=True),
                        trainer.optimizer_step,
                    )
                last_metrics.update(validation)
                for metric_name, metric_value in validation.items():
                    writer.add_scalar(
                        metric_name, metric_value, trainer.optimizer_step
                    )
                validation_reward = _validation_raw_proxy_reward(validation)
                if best_reward is None or validation_reward > best_reward:
                    best_reward = validation_reward
                    best_checkpoint = _checkpoint_payload(
                        variant=variant,
                        trainer=trainer,
                        source_sha=source_sha,
                        source_payload=source_payload,
                        metrics=last_metrics,
                        diagnostic_only=run_mode != "formal",
                        run_mode=run_mode,
                        reward_config=reward_config,
                        scenario_contract_sha=scenario_contract_sha,
                        scenario_seeds=config.scenario_seeds,
                        environment_steps=environment_steps,
                        best_validation_reward=best_reward,
                        best_checkpoint_sha256=None,
                    )
                    save_grpo_checkpoint(best_path, best_checkpoint)
                    best_checkpoint_sha256 = _checkpoint_file_sha256(best_path)
                assert best_reward is not None
                assert best_checkpoint_sha256 is not None
                checkpoint = _checkpoint_payload(
                    variant=variant,
                    trainer=trainer,
                    source_sha=source_sha,
                    source_payload=source_payload,
                    metrics=last_metrics,
                    diagnostic_only=run_mode != "formal",
                    run_mode=run_mode,
                    reward_config=reward_config,
                    scenario_contract_sha=scenario_contract_sha,
                    scenario_seeds=config.scenario_seeds,
                    environment_steps=environment_steps,
                    best_validation_reward=best_reward,
                    best_checkpoint_sha256=best_checkpoint_sha256,
                )
                save_grpo_checkpoint(last_path, checkpoint)
                last_validated_step = trainer.optimizer_step
    finally:
        writer.close()

    if best_reward is None or best_checkpoint_sha256 is None:
        raise OnlineGRPOError("online GRPO never completed fixed validation")
    payload = _checkpoint_payload(
        variant=variant,
        trainer=trainer,
        source_sha=source_sha,
        source_payload=source_payload,
        metrics=last_metrics,
        diagnostic_only=run_mode != "formal",
        run_mode=run_mode,
        reward_config=reward_config,
        scenario_contract_sha=scenario_contract_sha,
        scenario_seeds=config.scenario_seeds,
        environment_steps=environment_steps,
        best_validation_reward=best_reward,
        best_checkpoint_sha256=best_checkpoint_sha256,
    )
    last_path = save_grpo_checkpoint(last_path, payload)

    restored, _, _ = _load_trainer(
        variant,
        Path(source_checkpoint),
        torch_device,
        grpo_config=grpo_config,
        allow_diagnostic_source=run_mode == "smoke",
    )
    loaded = checkpoint_loader(
        last_path,
        restored,
        expected_source_stage1_sha256=source_sha,
    )
    _validate_online_checkpoint_metadata(
        loaded,
        run_mode=run_mode,
        reward_config=reward_config,
        scenario_contract_sha=scenario_contract_sha,
        scenario_seeds=config.scenario_seeds,
    )

    plot_paths = generate_grpo_plots(
        run_dir / "tb",
        run_dir / "plots",
    )

    report = {
        "format": "bev_joint_grpo_online_report_v3",
        "variant": variant,
        "run_mode": run_mode,
        "diagnostic_only": run_mode != "formal",
        "eligible_for_formal_training": run_mode == "formal",
        "optimizer_steps": trainer.optimizer_step,
        "environment_steps": environment_steps,
        "sampled_rollouts": sampled_rollouts,
        "uninformative_rollouts": uninformative_rollouts,
        "v2_rule_conditioning": {
            "enabled": True,
            **rule_diagnostics,
        },
        "reward_contract_version": _reward_contract_version(),
        "reward_contract_sha256": JOINT_REWARD_CONTRACT_SHA256,
        "reward_config_sha256": joint_reward_config_sha256(reward_config),
        "reward_application_contract": GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
        "reward_application_contract_sha256": (
            GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
        ),
        "reward_input_domain": "tau_d",
        "candidate_selection_domain": "tau_d",
        "execution_input_domain": "tau_cmd",
        "best_checkpoint_metric": "validation/raw_proxy_reward_mean",
        "tracking_expansion_enabled": False,
        "calibration_required": False,
        "simulator_validation_role": "diagnostic_only",
        "simulator_diagnostic_errors": simulator_diagnostic_errors,
        "scenario_contract_sha256": scenario_contract_sha,
        "training_scenarios": [list(value) for value in config.scenarios],
        "training_seeds": [int(value) for value in config.scenario_seeds],
        "training_bucket_updates": [
            {
                "scenario": scenario[0],
                "route": scenario[1],
                "seed": seed,
                "sampled_rollouts": bucket_sample_counts[index],
                "optimizer_updates": bucket_update_counts[index],
            }
            for index, (scenario, seed) in enumerate(training_buckets)
        ],
        "validation_seeds": list(HOLDOUT_SEEDS),
        "last_checkpoint": str(last_path.resolve()),
        "best_checkpoint": str(best_path.resolve()),
        "best_validation_reward": best_reward,
        "metrics": last_metrics,
        "checkpoint_round_trip": True,
        "advantage_vector_logging": {
            "storage": "tensorboard_tensor",
            "tensorboard_tag": ADVANTAGE_VECTOR_TAG,
            "tensorboard_dir": str((run_dir / "tb").resolve()),
            "shape": [1, int(trainer.config.group_size)],
            "interval_optimizer_steps": (
                config.advantage_vector_log_interval_steps
            ),
            "record_count": advantage_vector_record_count,
            "scope": "current_run_only",
            "run_start_optimizer_step": run_start_optimizer_step,
            "group_axis_semantics": (
                "independent_random_sample_slot_without_cross_step_identity"
            ),
            "heatmap": str(plot_paths["advantage_heatmap"].resolve()),
        },
        "training_plots": {
            "reward_curve": str(plot_paths["reward_curve"].resolve()),
            "reward_tags": list(REWARD_CURVE_TAGS),
            "validation_reward_curve": str(
                plot_paths["validation_reward_curve"].resolve()
            ),
            "validation_reward_tags": list(
                VALIDATION_REWARD_CURVE_TAGS
            ),
            "reward_domain": "raw_tau_d",
            "grpo_loss_curve": str(
                plot_paths["grpo_loss_curve"].resolve()
            ),
            "grpo_loss_tags": [
                "loss/total",
                "loss/mode_pg",
                "loss/trajectory_pg",
            ],
            "kl_loss_curve": str(plot_paths["kl_loss_curve"].resolve()),
            "kl_loss_tags": [
                "loss/reference_kl",
                "loss/mode_reference_kl",
                "loss/trajectory_reference_kl",
            ],
            "reference_kl_weighted": False,
            "x_axis": "absolute_optimizer_step",
        },
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _config_from_yaml(path: Path) -> JointGRPOOnlineConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise OnlineGRPOError(f"unable to read online GRPO config: {path}") from exc
    if not isinstance(payload, Mapping):
        raise OnlineGRPOError("online GRPO YAML root must be a mapping")
    online = payload.get("online")
    if not isinstance(online, Mapping):
        raise OnlineGRPOError("online GRPO YAML requires an online mapping")
    scenarios = tuple(
        (str(value["scenario"]), str(value["route"]))
        for value in online.get("scenarios", ())
        if isinstance(value, Mapping)
    )
    return JointGRPOOnlineConfig(
        device=str(online.get("device", "cuda")),
        seed=int(online.get("seed", 17)),
        group_size=online.get("group_size", 4),
        total_optimizer_steps=int(online.get("total_optimizer_steps", 5000)),
        resume_checkpoint=(
            Path(str(online["resume_checkpoint"]))
            if online.get("resume_checkpoint")
            else None
        ),
        scenarios=scenarios or PRIMARY_S5_S9_SCENARIOS,
        scenario_seeds=tuple(
            int(value) for value in online.get("scenario_seeds", DEVELOPMENT_SEEDS)
        ),
        environment_steps_per_episode=int(
            online.get("environment_steps_per_episode", 100)
        ),
        validation_interval_steps=int(
            online.get("validation_interval_steps", 100)
        ),
        advantage_vector_log_interval_steps=online.get(
            "advantage_vector_log_interval_steps", 100
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=("A", "B"), required=True)
    parser.add_argument("--run-mode", choices=("formal", "smoke"), required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-optimizer-steps", type=int)
    arguments = parser.parse_args()
    config = _config_from_yaml(arguments.config)
    report = run_joint_grpo_training(
        config,
        variant=arguments.variant,
        run_mode=arguments.run_mode,
        source_checkpoint=arguments.source_checkpoint,
        output_root=arguments.output_root,
        max_optimizer_steps=arguments.max_optimizer_steps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = [
    "DEVELOPMENT_SEEDS",
    "HOLDOUT_SEEDS",
    "PRIMARY_S5_S9_SCENARIOS",
    "JointGRPOOnlineConfig",
    "OnlineGRPOError",
    "constant_velocity_actions",
    "episode_has_ended",
    "joint_trajectory_action",
    "model_inputs_to_batch",
    "optimize_selected_model_trajectories",
    "run_joint_grpo_training",
]


if __name__ == "__main__":
    raise SystemExit(main())
