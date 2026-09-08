"""Online raw-diffusion proxy-reward joint GRPO training for Variants A/B."""

from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from evaluation.plot_grpo import (
    ACCEPTED_ROLLOUT_AXIS_LABEL,
    ADVANTAGE_VECTOR_TAG,
    FROZEN_PRETRAIN_RAW_PROXY_REWARD_TAG,
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
    DDIMNoiseBundle,
    DDIMTransitionError,
    DEFAULT_DDIM_PATH,
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256,
    JointGRPOConfig,
    JointRewardConfig,
    JointTrajectoryProxyReward,
    KinematicTrajectoryOptimizer,
    KinematicTrajectoryOptimizerConfig,
    TrajectoryOptimizationError,
    TrajectoryOptimizationResult,
    joint_grpo_optimizer_contract,
    joint_grpo_optimizer_contract_sha256,
)
from models.bev_planner.joint_reward import (
    VEHICLE_MODE_REWARD_CONTRACT,
    VEHICLE_MODE_REWARD_CONTRACT_SHA256,
    VehicleModeCounterfactualReward,
    VehicleModePretrainRewardResult,
    VehicleModeRewardResult,
    VehicleModeRewardConfig,
    vehicle_mode_reward_config_sha256,
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
ROLLOUT_COLLECTION_CONTRACT_VERSION = "stage2_joint_grpo_persistent_episode_v7"
BEST_CHECKPOINT_METRIC = (
    "validation/safety_constrained_simulator_reward_gain_trailing3"
)
VALIDATION_STATE_BANK_FORMAT = "bev_joint_grpo_validation_state_bank_v1"
MUTABLE_RUNTIME_CONFIG_PATH = "configs/train/bev_joint_grpo.yaml"


class OnlineGRPOError(RuntimeError):
    """Raised when online raw-domain GRPO violates its contract."""


@dataclass(frozen=True)
class _FixedValidationFrozenEntry:
    """Immutable frozen-policy work reused within one training process."""

    all_mode_trajectories: np.ndarray
    selected_trajectory: np.ndarray
    selected_mode: np.ndarray
    pretrain_reward: VehicleModePretrainRewardResult
    paired_candidates: np.ndarray
    paired_reward: VehicleModeRewardResult


def _performance_summary(
    totals: Mapping[str, float], *, validation_calls: int
) -> dict[str, object]:
    """Return accumulated wall timings without changing checkpoint payloads."""

    return {
        "timing_totals_seconds": {
            str(name): float(value) for name, value in sorted(totals.items())
        },
        "validation_calls": int(validation_calls),
    }


def _implementation_commit() -> str:
    """Return the committed code identity while allowing the runtime YAML."""

    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            (
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.rstrip("\n")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OnlineGRPOError("unable to resolve GRPO implementation commit") from exc
    disallowed = []
    for record in dirty.splitlines():
        status = record[:2]
        path = record[3:] if len(record) >= 4 else ""
        runtime_config_edit = (
            path == MUTABLE_RUNTIME_CONFIG_PATH
            and status in {" M", "M ", "MM"}
        )
        if not runtime_config_edit:
            disallowed.append(record)
    if len(commit) != 40 or disallowed:
        detail = ", ".join(disallowed) if disallowed else "invalid HEAD"
        raise OnlineGRPOError(
            "bounded GRPO diagnostics require committed implementation files; "
            f"disallowed tracked changes: {detail}"
        )
    return commit


def _frozen_pretrain_reward_logging_metadata(
    planner: torch.nn.Module,
) -> dict[str, object]:
    """Describe the per-live-state frozen Stage1 reward diagnostic."""

    planner_config = planner.config
    return {
        "tensorboard_tag": FROZEN_PRETRAIN_RAW_PROXY_REWARD_TAG,
        "metrics_jsonl_field": FROZEN_PRETRAIN_RAW_PROXY_REWARD_TAG,
        "step_axis": "absolute_accepted_update_state",
        "same_live_state_as_current_exploration": True,
        "reward_baseline_trajectory_count": 30,
        "reward_baseline_selection": "all_valid_vehicle_modes",
        "execution_trajectory_count": 3,
        "execution_selection": "valid_mode_masked_argmax_per_vehicle",
        "trajectory_domain": "raw_tau_d",
        "inference": "frozen_stage1_standard_deterministic_ddim",
        "inference_seed": int(planner_config.inference_seed),
        "ddim_path": DEFAULT_DDIM_PATH.as_dict(),
        "fixed_inference_noise": True,
        "advantage_formula_role": "paired_sample_baseline_filter",
        "active_mode_gate_role": "none; signal is derived per sample",
        "environment_action_role": "sole_executed_policy",
    }


@dataclass(frozen=True)
class JointGRPOOnlineConfig:
    device: str = "cuda"
    seed: int = 17
    trajectories_per_mode: int = 48
    total_rollout_groups: int = 100
    resume_checkpoint: Path | None = None
    validation_state_bank: Path = Path(
        "evaluation/artifacts/grpo_validation_state_bank_v1.pt"
    )
    scenarios: tuple[tuple[str, str], ...] = PRIMARY_S5_S9_SCENARIOS
    scenario_seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    environment_steps_per_episode: int = 100
    rollout_groups_per_bucket_visit: int = 10
    rollout_start_offset_max_steps: int = 200
    rollout_start_min_remaining_steps: int = 10
    validation_interval_rollouts: int = 20
    advantage_vector_log_interval_rollouts: int = 10
    max_sampling_attempts_per_state: int = 3
    max_sampling_attempts_multiplier: int = 3

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise OnlineGRPOError("online GRPO device must be cpu or cuda")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise OnlineGRPOError("online GRPO seed must be an integer")
        if (
            isinstance(self.trajectories_per_mode, bool)
            or not isinstance(self.trajectories_per_mode, int)
            or self.trajectories_per_mode < 2
        ):
            raise OnlineGRPOError(
                "trajectories_per_mode must be an integer greater than or " "equal to 2"
            )
        for name in (
            "total_rollout_groups",
            "environment_steps_per_episode",
            "rollout_groups_per_bucket_visit",
            "validation_interval_rollouts",
            "advantage_vector_log_interval_rollouts",
            "max_sampling_attempts_per_state",
            "max_sampling_attempts_multiplier",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise OnlineGRPOError(f"{name} must be a positive integer")
        if (
            isinstance(self.rollout_start_offset_max_steps, bool)
            or not isinstance(self.rollout_start_offset_max_steps, int)
            or self.rollout_start_offset_max_steps < 0
        ):
            raise OnlineGRPOError(
                "rollout_start_offset_max_steps must be a non-negative integer"
            )
        if (
            isinstance(self.rollout_start_min_remaining_steps, bool)
            or not isinstance(self.rollout_start_min_remaining_steps, int)
            or self.rollout_start_min_remaining_steps < 0
        ):
            raise OnlineGRPOError(
                "rollout_start_min_remaining_steps must be a non-negative " "integer"
            )
        if self.environment_steps_per_episode < 10:
            raise OnlineGRPOError("environment_steps_per_episode must be at least 10")
        if (
            self.rollout_start_min_remaining_steps
            < self.rollout_groups_per_bucket_visit
        ):
            raise OnlineGRPOError(
                "rollout_start_min_remaining_steps must be greater than or "
                "equal to rollout_groups_per_bucket_visit"
            )
        if self.environment_steps_per_episode < self.rollout_start_min_remaining_steps:
            raise OnlineGRPOError(
                "environment_steps_per_episode must be greater than or equal "
                "to rollout_start_min_remaining_steps"
            )
        if not self.scenarios or any(
            len(value) != 2 or not value[0] or not value[1] for value in self.scenarios
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
            object.__setattr__(self, "resume_checkpoint", Path(self.resume_checkpoint))
        object.__setattr__(
            self, "validation_state_bank", Path(self.validation_state_bank)
        )


@dataclass(frozen=True)
class JointGRPOTrainingConfig:
    variant: Literal["A", "B"]
    run_mode: Literal["formal", "smoke"]
    source_checkpoint: Path
    online: JointGRPOOnlineConfig

    def __post_init__(self) -> None:
        if self.variant not in ("A", "B"):
            raise OnlineGRPOError("online GRPO variant must be A or B")
        if self.run_mode not in ("formal", "smoke"):
            raise OnlineGRPOError("run_mode must be formal or smoke")
        if not isinstance(self.source_checkpoint, Path):
            raise OnlineGRPOError("source_checkpoint must be a Path")
        if not isinstance(self.online, JointGRPOOnlineConfig):
            raise OnlineGRPOError("online must be a JointGRPOOnlineConfig")


def rollout_collection_contract(
    config: JointGRPOOnlineConfig,
) -> dict[str, object]:
    """Return the exact persistent-episode collection protocol."""

    return {
        "version": ROLLOUT_COLLECTION_CONTRACT_VERSION,
        "budget_unit": "accepted_update_state",
        "attempt_budget_unit": "same_live_state_noise_resample",
        "pretrain_baseline_domain": "raw_tau_d",
        "pretrain_inference_per_live_state": 1,
        "comparison_unit": "vehicle_mode",
        "trajectories_per_mode": int(config.trajectories_per_mode),
        "advantage": "fixed_scale_baseline_relative_safety_truncation",
        "signal_mode": "at_least_one_nonzero_sample_advantage",
        "max_sampling_attempts_per_state": int(config.max_sampling_attempts_per_state),
        "max_sampling_attempts_multiplier": int(
            config.max_sampling_attempts_multiplier
        ),
        "retry_scope": "same_live_state",
        "partial_active_policy": "merge_all_signal_modes_without_backfill",
        "retry_condition": "all_vehicle_modes_have_zero_signal",
        "sampled_candidate_execution": False,
        "environment_action": "cached_frozen_stage1_argmax_only",
        "bucket_order": "ordered_scenario_then_seed",
        "bucket_target_assignment": (
            "balanced_floor_with_remainder_to_lower_bucket_indices"
        ),
        "rollout_groups_per_bucket_visit": int(config.rollout_groups_per_bucket_visit),
        "environment_steps_per_episode": int(config.environment_steps_per_episode),
        "validation_interval_rollouts": int(config.validation_interval_rollouts),
        "baseline_environment_steps_per_live_state": 1,
        "episode_reuse": "persistent_within_bucket_visit",
        "interrupted_visit_resume": "same_bucket_same_visit_progress",
        "checkpoint_boundary": "closed_environment_only",
        "active_environment_serialized": False,
        "rule_maker_commitment_scope": "live_environment_episode",
        "rollout_start_ready_gate": ("history_ready_and_primary_scenario_ready"),
        "rollout_start_offset_distribution": "inclusive_uniform_integer",
        "rollout_start_generator": "independent_rollout_start_torch_generator",
        "rollout_initial_noise": (
            "stateless_generator_derived_from_training_seed_live_state_retry"
        ),
        "ddim_transition_noise": (
            "independent_stateless_generator_derived_from_training_seed_"
            "live_state_retry"
        ),
        "retry_sequence_invariance": (
            "later_live_state_noise_is_independent_of_prior_retry_count"
        ),
        "rollout_start_offset_max_steps": int(config.rollout_start_offset_max_steps),
        "rollout_start_min_remaining_steps": int(
            config.rollout_start_min_remaining_steps
        ),
        "rollout_start_rng_draws": (
            "exactly_one_per_feasible_live_training_episode_including_zero_"
            "upper_bound"
        ),
        "rollout_start_upper_bound": (
            "min(configured_max,episode_cap_minus_t_ready_minus_min_remaining)"
        ),
        "scenario_window_close": {
            "S5_hard_brake_lead": (
                "conflict_evidence.formation_recovered_after_hazard"
            ),
            "S6_background_merge_in": (
                "conflict_evidence.formation_recovered_after_merge"
            ),
            "S7_ego_merge_from_ramp": (
                "route_completion.all_agents_entered_mainline_and_conflict_"
                "evidence.formation_recovered_after_merge"
            ),
            "S8_ego_exit_to_ramp": (
                "route_completion.all_agents_continued_on_exit_ramp_and_"
                "conflict_evidence.formation_recovered_on_ramp"
            ),
            "S9_narrow_channel_negotiation": (
                "route_completion.all_agents_returned_to_original_lane_and_"
                "conflict_evidence.formation_recovered_after_return"
            ),
        },
        "late_target_retry": (
            "same_bucket_same_visit_with_temporary_observed_last_open_upper_" "bound"
        ),
        "partial_visit_policy": (
            "retain_completed_rollouts_and_updates_then_new_episode_fresh_" "offset"
        ),
        "validation_start": "earliest_ready_without_random_offset",
    }


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise OnlineGRPOError("CUDA was requested but is unavailable")
    return torch.device(name)


def _reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


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
            raise OnlineGRPOError("execution mode_valid_mask must be bool [3,10]")
        result[name] = (
            torch.from_numpy(np.array(array, copy=True)).unsqueeze(0).to(device)
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
    if bool(terminated.get("__all__", False)) or bool(truncated.get("__all__", False)):
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
            "initial_speed_km_h": deterministic_initial_speed_km_h(scenario[0], seed),
            "start_seed": int(seed),
            "num_scenarios": 1,
            "scenario_id": str(scenario[0]),
            "local_route": str(scenario[1]),
        }
    )
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


@dataclass(frozen=True)
class _OnlineRuleCondition:
    model_inputs: object
    proposal_batch: object | None
    rule_actions: dict[str, int]
    is_commitment: bool
    committed_execution_id: int | None
    committed_plan_actions: dict[str, int] | None


def _condition_online_model_inputs(
    rule_maker: object,
    env: object,
    builder: JointBEVSampleBuilder,
    values: object,
    *,
    committed_execution_id: int | None,
    committed_plan_actions: Mapping[str, int] | None,
) -> _OnlineRuleCondition:
    proposal_batch = None
    rule_condition: dict[str, int] | None = None
    rule_condition_is_commitment = False
    committed_actions = (
        None
        if committed_plan_actions is None
        else {agent_id: int(committed_plan_actions[agent_id]) for agent_id in AGENT_IDS}
    )
    try:
        hard_modes = hard_valid_modes_by_rule_action(
            AGENT_IDS, np.asarray(values.mode_valid_mask)
        )
        if rule_maker.has_active_lane_change_commitments:
            if committed_execution_id is None or committed_actions is None:
                raise OnlineGRPOError(
                    "online RuleMaker commitment has no execution state"
                )
            rule_maker.advance_committed_execution(
                env,
                list(AGENT_IDS),
                committed_execution_id,
            )
            if rule_maker.has_active_lane_change_commitments:
                rule_condition = {
                    agent_id: int(action)
                    for agent_id, action in (
                        rule_maker.committed_execution_rule_actions(
                            env, committed_actions
                        )
                    ).items()
                }
                rule_condition_is_commitment = True
            else:
                committed_execution_id = None
                committed_actions = None
        if rule_condition is None:
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
    conditioned = builder.augment_v2_model_inputs(
        env,
        values,
        rule_action_condition=rule_condition,
        rule_formation_state=rule_maker.is_formation_locked,
    )
    return _OnlineRuleCondition(
        model_inputs=conditioned,
        proposal_batch=proposal_batch,
        rule_actions={
            agent_id: int(rule_condition[agent_id]) for agent_id in AGENT_IDS
        },
        is_commitment=rule_condition_is_commitment,
        committed_execution_id=committed_execution_id,
        committed_plan_actions=committed_actions,
    )


def build_grpo_validation_state_bank(
    path: Path,
    *,
    scenarios: Sequence[tuple[str, str]] = PRIMARY_S5_S9_SCENARIOS,
    seeds: Sequence[int] = HOLDOUT_SEEDS,
) -> dict[str, object]:
    """Capture the fixed S5--S9 validation states and common DDIM noises."""

    if tuple(tuple(value) for value in scenarios) != tuple(PRIMARY_S5_S9_SCENARIOS):
        raise OnlineGRPOError("validation state bank requires the fixed S5--S9 set")
    if tuple(int(value) for value in seeds) != tuple(HOLDOUT_SEEDS):
        raise OnlineGRPOError("validation state bank requires seeds 31/47")
    records: list[dict[str, object]] = []
    planner_identity = SimpleNamespace(
        config=SimpleNamespace(model_version="v2")
    )
    trajectory_optimizer = KinematicTrajectoryOptimizer()
    for scenario_index, scenario in enumerate(scenarios):
        for seed in seeds:
            env = _new_env(tuple(scenario), int(seed))
            builder = JointBEVSampleBuilder(AGENT_IDS)
            builder.reset()
            rule_maker = _new_online_rule_maker(planner_identity, env)
            prefix: list[dict[str, np.ndarray]] = []
            dt_s = simulator_decision_dt_s(env)
            try:
                for step_index in range(100):
                    builder.capture_state(env, step_index * dt_s)
                    if builder.history_ready() and _scenario_ready_for_primary_sampling(
                        env
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
                            "validation state-bank warm-up ended before readiness"
                        )
                else:
                    raise OnlineGRPOError(
                        "validation state bank never reached a realized state"
                    )
                values = builder.build_model_inputs(env)
                condition = _condition_online_model_inputs(
                    rule_maker,
                    env,
                    builder,
                    values,
                    committed_execution_id=None,
                    committed_plan_actions=None,
                )
                values = condition.model_inputs
                scenario_summary = _scenario_summary(env)
                execution_mask = execution_mode_valid_mask(
                    values, optimizer=trajectory_optimizer
                )
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    10_000_019 + 1_009 * int(seed) + scenario_index
                )
                noise = DDIMNoiseBundle.sample(
                    (1, 3, 10, 8, 2),
                    device=torch.device("cpu"),
                    generator=generator,
                )
                records.append(
                    {
                        "scenario": tuple(scenario),
                        "seed": int(seed),
                        "prefix": prefix,
                        "model_inputs": {
                            name: np.array(value, copy=True)
                            for name, value in values.as_dict().items()
                        },
                        "execution_mode_valid_mask": np.array(
                            execution_mask, copy=True
                        ),
                        "rule_actions": dict(condition.rule_actions),
                        "scenario_state_contract": {
                            "scenario_random_seed": scenario_summary.get(
                                "scenario_random_seed"
                            ),
                            "severity_bucket": scenario_summary.get(
                                "severity_bucket"
                            ),
                            "resolved_scenario_parameters": dict(
                                scenario_summary.get(
                                    "resolved_scenario_parameters", {}
                                )
                            ),
                        },
                        "reference_pose_global": np.array(
                            capture_joint_pose_global(env), copy=True
                        ),
                        "initial_noise": noise.initial_noise.cpu(),
                        "transition_noises": tuple(
                            value.cpu() for value in noise.transition_noises
                        ),
                    }
                )
            finally:
                env.close()
    payload: dict[str, object] = {
        "format": VALIDATION_STATE_BANK_FORMAT,
        "scenarios": [tuple(value) for value in scenarios],
        "seeds": [int(value) for value in seeds],
        "ddim_path": DEFAULT_DDIM_PATH.as_dict(),
        "records": records,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def _load_grpo_validation_state_bank(
    path: Path,
    *,
    scenarios: Sequence[tuple[str, str]],
    seeds: Sequence[int],
) -> tuple[dict[tuple[tuple[str, str], int], Mapping[str, object]], str]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise OnlineGRPOError(
            f"unable to load validation state bank: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise OnlineGRPOError("validation state bank must be an object")
    if (
        payload.get("format") != VALIDATION_STATE_BANK_FORMAT
        or payload.get("ddim_path") != DEFAULT_DDIM_PATH.as_dict()
        or payload.get("scenarios") != [tuple(value) for value in scenarios]
        or payload.get("seeds") != [int(value) for value in seeds]
    ):
        raise OnlineGRPOError("validation state bank contract mismatch")
    raw_records = payload.get("records")
    expected_count = len(scenarios) * len(seeds)
    if not isinstance(raw_records, list) or len(raw_records) != expected_count:
        raise OnlineGRPOError("validation state bank record count mismatch")
    records: dict[tuple[tuple[str, str], int], Mapping[str, object]] = {}
    for record in raw_records:
        if not isinstance(record, Mapping):
            raise OnlineGRPOError("validation state bank record is invalid")
        raw_scenario = record.get("scenario")
        if not isinstance(raw_scenario, (list, tuple)) or len(raw_scenario) != 2:
            raise OnlineGRPOError("validation state bank scenario is invalid")
        key = ((str(raw_scenario[0]), str(raw_scenario[1])), int(record["seed"]))
        if key in records:
            raise OnlineGRPOError("validation state bank has duplicate records")
        records[key] = record
    expected = {
        (tuple(scenario), int(seed)) for scenario in scenarios for seed in seeds
    }
    if set(records) != expected:
        raise OnlineGRPOError("validation state bank scenario/seed matrix mismatch")
    return records, _checkpoint_file_sha256(Path(path))


def _replay_fixed_validation_state(
    record: Mapping[str, object],
    *,
    scenario: tuple[str, str],
    seed: int,
    device: torch.device,
    trajectory_optimizer: KinematicTrajectoryOptimizer,
) -> tuple[
    object,
    object,
    _OnlineRuleCondition,
    object,
    np.ndarray,
    dict[str, Tensor],
    DDIMNoiseBundle,
    list[dict[str, np.ndarray]],
    np.ndarray,
]:
    env = _new_env(scenario, seed)
    builder = JointBEVSampleBuilder(AGENT_IDS)
    builder.reset()
    rule_maker = _new_online_rule_maker(
        SimpleNamespace(config=SimpleNamespace(model_version="v2")), env
    )
    raw_prefix = record.get("prefix")
    if not isinstance(raw_prefix, list):
        env.close()
        raise OnlineGRPOError("validation state bank prefix is invalid")
    prefix: list[dict[str, np.ndarray]] = []
    dt_s = simulator_decision_dt_s(env)
    for step_index, raw_action in enumerate(raw_prefix):
        if not isinstance(raw_action, Mapping):
            env.close()
            raise OnlineGRPOError("validation state bank action is invalid")
        builder.capture_state(env, step_index * dt_s)
        action = {
            str(name): np.asarray(value).copy()
            for name, value in raw_action.items()
        }
        prefix.append(action)
        _, _, terminated, truncated, info = env.step(action)
        if episode_has_ended(terminated, truncated, info):
            env.close()
            raise OnlineGRPOError("fixed validation prefix ended unexpectedly")
    builder.capture_state(env, len(prefix) * dt_s)
    values = builder.build_model_inputs(env)
    condition = _condition_online_model_inputs(
        rule_maker,
        env,
        builder,
        values,
        committed_execution_id=None,
        committed_plan_actions=None,
    )
    values = condition.model_inputs
    stored_inputs = record.get("model_inputs")
    if not isinstance(stored_inputs, Mapping):
        env.close()
        raise OnlineGRPOError("validation state bank inputs are invalid")
    current_inputs = values.as_dict()
    if set(stored_inputs) != set(current_inputs) or any(
        not np.array_equal(np.asarray(stored_inputs[name]), current_inputs[name])
        for name in current_inputs
    ):
        env.close()
        raise OnlineGRPOError("fixed validation replay changed model inputs")
    stored_rule_actions = record.get("rule_actions")
    if dict(condition.rule_actions) != dict(stored_rule_actions):
        env.close()
        raise OnlineGRPOError("fixed validation replay changed RuleMaker condition")
    summary = _scenario_summary(env)
    replayed_scenario_contract = {
        "scenario_random_seed": summary.get("scenario_random_seed"),
        "severity_bucket": summary.get("severity_bucket"),
        "resolved_scenario_parameters": dict(
            summary.get("resolved_scenario_parameters", {})
        ),
    }
    if replayed_scenario_contract != record.get("scenario_state_contract"):
        env.close()
        raise OnlineGRPOError(
            "fixed validation replay changed resolved scenario parameters"
        )
    execution_mask = execution_mode_valid_mask(
        values, optimizer=trajectory_optimizer
    )
    if not np.array_equal(
        execution_mask, np.asarray(record.get("execution_mode_valid_mask"))
    ):
        env.close()
        raise OnlineGRPOError("fixed validation replay changed executable modes")
    batch = model_inputs_to_batch(values, device, mode_valid_mask=execution_mask)
    try:
        bundle = DDIMNoiseBundle(
            initial_noise=record["initial_noise"].to(device=device),
            transition_noises=tuple(
                value.to(device=device) for value in record["transition_noises"]
            ),
        )
        bundle.validate(
            batch["coarse_trajectories"][..., :2].shape, device=device
        )
        reference_pose = np.asarray(record["reference_pose_global"])
    except (KeyError, AttributeError, DDIMTransitionError) as exc:
        env.close()
        raise OnlineGRPOError("validation state bank noise/pose is invalid") from exc
    return (
        env,
        rule_maker,
        condition,
        values,
        execution_mask,
        batch,
        bundle,
        prefix,
        reference_pose,
    )


def _validate_online_trajectory_controls(env: object, trajectories: np.ndarray) -> None:
    action = joint_trajectory_action(trajectories)
    for agent_id in AGENT_IDS:
        try:
            control = np.asarray(env.trajectory_to_control(agent_id, action[agent_id]))
        except Exception as exc:
            raise OnlineGRPOError(
                f"trajectory control failed for {agent_id}: {exc}"
            ) from exc
        if not np.isfinite(control).all():
            raise OnlineGRPOError(f"trajectory control is non-finite for {agent_id}")


def _finalize_online_rule_action(
    rule_maker: object,
    condition: _OnlineRuleCondition,
    *,
    env: object,
    scenario: tuple[str, str],
    values: object,
    selected_modes: np.ndarray,
    optimization: TrajectoryOptimizationResult,
    optimizer: KinematicTrajectoryOptimizer,
) -> tuple[
    TrajectoryOptimizationResult,
    dict[str, int],
    int | None,
    dict[str, int] | None,
]:
    diagnostics = {
        "conditioned_rollouts": 0,
        "proposal_match_attempts": 0,
        "proposal_matches": 0,
        "condition_failures": 0,
        "forced_safe_stops": 0,
        "s7_feedback_exception_hits": 0,
        "commitment_conditioned_rollouts": 0,
        "commitment_feedback_incompatible": 0,
    }
    diagnostics["conditioned_rollouts"] = 1
    diagnostics["proposal_match_attempts"] = int(not condition.is_commitment)
    try:
        _, feedback_actions, s7_exceptions = diffusion_mode_feedback_actions(
            np.asarray(selected_modes, dtype=np.int64).tolist(),
            AGENT_IDS,
            scenario_id=scenario[0],
            local_route=scenario[1],
        )
        if condition.is_commitment:
            diagnostics["commitment_conditioned_rollouts"] = 1
            compatible = all(
                feedback_actions[agent_id] == int(condition.rule_actions[agent_id])
                for agent_id in AGENT_IDS
            )
            matched = None
        else:
            compatible = True
            matched = (
                None
                if condition.proposal_batch is None
                else match_joint_action_proposal(
                    condition.proposal_batch, feedback_actions, AGENT_IDS
                )
            )
    except LaneChangeCommitmentError as exc:
        raise OnlineGRPOError(
            f"online RuleMaker action matching failed: {exc}"
        ) from exc
    diagnostics["s7_feedback_exception_hits"] = sum(
        int(value) for value in s7_exceptions.values()
    )

    if not compatible or (not condition.is_commitment and matched is None):
        diagnostics["condition_failures"] = 1
        diagnostics["forced_safe_stops"] = 1
        diagnostics["commitment_feedback_incompatible"] = int(
            condition.is_commitment and not compatible
        )
        coarse = np.asarray(values.coarse_trajectories)
        raw_stop = np.asarray(coarse[:, int(ModeIndex.STOP)], dtype=np.float32)[None]
        resolved = optimize_selected_model_trajectories(
            values,
            raw_stop,
            np.full((1, 3), int(ModeIndex.STOP), dtype=np.int64),
            optimizer=optimizer,
        )
    else:
        diagnostics["proposal_matches"] = int(not condition.is_commitment)
        resolved = optimization

    _validate_online_trajectory_controls(env, resolved.optimized_trajectories[0])
    committed_execution_id = condition.committed_execution_id
    committed_plan_actions = condition.committed_plan_actions
    if matched is not None:
        if condition.proposal_batch is None:
            raise OnlineGRPOError("matched RuleMaker action has no proposal batch")
        try:
            rule_maker.accept_joint_action(
                condition.proposal_batch.batch_id, matched.proposal_id
            )
        except LaneChangeCommitmentError as exc:
            raise OnlineGRPOError(
                f"online RuleMaker proposal acceptance failed: {exc}"
            ) from exc
        if rule_maker.has_active_lane_change_commitments:
            committed_execution_id = int(condition.proposal_batch.batch_id)
            committed_plan_actions = joint_proposal_actions(matched, AGENT_IDS)
    return (
        resolved,
        diagnostics,
        committed_execution_id,
        committed_plan_actions,
    )


def execute_cached_frozen_baseline(
    *,
    env: object,
    rule_maker: object,
    condition: _OnlineRuleCondition,
    scenario: tuple[str, str],
    model_inputs: object,
    frozen_raw_trajectories: np.ndarray,
    frozen_selected_modes: np.ndarray,
    optimizer: KinematicTrajectoryOptimizer,
) -> tuple[
    tuple[object, object, object, object, object],
    TrajectoryOptimizationResult,
    dict[str, int],
    int | None,
    dict[str, int] | None,
]:
    """Execute one cached frozen Stage-1 action and advance the env once.

    The sampled GRPO candidates are deliberately absent from this interface,
    making the online execution boundary auditable and hard to misuse.
    """

    raw = np.asarray(frozen_raw_trajectories)
    modes = np.asarray(frozen_selected_modes)
    if raw.shape != (1, 3, 8, 3) or modes.shape != (1, 3):
        raise OnlineGRPOError(
            "cached frozen baseline must be [1,3,8,3] with modes [1,3]"
        )
    optimization = optimize_selected_model_trajectories(
        model_inputs,
        raw,
        modes,
        optimizer=optimizer,
    )
    (
        optimization,
        diagnostics,
        committed_execution_id,
        committed_plan_actions,
    ) = _finalize_online_rule_action(
        rule_maker,
        condition,
        env=env,
        scenario=scenario,
        values=model_inputs,
        selected_modes=modes[0],
        optimization=optimization,
        optimizer=optimizer,
    )
    action = joint_trajectory_action(optimization.optimized_trajectories[0])
    step_result = env.step(action)
    if not isinstance(step_result, tuple) or len(step_result) != 5:
        raise OnlineGRPOError("online env.step must return a five-item tuple")
    return (
        step_result,
        optimization,
        diagnostics,
        committed_execution_id,
        committed_plan_actions,
    )


def _load_trainer(
    variant: str,
    source_checkpoint: Path,
    device: torch.device,
    *,
    grpo_config: JointGRPOConfig,
    allow_diagnostic_source: bool,
):
    loader = load_stage1_a_for_grpo if variant == "A" else load_stage1_b_for_grpo
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
    version = VEHICLE_MODE_REWARD_CONTRACT.get("version")
    if not isinstance(version, str) or not version:
        raise OnlineGRPOError("vehicle-mode reward contract version is invalid")
    return version


def _fixed_scale_reward_signals(
    current_rewards: np.ndarray,
    frozen_rewards: np.ndarray,
    collision: np.ndarray,
    out_of_drivable: np.ndarray,
    valid_mode_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror the fixed-scale torch advantage contract for online gating/logging."""

    values = np.asarray(current_rewards)
    paired = np.asarray(frozen_rewards)
    collision_values = np.asarray(collision)
    out_values = np.asarray(out_of_drivable)
    valid = np.asarray(valid_mode_mask)
    if values.ndim != 3 or values.shape[:2] != (3, 10):
        raise OnlineGRPOError("vehicle-mode rewards must have shape [3,10,N]")
    if values.shape[2] < 2 or values.dtype not in (np.float32, np.float64):
        raise OnlineGRPOError("vehicle-mode rewards must be float [3,10,N>=2]")
    if paired.shape != values.shape or paired.dtype not in (np.float32, np.float64):
        raise OnlineGRPOError("paired frozen rewards must be float [3,10,N]")
    if (
        collision_values.shape != values.shape
        or collision_values.dtype != np.bool_
        or out_values.shape != values.shape
        or out_values.dtype != np.bool_
    ):
        raise OnlineGRPOError("paired safety masks must be bool [3,10,N]")
    if valid.shape != (3, 10) or valid.dtype != np.bool_:
        raise OnlineGRPOError("valid mode mask must be bool [3,10]")
    if not np.isfinite(values).all() or not np.isfinite(paired).all():
        raise OnlineGRPOError("paired vehicle-mode rewards must be finite")
    centered = values - values.mean(axis=-1, keepdims=True)
    unsafe = collision_values | out_values
    advantages = np.where(
        unsafe,
        -1.0,
        np.where(values >= paired - 1e-6, np.maximum(centered, 0.0), 0.0),
    ).astype(np.float32, copy=False)
    advantages *= valid[..., None]
    signal = valid & np.any(advantages != 0.0, axis=-1)
    return centered.astype(np.float32, copy=False), advantages, signal


def _diffusion_attempt_generators(
    *,
    device: torch.device,
    training_seed: int,
    live_state_index: int,
    retry_index: int,
) -> tuple[torch.Generator, torch.Generator]:
    """Create independent reproducible initial/transition noise streams."""

    if min(live_state_index, retry_index) < 0:
        raise OnlineGRPOError("diffusion RNG indices must be non-negative")
    modulus = 2**63 - 1
    state_key = (
        int(training_seed)
        + 1_000_003 * int(live_state_index)
        + 10_007 * int(retry_index)
    ) % modulus
    initial = torch.Generator(device=device)
    transition = torch.Generator(device=device)
    initial.manual_seed((state_key + 2_000_033) % modulus)
    transition.manual_seed((state_key + 4_000_037) % modulus)
    return initial, transition


def _should_record_advantage_vector(
    rollout_group: int,
    target_rollout_groups: int,
    interval_rollouts: int,
) -> bool:
    for name, value in (
        ("rollout_group", rollout_group),
        ("target_rollout_groups", target_rollout_groups),
        ("interval_rollouts", interval_rollouts),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise OnlineGRPOError(f"{name} must be a positive integer")
    if rollout_group > target_rollout_groups:
        raise OnlineGRPOError("rollout_group cannot exceed target_rollout_groups")
    return (
        rollout_group % interval_rollouts == 0 or rollout_group == target_rollout_groups
    )


def _write_advantage_vector_summary(
    writer: SummaryWriter,
    advantages: torch.Tensor,
    rollout_group: int,
    target_rollout_groups: int,
    interval_rollouts: int,
    *,
    trajectories_per_mode: int,
) -> bool:
    if (
        isinstance(trajectories_per_mode, bool)
        or not isinstance(trajectories_per_mode, int)
        or trajectories_per_mode < 2
    ):
        raise OnlineGRPOError(
            "trajectories_per_mode must be an integer greater than or equal " "to 2"
        )
    if not _should_record_advantage_vector(
        rollout_group, target_rollout_groups, interval_rollouts
    ):
        return False
    vector = advantages.detach().cpu()
    if vector.dtype != torch.float32 or tuple(vector.shape) != (
        1,
        3,
        10,
        trajectories_per_mode,
    ):
        raise OnlineGRPOError(
            "advantage tensor must be float32 with shape "
            f"[1,3,10,{trajectories_per_mode}]"
        )
    if not bool(torch.isfinite(vector).all()):
        raise OnlineGRPOError("advantage vector must be finite")
    writer.add_tensor(ADVANTAGE_VECTOR_TAG, vector, rollout_group)
    return True


def _split_loss_metrics_by_step_axis(
    metrics: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    rollout_metrics = {
        name: float(value)
        for name, value in metrics.items()
        if name.startswith("advantage/")
    }
    optimizer_metrics = {
        name: float(value)
        for name, value in metrics.items()
        if not name.startswith("advantage/")
    }
    return rollout_metrics, optimizer_metrics


def _advantage_scalar_metrics(advantages: torch.Tensor) -> dict[str, float]:
    return {
        "advantage/mean": float(advantages.mean().detach().cpu()),
        "advantage/std": float(advantages.std(unbiased=False).detach().cpu()),
        "advantage/min": float(advantages.min().detach().cpu()),
        "advantage/max": float(advantages.max().detach().cpu()),
    }


def _attempt_budget_is_exhausted(
    *,
    accepted_update_states: int,
    target_accepted_update_states: int,
    sampling_attempts: int,
    max_sampling_attempts: int,
) -> bool:
    """Whether collection hit its hard attempt cap before its accepted target."""

    for name, value in (
        ("accepted_update_states", accepted_update_states),
        ("target_accepted_update_states", target_accepted_update_states),
        ("sampling_attempts", sampling_attempts),
        ("max_sampling_attempts", max_sampling_attempts),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise OnlineGRPOError(f"{name} must be a non-negative integer")
    if target_accepted_update_states <= 0:
        raise OnlineGRPOError(
            "target_accepted_update_states must be a positive integer"
        )
    if max_sampling_attempts <= 0:
        raise OnlineGRPOError("max_sampling_attempts must be a positive integer")
    if accepted_update_states > target_accepted_update_states:
        raise OnlineGRPOError("accepted rollout target was exceeded")
    if sampling_attempts < accepted_update_states:
        raise OnlineGRPOError("attempted rollout count is below accepted count")
    return (
        accepted_update_states < target_accepted_update_states
        and sampling_attempts >= max_sampling_attempts
    )


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


def _balanced_bucket_targets(total_rollout_groups: int, bucket_count: int) -> list[int]:
    for name, value in (
        ("total_rollout_groups", total_rollout_groups),
        ("bucket_count", bucket_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise OnlineGRPOError(f"{name} must be a positive integer")
    base, remainder = divmod(total_rollout_groups, bucket_count)
    return [
        base + int(bucket_index < remainder) for bucket_index in range(bucket_count)
    ]


def _next_unfinished_bucket_index(
    bucket_sample_counts: Sequence[int],
    bucket_target_counts: Sequence[int],
    *,
    start_index: int,
) -> int | None:
    if len(bucket_sample_counts) != len(bucket_target_counts) or not (
        bucket_sample_counts
    ):
        raise OnlineGRPOError("training bucket counters are invalid")
    bucket_count = len(bucket_sample_counts)
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or not 0 <= start_index < bucket_count
    ):
        raise OnlineGRPOError("training bucket start cursor is invalid")
    for offset in range(bucket_count):
        index = (start_index + offset) % bucket_count
        if bucket_sample_counts[index] < bucket_target_counts[index]:
            return index
    return None


def _bucket_visit_is_complete(
    *,
    bucket_index: int,
    bucket_sample_counts: Sequence[int],
    bucket_target_counts: Sequence[int],
    current_visit_progress: int,
    rollout_groups_per_bucket_visit: int,
) -> bool:
    return (
        current_visit_progress >= rollout_groups_per_bucket_visit
        or bucket_sample_counts[bucket_index] >= bucket_target_counts[bucket_index]
    )


def _scenario_summary(env: object) -> dict[str, object]:
    orchestrator = getattr(env, "_scenario_orchestrator", None)
    getter = getattr(orchestrator, "get_episode_summary", None)
    if not callable(getter):
        raise OnlineGRPOError("environment has no scenario realization summary")
    value = getter()
    if not isinstance(value, Mapping):
        raise OnlineGRPOError("scenario realization summary is invalid")
    return dict(value)


def _scenario_ready_from_summary(summary: Mapping[str, object]) -> bool:
    """Return whether one already-captured scenario summary is trainable."""

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
            isinstance(notes, (list, tuple)) and "lead_brake_profile" in notes
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


def _scenario_ready_for_primary_sampling(env: object) -> bool:
    return _scenario_ready_from_summary(_scenario_summary(env))


def _scenario_sampling_window_closed_from_summary(
    summary: Mapping[str, object],
) -> bool:
    """Return the scenario-specific, sticky end of the training window."""

    scenario_id = str(summary.get("scenario_id", ""))
    conflict = summary.get("conflict_evidence", {})
    route = summary.get("route_completion", {})
    if not isinstance(conflict, Mapping) or not isinstance(route, Mapping):
        raise OnlineGRPOError("scenario realization summary evidence is invalid")
    if scenario_id == "S5_hard_brake_lead":
        return bool(conflict.get("formation_recovered_after_hazard", False))
    if scenario_id == "S6_background_merge_in":
        return bool(conflict.get("formation_recovered_after_merge", False))
    if scenario_id == "S7_ego_merge_from_ramp":
        return bool(route.get("all_agents_entered_mainline", False)) and bool(
            conflict.get("formation_recovered_after_merge", False)
        )
    if scenario_id == "S8_ego_exit_to_ramp":
        return bool(route.get("all_agents_continued_on_exit_ramp", False)) and bool(
            conflict.get("formation_recovered_on_ramp", False)
        )
    if scenario_id == "S9_narrow_channel_negotiation":
        return bool(route.get("all_agents_returned_to_original_lane", False)) and bool(
            conflict.get("formation_recovered_after_return", False)
        )
    raise OnlineGRPOError(
        f"unsupported primary scenario sampling window: {scenario_id}"
    )


def _scenario_sampling_window_closed(env: object) -> bool:
    return _scenario_sampling_window_closed_from_summary(_scenario_summary(env))


def _rollout_start_offset_upper_bound(
    config: JointGRPOOnlineConfig,
    ready_step: int,
    temporary_max_offset_steps: int | None = None,
) -> int:
    """Return the inclusive feasible offset upper bound for one episode."""

    if (
        isinstance(ready_step, bool)
        or not isinstance(ready_step, int)
        or ready_step < 0
    ):
        raise OnlineGRPOError("rollout ready step must be a non-negative integer")
    if temporary_max_offset_steps is not None and (
        isinstance(temporary_max_offset_steps, bool)
        or not isinstance(temporary_max_offset_steps, int)
        or temporary_max_offset_steps < 0
    ):
        raise OnlineGRPOError(
            "temporary rollout start offset bound must be a non-negative integer"
        )
    remaining_upper = (
        config.environment_steps_per_episode
        - ready_step
        - config.rollout_start_min_remaining_steps
    )
    if remaining_upper < 0:
        raise OnlineGRPOError(
            "no feasible rollout start offset leaves the configured minimum "
            "remaining steps"
        )
    upper = min(config.rollout_start_offset_max_steps, remaining_upper)
    if temporary_max_offset_steps is not None:
        upper = min(upper, temporary_max_offset_steps)
    return int(upper)


def _sample_rollout_start_offset(
    config: JointGRPOOnlineConfig,
    ready_step: int,
    generator: torch.Generator,
    temporary_max_offset_steps: int | None = None,
) -> tuple[int, int]:
    """Draw one inclusive uniform offset from the shared training generator."""

    if not isinstance(generator, torch.Generator):
        raise OnlineGRPOError("rollout start sampling requires a torch.Generator")
    upper = _rollout_start_offset_upper_bound(
        config,
        ready_step,
        temporary_max_offset_steps=temporary_max_offset_steps,
    )
    # torch.randint advances the generator even when the only possible result
    # is zero.  This makes the one-draw-per-feasible-episode contract explicit.
    offset = int(
        torch.randint(
            0,
            upper + 1,
            (1,),
            generator=generator,
            device=generator.device,
            dtype=torch.int64,
        ).item()
    )
    return offset, upper


def _write_rollout_start_event(
    path: Path,
    *,
    optimizer_step: int,
    rollout_group: int,
    bucket_index: int,
    scenario: tuple[str, str],
    seed: int,
    bucket_episode: int,
    ready_step: int,
    upper_bound: int,
    sampled_offset: int,
    target_step: int,
    status: str,
) -> None:
    record = {
        "event": "rollout_start",
        "optimizer_step": int(optimizer_step),
        "rollout_group": int(rollout_group),
        "training_bucket_index": int(bucket_index),
        "scenario": str(scenario[0]),
        "route": str(scenario[1]),
        "seed": int(seed),
        "bucket_episode": int(bucket_episode),
        "t_ready": int(ready_step),
        "feasible_upper_bound": int(upper_bound),
        "sampled_offset": int(sampled_offset),
        "target_step": int(target_step),
        "status": str(status),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _write_dynamic_sampling_attempt_event(
    path: Path,
    *,
    optimizer_step: int,
    accepted_update_state: int,
    sampling_attempt: int,
    retry_index: int,
    bucket_index: int,
    scenario: tuple[str, str],
    seed: int,
    reward_result: VehicleModeRewardResult,
    paired_frozen_rewards: np.ndarray,
    centered_rewards: np.ndarray,
    advantages: np.ndarray,
    hard_valid_mode_mask: np.ndarray,
    signal_mode_mask: np.ndarray,
    reward_config: VehicleModeRewardConfig,
) -> dict[str, float]:
    if not isinstance(reward_result, VehicleModeRewardResult):
        raise OnlineGRPOError("sampling diagnostics require vehicle-mode rewards")
    values = np.asarray(reward_result.rewards, dtype=np.float64)
    baseline = np.asarray(reward_result.pretrain_rewards, dtype=np.float64)
    valid = np.asarray(reward_result.valid_mode_mask, dtype=np.bool_)
    hard_valid = np.asarray(hard_valid_mode_mask, dtype=np.bool_)
    paired = np.asarray(paired_frozen_rewards, dtype=np.float64)
    centered_all = np.asarray(centered_rewards, dtype=np.float64)
    advantages_all = np.asarray(advantages, dtype=np.float64)
    active = np.asarray(signal_mode_mask, dtype=np.bool_)
    if values.ndim != 3 or values.shape[:2] != (3, 10):
        raise OnlineGRPOError("sampling rewards must have shape [3,10,N]")
    if (
        baseline.shape != (3, 10)
        or paired.shape != values.shape
        or centered_all.shape != values.shape
        or advantages_all.shape != values.shape
        or valid.shape != (3, 10)
        or hard_valid.shape != (3, 10)
        or active.shape != (3, 10)
    ):
        raise OnlineGRPOError(
            "sampling pretrain rewards and active mask must be [3,10]"
        )
    if not valid.any() or np.any(valid & ~hard_valid) or np.any(active & ~valid):
        raise OnlineGRPOError("sampling masks contain no valid modes or conflict")
    weights = {
        "progress_score": float(reward_config.progress_weight),
        "gap_penalty": -float(reward_config.gap_weight),
        "ttc_penalty": -float(reward_config.ttc_weight),
        "road_penalty": -float(reward_config.road_weight),
        "comfort_penalty": -float(reward_config.comfort_weight),
    }
    reconstructed = sum(
        coefficient * np.asarray(reward_result.components[name], dtype=np.float64)
        for name, coefficient in weights.items()
    )
    reconstructed -= float(reward_config.collision_penalty) * np.asarray(
        reward_result.collision, dtype=np.float64
    )
    reconstructed -= float(reward_config.out_of_drivable_penalty) * np.asarray(
        reward_result.out_of_drivable, dtype=np.float64
    )
    if not np.allclose(reconstructed[valid], values[valid], atol=2e-5, rtol=1e-5):
        raise OnlineGRPOError("reward components do not reconstruct total reward")

    group_records: list[dict[str, object]] = []
    group_mean_gains = np.zeros((3, 10), dtype=np.float64)
    group_median_gains = np.zeros((3, 10), dtype=np.float64)
    group_fractions = np.zeros((3, 10), dtype=np.float64)
    positive_below_fractions = np.zeros((3, 10), dtype=np.float64)
    for role in range(3):
        for mode in range(10):
            if not valid[role, mode]:
                group_records.append(
                    {
                        "vehicle": role,
                        "mode": mode,
                        "valid": False,
                        "active": False,
                        "inactive_reason": (
                            "optimizer_inexecutable"
                            if hard_valid[role, mode]
                            else "hard_invalid"
                        ),
                    }
                )
                continue
            group = values[role, mode]
            reference = float(baseline[role, mode])
            paired_group = paired[role, mode]
            centered = centered_all[role, mode]
            advantage = advantages_all[role, mode]
            mean_gain = float(np.mean(group - paired_group))
            median_gain = float(np.median(group - paired_group))
            fraction_ge = float(np.mean(group >= paired_group))
            positive_below = float(
                np.mean((advantage > 0.0) & (group < paired_group - 1e-6))
            )
            group_mean_gains[role, mode] = mean_gain
            group_median_gains[role, mode] = median_gain
            group_fractions[role, mode] = fraction_ge
            positive_below_fractions[role, mode] = positive_below
            component_record = {
                name: {
                    "raw_mean": float(
                        np.asarray(reward_result.components[name])[role, mode].mean()
                    ),
                    "weighted_mean": float(
                        coefficient
                        * np.asarray(reward_result.components[name])[
                            role, mode
                        ].mean()
                    ),
                }
                for name, coefficient in weights.items()
            }
            component_record.update(
                {
                    "collision": {
                        "raw_mean": float(
                            np.asarray(reward_result.collision)[role, mode].mean()
                        ),
                        "weighted_mean": float(
                            -reward_config.collision_penalty
                            * np.asarray(reward_result.collision)[role, mode].mean()
                        ),
                    },
                    "out_of_drivable": {
                        "raw_mean": float(
                            np.asarray(reward_result.out_of_drivable)[
                                role, mode
                            ].mean()
                        ),
                        "weighted_mean": float(
                            -reward_config.out_of_drivable_penalty
                            * np.asarray(reward_result.out_of_drivable)[
                                role, mode
                            ].mean()
                        ),
                    },
                }
            )
            group_records.append(
                {
                    "vehicle": role,
                    "mode": mode,
                    "valid": True,
                    "active": bool(active[role, mode]),
                    "inactive_reason": (
                        None
                        if active[role, mode]
                        else "no_nonzero_sample_advantage"
                    ),
                    "pretrain_reward": reference,
                    "paired_frozen_reward_mean": float(paired_group.mean()),
                    "reward_mean": float(group.mean()),
                    "reward_std": float(group.std()),
                    "reward_min": float(group.min()),
                    "reward_max": float(group.max()),
                    "reward_p05": float(np.quantile(group, 0.05)),
                    "reward_p50": float(np.quantile(group, 0.50)),
                    "reward_p95": float(np.quantile(group, 0.95)),
                    "mean_gain": mean_gain,
                    "median_gain": median_gain,
                    "max_gain": float(np.max(group - paired_group)),
                    "fraction_reward_ge_pretrain": fraction_ge,
                    "positive_advantage_below_pretrain_fraction": positive_below,
                    "advantage_min": float(advantage.min()),
                    "advantage_max": float(advantage.max()),
                    "advantage_rms": float(np.sqrt(np.mean(advantage**2))),
                    "collision_rate": float(
                        np.asarray(reward_result.collision)[role, mode].mean()
                    ),
                    "out_of_drivable_rate": float(
                        np.asarray(reward_result.out_of_drivable)[
                            role, mode
                        ].mean()
                    ),
                    "clearance_violation_rate": float(
                        np.asarray(reward_result.clearance_violation)[
                            role, mode
                        ].mean()
                    ),
                    "components": component_record,
                    "minimum_background_gap_m": float(
                        np.asarray(
                            reward_result.components["minimum_background_gap_m"]
                        )[role, mode].min()
                    ),
                    "minimum_teammate_gap_m": float(
                        np.asarray(
                            reward_result.components["minimum_teammate_gap_m"]
                        )[role, mode].min()
                    ),
                    "minimum_road_margin_m": float(
                        np.asarray(
                            reward_result.components["minimum_road_margin_m"]
                        )[role, mode].min()
                    ),
                    "minimum_ttc_s": float(
                        np.asarray(reward_result.components["minimum_ttc_s"])[
                            role, mode
                        ].min()
                    ),
                }
            )
    def hierarchical_mean(
        group_values: np.ndarray, group_mask: np.ndarray
    ) -> float:
        role_means = [
            float(group_values[role][group_mask[role]].mean())
            if bool(group_mask[role].any())
            else 0.0
            for role in range(3)
        ]
        return float(np.mean(role_means))

    valid_values = values[valid]
    paired_valid = paired[valid]
    deltas = valid_values - paired_valid
    reward_group_means = values.mean(axis=-1)
    reward_group_gains = (values - paired).mean(axis=-1)
    metrics = {
        "train/valid_all/vehicle_reward_mean": hierarchical_mean(
            reward_group_means, valid
        ),
        "train/valid_all/same_mode_pretrain_reward_mean": hierarchical_mean(
            baseline, valid
        ),
        "train/valid_all/reward_gain_mean": hierarchical_mean(
            reward_group_gains, valid
        ),
        "train/valid_all/reward_gain_max": float(deltas.max()),
        "train/valid_all/fraction_reward_ge_pretrain": float(
            np.mean(valid_values >= paired_valid)
        ),
        "train/active_only/group_mean_gain": hierarchical_mean(
            group_mean_gains, active
        ),
        "train/active_only/group_median_gain": hierarchical_mean(
            group_median_gains, active
        ),
        "train/active_only/fraction_reward_ge_pretrain": hierarchical_mean(
            group_fractions, active
        ),
        "train/active_only/positive_advantage_below_pretrain_fraction": (
            hierarchical_mean(positive_below_fractions, active)
        ),
        "signal_mode_count": float(active.sum()),
        "no_signal_mode_count": float((valid & ~active).sum()),
        "invalid_or_unexecutable_mode_count": float((~valid).sum()),
        "signal_vehicle_count": float(np.any(active, axis=1).sum()),
        "train/valid_all/paired_reward_gain_mean": float(deltas.mean()),
        "train/valid_all/paired_reward_gain_median": float(np.median(deltas)),
        "train/valid_all/paired_reward_gain_p05": float(np.quantile(deltas, 0.05)),
        "train/valid_all/paired_reward_gain_p95": float(np.quantile(deltas, 0.95)),
        "train/valid_all/positive_fraction": float(
            np.mean(advantages_all[valid] > 0.0)
        ),
        "train/valid_all/baseline_filtered_fraction": float(
            np.mean((centered_all[valid] > 0.0) & (values[valid] < paired_valid - 1e-6))
        ),
        "train/valid_all/collision_negative_fraction": float(
            np.mean(np.asarray(reward_result.collision)[valid])
        ),
        "train/valid_all/out_negative_fraction": float(
            np.mean(np.asarray(reward_result.out_of_drivable)[valid])
        ),
    }
    active_samples = np.broadcast_to(active[..., None], values.shape)
    for name, coefficient in weights.items():
        component = np.asarray(reward_result.components[name], dtype=np.float64)
        component_group_means = component.mean(axis=-1)
        metrics[f"train/valid_all/component/{name}_raw"] = hierarchical_mean(
            component_group_means, valid
        )
        metrics[f"train/valid_all/component/{name}_weighted"] = (
            coefficient * hierarchical_mean(component_group_means, valid)
        )
        if active.any():
            metrics[f"train/active_only/component/{name}_raw"] = hierarchical_mean(
                component_group_means, active
            )
            metrics[f"train/active_only/component/{name}_weighted"] = (
                coefficient * hierarchical_mean(component_group_means, active)
            )
    for name, indicator, coefficient in (
        ("collision", reward_result.collision, -reward_config.collision_penalty),
        (
            "out_of_drivable",
            reward_result.out_of_drivable,
            -reward_config.out_of_drivable_penalty,
        ),
    ):
        component = np.asarray(indicator, dtype=np.float64)
        component_group_means = component.mean(axis=-1)
        metrics[f"train/valid_all/component/{name}_raw"] = hierarchical_mean(
            component_group_means, valid
        )
        metrics[f"train/valid_all/component/{name}_weighted"] = (
            coefficient * hierarchical_mean(component_group_means, valid)
        )
        if active.any():
            metrics[f"train/active_only/component/{name}_raw"] = hierarchical_mean(
                component_group_means, active
            )
            metrics[f"train/active_only/component/{name}_weighted"] = (
                coefficient * hierarchical_mean(component_group_means, active)
            )
    road_margin = np.asarray(
        reward_result.components["minimum_road_margin_m"], dtype=np.float64
    )
    metrics["train/valid_all/minimum_road_margin_m"] = float(
        road_margin[valid].min()
    )
    if active.any():
        metrics["train/active_only/minimum_road_margin_m"] = float(
            road_margin[active_samples].min()
        )
    record = {
        "event": "dynamic_sampling_attempt",
        "optimizer_step": int(optimizer_step),
        "accepted_update_state": int(accepted_update_state),
        "sampling_attempt": int(sampling_attempt),
        "retry_index": int(retry_index),
        "training_bucket_index": int(bucket_index),
        "scenario": str(scenario[0]),
        "route": str(scenario[1]),
        "seed": int(seed),
        "accepted": bool(active.any()),
        "rejection_reason": (None if active.any() else "all_vehicle_modes_no_signal"),
        "vehicle_mode_groups": group_records,
        **metrics,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return metrics


def _write_baseline_execution_event(
    path: Path,
    *,
    optimizer_step: int,
    accepted_update_states: int,
    rejected_sampling_attempts: int,
    stability_guard_rejections: int,
    exhausted_states: int,
    baseline_execution_steps: int,
    environment_steps: int,
    bucket_index: int,
    scenario: tuple[str, str],
    seed: int,
    pretrain_reward_mean: float,
    performance: Mapping[str, float],
) -> None:
    record = {
        "event": "frozen_baseline_execution",
        "optimizer_step": int(optimizer_step),
        "accepted_update_states": int(accepted_update_states),
        "rejected_sampling_attempts": int(rejected_sampling_attempts),
        "stability_guard_rejections": int(stability_guard_rejections),
        "exhausted_states": int(exhausted_states),
        "baseline_execution_steps": int(baseline_execution_steps),
        "environment_steps": int(environment_steps),
        "training_bucket_index": int(bucket_index),
        "scenario": str(scenario[0]),
        "route": str(scenario[1]),
        "seed": int(seed),
        "same_mode_pretrain_reward_mean": float(pretrain_reward_mean),
        **{str(name): float(value) for name, value in performance.items()},
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


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
        raise OnlineGRPOError("GRPO-Open tau_d reward requires zero tracking margins")
    default = JointRewardConfig()
    if dataclasses.asdict(config) != dataclasses.asdict(default):
        raise OnlineGRPOError(
            "GRPO-Open tau_d reward config must match the frozen base config"
        )


def _validate_vehicle_mode_reward_config(
    config: VehicleModeRewardConfig,
    *,
    trajectories_per_mode: int,
) -> None:
    if not isinstance(config, VehicleModeRewardConfig):
        raise OnlineGRPOError("training reward config must be VehicleModeRewardConfig")
    if config.trajectories_per_mode != trajectories_per_mode:
        raise OnlineGRPOError(
            "vehicle-mode reward trajectory count does not match GRPO config"
        )


def _grpo_config_artifact_payload(
    config: JointGRPOConfig,
) -> dict[str, object]:
    return dataclasses.asdict(config)


def _sampler_state(
    *,
    accepted_update_states: int,
    sampling_attempts: int,
    rejected_sampling_attempts: int,
    stability_guard_rejections: int,
    exhausted_states: int,
    baseline_execution_steps: int,
    zero_signal_epochs: int,
    warmup_environment_steps: int,
    bucket_target_counts: Sequence[int],
    bucket_accepted_update_counts: Sequence[int],
    bucket_sampling_attempt_counts: Sequence[int],
    bucket_rejected_sampling_attempt_counts: Sequence[int],
    bucket_stability_guard_rejection_counts: Sequence[int],
    bucket_exhausted_state_counts: Sequence[int],
    bucket_baseline_execution_step_counts: Sequence[int],
    bucket_zero_signal_epoch_counts: Sequence[int],
    bucket_optimizer_step_counts: Sequence[int],
    bucket_episode_counts: Sequence[int],
    next_bucket_index: int,
    current_visit_progress: int,
    rollout_start_generator_state: torch.Tensor,
    last_validated_update_state: int,
    rollout_groups_per_bucket_visit: int,
    optimizer_step: int,
    environment_steps: int,
    max_sampling_attempts: int,
) -> dict[str, object]:
    raw = {
        "accepted_update_states": accepted_update_states,
        "sampling_attempts": sampling_attempts,
        "rejected_sampling_attempts": rejected_sampling_attempts,
        "stability_guard_rejections": stability_guard_rejections,
        "exhausted_states": exhausted_states,
        "baseline_execution_steps": baseline_execution_steps,
        "zero_signal_epochs": zero_signal_epochs,
        "warmup_environment_steps": warmup_environment_steps,
        "bucket_target_counts": list(bucket_target_counts),
        "bucket_accepted_update_counts": list(bucket_accepted_update_counts),
        "bucket_sampling_attempt_counts": list(bucket_sampling_attempt_counts),
        "bucket_rejected_sampling_attempt_counts": list(
            bucket_rejected_sampling_attempt_counts
        ),
        "bucket_stability_guard_rejection_counts": list(
            bucket_stability_guard_rejection_counts
        ),
        "bucket_exhausted_state_counts": list(bucket_exhausted_state_counts),
        "bucket_baseline_execution_step_counts": list(
            bucket_baseline_execution_step_counts
        ),
        "bucket_zero_signal_epoch_counts": list(bucket_zero_signal_epoch_counts),
        "bucket_optimizer_step_counts": list(bucket_optimizer_step_counts),
        "bucket_episode_counts": list(bucket_episode_counts),
        "next_bucket_index": next_bucket_index,
        "current_visit_progress": current_visit_progress,
        "generator_states": {
            "rollout_start": rollout_start_generator_state.detach().cpu().clone(),
        },
        "last_validated_update_state": last_validated_update_state,
    }
    return _validate_sampler_state(
        raw,
        bucket_count=len(bucket_accepted_update_counts),
        expected_bucket_target_counts=bucket_target_counts,
        rollout_groups_per_bucket_visit=rollout_groups_per_bucket_visit,
        optimizer_step=optimizer_step,
        environment_steps=environment_steps,
        max_sampling_attempts=max_sampling_attempts,
    )


def _validate_sampler_state(
    raw: object,
    *,
    bucket_count: int,
    expected_bucket_target_counts: Sequence[int],
    rollout_groups_per_bucket_visit: int,
    optimizer_step: int,
    environment_steps: int,
    max_sampling_attempts: int,
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise OnlineGRPOError("online GRPO checkpoint sampler_state is invalid")
    if (
        isinstance(bucket_count, bool)
        or not isinstance(bucket_count, int)
        or bucket_count <= 0
    ):
        raise OnlineGRPOError("training bucket count must be a positive integer")
    raw = dict(raw)
    raw.setdefault("stability_guard_rejections", 0)
    raw.setdefault(
        "bucket_stability_guard_rejection_counts", [0 for _ in range(bucket_count)]
    )
    if (
        isinstance(rollout_groups_per_bucket_visit, bool)
        or not isinstance(rollout_groups_per_bucket_visit, int)
        or rollout_groups_per_bucket_visit <= 0
    ):
        raise OnlineGRPOError("rollout bucket visit quota is invalid")
    for name, minimum in (
        ("accepted_update_states", 0),
        ("sampling_attempts", 0),
        ("rejected_sampling_attempts", 0),
        ("stability_guard_rejections", 0),
        ("exhausted_states", 0),
        ("baseline_execution_steps", 0),
        ("zero_signal_epochs", 0),
        ("warmup_environment_steps", 0),
        ("next_bucket_index", 0),
        ("current_visit_progress", 0),
        ("last_validated_update_state", -1),
    ):
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise OnlineGRPOError(
                f"online GRPO checkpoint sampler_state {name} is invalid"
            )
    accepted = int(raw["accepted_update_states"])
    attempts = int(raw["sampling_attempts"])
    rejected = int(raw["rejected_sampling_attempts"])
    guard_rejected = int(raw["stability_guard_rejections"])
    exhausted = int(raw["exhausted_states"])
    baseline_steps = int(raw["baseline_execution_steps"])
    zero_signal = int(raw["zero_signal_epochs"])
    warmup_steps = int(raw["warmup_environment_steps"])
    next_bucket = int(raw["next_bucket_index"])
    current_visit_progress = int(raw["current_visit_progress"])
    last_validated = int(raw["last_validated_update_state"])
    if attempts != accepted + rejected + guard_rejected or last_validated > accepted:
        raise OnlineGRPOError("online GRPO checkpoint sampler counters conflict")
    if baseline_steps != accepted + exhausted + guard_rejected:
        raise OnlineGRPOError("online GRPO baseline execution counters conflict")
    if (
        isinstance(environment_steps, bool)
        or not isinstance(environment_steps, int)
        or environment_steps < 0
        or environment_steps != warmup_steps + baseline_steps
    ):
        raise OnlineGRPOError(
            "online GRPO checkpoint environment step counters conflict"
        )
    _attempt_budget_is_exhausted(
        accepted_update_states=accepted,
        target_accepted_update_states=sum(
            int(value) for value in expected_bucket_target_counts
        ),
        sampling_attempts=attempts,
        max_sampling_attempts=max_sampling_attempts,
    )
    if next_bucket >= bucket_count:
        raise OnlineGRPOError("online GRPO checkpoint bucket cursor is invalid")

    counts: dict[str, list[int]] = {}
    for name in (
        "bucket_target_counts",
        "bucket_accepted_update_counts",
        "bucket_sampling_attempt_counts",
        "bucket_rejected_sampling_attempt_counts",
        "bucket_stability_guard_rejection_counts",
        "bucket_exhausted_state_counts",
        "bucket_baseline_execution_step_counts",
        "bucket_zero_signal_epoch_counts",
        "bucket_optimizer_step_counts",
        "bucket_episode_counts",
    ):
        values = raw.get(name)
        if (
            not isinstance(values, (list, tuple))
            or len(values) != bucket_count
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            )
        ):
            raise OnlineGRPOError(
                f"online GRPO checkpoint sampler_state {name} is invalid"
            )
        counts[name] = [int(value) for value in values]
    expected_targets = [int(value) for value in expected_bucket_target_counts]
    if (
        len(expected_targets) != bucket_count
        or counts["bucket_target_counts"] != expected_targets
    ):
        raise OnlineGRPOError("online GRPO checkpoint bucket targets mismatch")
    if sum(counts["bucket_accepted_update_counts"]) != accepted:
        raise OnlineGRPOError("online GRPO checkpoint bucket accepts conflict")
    if sum(counts["bucket_sampling_attempt_counts"]) != attempts:
        raise OnlineGRPOError("online GRPO checkpoint bucket attempts conflict")
    if sum(counts["bucket_rejected_sampling_attempt_counts"]) != rejected:
        raise OnlineGRPOError("online GRPO checkpoint bucket rejections conflict")
    if (
        sum(counts["bucket_stability_guard_rejection_counts"])
        != guard_rejected
    ):
        raise OnlineGRPOError(
            "online GRPO checkpoint bucket guard rejections conflict"
        )
    if sum(counts["bucket_exhausted_state_counts"]) != exhausted:
        raise OnlineGRPOError("online GRPO checkpoint bucket exhaustions conflict")
    if sum(counts["bucket_baseline_execution_step_counts"]) != baseline_steps:
        raise OnlineGRPOError("online GRPO checkpoint baseline steps conflict")
    if sum(counts["bucket_zero_signal_epoch_counts"]) != zero_signal:
        raise OnlineGRPOError("online GRPO checkpoint zero-signal epochs conflict")
    if sum(counts["bucket_optimizer_step_counts"]) != optimizer_step:
        raise OnlineGRPOError("online GRPO checkpoint bucket updates conflict")
    if any(
        attempt_count != accepted_count + rejected_count + guard_rejected_count
        for attempt_count, accepted_count, rejected_count, guard_rejected_count in zip(
            counts["bucket_sampling_attempt_counts"],
            counts["bucket_accepted_update_counts"],
            counts["bucket_rejected_sampling_attempt_counts"],
            counts["bucket_stability_guard_rejection_counts"],
        )
    ):
        raise OnlineGRPOError("online GRPO checkpoint bucket attempt counters conflict")
    if any(
        baseline_count != accepted_count + exhausted_count + guard_rejected_count
        for baseline_count, accepted_count, exhausted_count, guard_rejected_count in zip(
            counts["bucket_baseline_execution_step_counts"],
            counts["bucket_accepted_update_counts"],
            counts["bucket_exhausted_state_counts"],
            counts["bucket_stability_guard_rejection_counts"],
        )
    ):
        raise OnlineGRPOError("online GRPO bucket baseline counters conflict")
    if any(
        sampled_count > target_count
        for sampled_count, target_count in zip(
            counts["bucket_accepted_update_counts"],
            counts["bucket_target_counts"],
        )
    ):
        raise OnlineGRPOError("online GRPO checkpoint bucket target exceeded")
    if any(
        (attempt_count > 0 or baseline_count > 0) and episode_count == 0
        for attempt_count, baseline_count, episode_count in zip(
            counts["bucket_sampling_attempt_counts"],
            counts["bucket_baseline_execution_step_counts"],
            counts["bucket_episode_counts"],
        )
    ):
        raise OnlineGRPOError("online GRPO checkpoint bucket episodes conflict")

    unfinished = _next_unfinished_bucket_index(
        counts["bucket_accepted_update_counts"],
        counts["bucket_target_counts"],
        start_index=next_bucket,
    )
    if unfinished is None:
        if next_bucket != 0 or current_visit_progress != 0:
            raise OnlineGRPOError(
                "online GRPO checkpoint completed bucket cursor is invalid"
            )
    else:
        if unfinished != next_bucket:
            raise OnlineGRPOError("online GRPO checkpoint bucket cursor is invalid")
        expected_progress = (
            counts["bucket_accepted_update_counts"][next_bucket]
            % rollout_groups_per_bucket_visit
        )
        if current_visit_progress != expected_progress:
            raise OnlineGRPOError(
                "online GRPO checkpoint current visit progress is invalid"
            )
    for index, (sample_count, target_count) in enumerate(
        zip(
            counts["bucket_accepted_update_counts"],
            counts["bucket_target_counts"],
        )
    ):
        if (
            index != next_bucket
            and sample_count < target_count
            and sample_count % rollout_groups_per_bucket_visit != 0
        ):
            raise OnlineGRPOError(
                "online GRPO checkpoint has multiple partial bucket visits"
            )

    generator_states = raw.get("generator_states")
    if not isinstance(generator_states, Mapping) or set(generator_states) != {
        "rollout_start",
    }:
        raise OnlineGRPOError("online GRPO checkpoint generator_states is invalid")
    for generator_state in generator_states.values():
        if (
            not isinstance(generator_state, torch.Tensor)
            or generator_state.dtype != torch.uint8
            or generator_state.ndim != 1
            or generator_state.numel() == 0
        ):
            raise OnlineGRPOError(
                "online GRPO checkpoint generator_states is invalid"
            )
    return {
        "accepted_update_states": accepted,
        "sampling_attempts": attempts,
        "rejected_sampling_attempts": rejected,
        "stability_guard_rejections": guard_rejected,
        "exhausted_states": exhausted,
        "baseline_execution_steps": baseline_steps,
        "zero_signal_epochs": zero_signal,
        "warmup_environment_steps": warmup_steps,
        **counts,
        "next_bucket_index": next_bucket,
        "current_visit_progress": current_visit_progress,
        "generator_states": {
            name: value.detach().cpu().clone()
            for name, value in generator_states.items()
        },
        "last_validated_update_state": last_validated,
    }


def _validated_selection_history(
    raw: object,
) -> list[dict[str, float]]:
    if not isinstance(raw, (list, tuple)):
        raise OnlineGRPOError("validation selection history must be a sequence")
    expected = {
        "accepted_update_state",
        "simulator_reward_gain",
        "selected_reward_gain",
        "s7_out_delta",
        "safety_eligible",
    }
    checked: list[dict[str, float]] = []
    previous_state = -1.0
    for entry in raw:
        if not isinstance(entry, Mapping) or set(entry) != expected:
            raise OnlineGRPOError("validation selection history entry is invalid")
        values = {name: float(entry[name]) for name in expected}
        if not all(math.isfinite(value) for value in values.values()):
            raise OnlineGRPOError("validation selection history must be finite")
        if (
            values["accepted_update_state"] <= previous_state
            or values["safety_eligible"] not in (0.0, 1.0)
        ):
            raise OnlineGRPOError("validation selection history order is invalid")
        previous_state = values["accepted_update_state"]
        checked.append(values)
    return checked


def _checkpoint_payload(
    *,
    variant: str,
    trainer: object,
    source_sha: str,
    source_payload: Mapping[str, object],
    metrics: Mapping[str, float],
    diagnostic_only: bool,
    run_mode: str,
    reward_config: VehicleModeRewardConfig,
    scenario_contract_sha: str,
    scenario_seeds: Sequence[int],
    environment_steps: int,
    best_validation_reward: float | None,
    best_selected_reward_gain: float | None,
    best_checkpoint_sha256: str | None,
    validation_selection_history: Sequence[Mapping[str, float]],
    collection_contract: Mapping[str, object],
    sampler_state: Mapping[str, object],
) -> dict[str, object]:
    _validate_vehicle_mode_reward_config(
        reward_config,
        trajectories_per_mode=trainer.config.trajectories_per_mode,
    )
    if (best_validation_reward is None) is not (best_selected_reward_gain is None):
        raise OnlineGRPOError("best checkpoint score must be complete or absent")
    if best_validation_reward is not None and (
        not math.isfinite(float(best_validation_reward))
        or not math.isfinite(float(best_selected_reward_gain))
    ):
        raise OnlineGRPOError("best checkpoint score must be finite")
    if best_checkpoint_sha256 is not None and best_validation_reward is None:
        raise OnlineGRPOError("best checkpoint SHA requires an eligible score")
    checked_history = _validated_selection_history(validation_selection_history)
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
    builder = grpo_checkpoint_payload if variant == "A" else grpo_b_checkpoint_payload
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
            "reward_contract_sha256": VEHICLE_MODE_REWARD_CONTRACT_SHA256,
            "reward_config": dataclasses.asdict(reward_config),
            "reward_config_sha256": vehicle_mode_reward_config_sha256(reward_config),
            "reward_application_contract": GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
            "reward_application_contract_sha256": (
                GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
            ),
            "reward_input_domain": "tau_d",
            "training_candidate_domain": "tau_d_all_vehicle_modes",
            "environment_action_source": "cached_frozen_stage1_argmax",
            "execution_input_domain": "tau_cmd",
            "best_checkpoint_metric": BEST_CHECKPOINT_METRIC,
            "joint_reward_role": "historical_and_final_evaluation_only",
            "tracking_expansion_enabled": False,
            "calibration_required": False,
            "scenario_contract_sha256": scenario_contract_sha,
            "scenario_seeds": [int(value) for value in scenario_seeds],
            "environment_steps": int(environment_steps),
            "best_validation_reward": (
                None
                if best_validation_reward is None
                else float(best_validation_reward)
            ),
            "best_selected_reward_gain": (
                None
                if best_selected_reward_gain is None
                else float(best_selected_reward_gain)
            ),
            "best_checkpoint_sha256": best_checkpoint_sha256,
            "validation_selection_history": checked_history,
            "policy_update_contract": joint_grpo_optimizer_contract(),
            "policy_update_contract_sha256": joint_grpo_optimizer_contract_sha256(),
            "rollout_collection_contract": dict(collection_contract),
            "sampler_state": dict(sampler_state),
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
    reward_config: VehicleModeRewardConfig,
    scenario_contract_sha: str,
    scenario_seeds: Sequence[int],
    collection_contract: Mapping[str, object],
    bucket_count: int,
    bucket_target_counts: Sequence[int],
    rollout_groups_per_bucket_visit: int,
    optimizer_step: int,
    max_sampling_attempts: int | None = None,
) -> dict[str, object]:
    _validate_vehicle_mode_reward_config(
        reward_config,
        trajectories_per_mode=int(collection_contract.get("trajectories_per_mode", -1)),
    )
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
        "reward_contract_sha256": VEHICLE_MODE_REWARD_CONTRACT_SHA256,
        "reward_config": dataclasses.asdict(reward_config),
        "reward_config_sha256": vehicle_mode_reward_config_sha256(reward_config),
        "reward_application_contract": GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
        "reward_application_contract_sha256": (
            GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
        ),
        "reward_input_domain": "tau_d",
        "training_candidate_domain": "tau_d_all_vehicle_modes",
        "environment_action_source": "cached_frozen_stage1_argmax",
        "execution_input_domain": "tau_cmd",
        "best_checkpoint_metric": BEST_CHECKPOINT_METRIC,
        "joint_reward_role": "historical_and_final_evaluation_only",
        "tracking_expansion_enabled": False,
        "calibration_required": False,
        "policy_update_contract": joint_grpo_optimizer_contract(),
        "policy_update_contract_sha256": joint_grpo_optimizer_contract_sha256(),
        "rollout_collection_contract": dict(collection_contract),
        "scenario_contract_sha256": scenario_contract_sha,
        "scenario_seeds": [int(value) for value in scenario_seeds],
        "trajectory_optimizer_config": dataclasses.asdict(
            KinematicTrajectoryOptimizerConfig()
        ),
        "trajectory_optimizer_sha256": (KinematicTrajectoryOptimizerConfig().sha256()),
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
        raise OnlineGRPOError("online GRPO checkpoint environment_steps is invalid")
    best_reward = payload.get("best_validation_reward")
    best_selected = payload.get("best_selected_reward_gain")
    if (best_reward is None) is not (best_selected is None):
        raise OnlineGRPOError("online GRPO checkpoint best score is invalid")
    if best_reward is not None and (
        isinstance(best_reward, bool)
        or not isinstance(best_reward, (int, float))
        or not math.isfinite(float(best_reward))
        or isinstance(best_selected, bool)
        or not isinstance(best_selected, (int, float))
        or not math.isfinite(float(best_selected))
    ):
        raise OnlineGRPOError("online GRPO checkpoint best score is invalid")
    best_sha = payload.get("best_checkpoint_sha256")
    if best_sha is not None:
        if best_reward is None:
            raise OnlineGRPOError("online GRPO checkpoint best SHA has no score")
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
    _validated_selection_history(payload.get("validation_selection_history"))
    if max_sampling_attempts is None:
        multiplier = collection_contract.get("max_sampling_attempts_multiplier")
        if isinstance(multiplier, bool) or not isinstance(multiplier, int):
            raise OnlineGRPOError(
                "online GRPO collection attempt multiplier is invalid"
            )
        max_sampling_attempts = multiplier * sum(
            int(value) for value in bucket_target_counts
        )
    return _validate_sampler_state(
        payload.get("sampler_state"),
        bucket_count=bucket_count,
        expected_bucket_target_counts=bucket_target_counts,
        rollout_groups_per_bucket_visit=rollout_groups_per_bucket_visit,
        optimizer_step=optimizer_step,
        environment_steps=environment_steps,
        max_sampling_attempts=max_sampling_attempts,
    )


@torch.no_grad()
def _fixed_raw_proxy_and_simulator_validation(
    trainer: object,
    *,
    device: torch.device,
    reward_config: JointRewardConfig,
    scenarios: Sequence[tuple[str, str]],
    seeds: Sequence[int],
    validation_state_bank: Mapping[
        tuple[tuple[str, str], int], Mapping[str, object]
    ],
    frozen_cache: dict[
        tuple[tuple[str, str], int], _FixedValidationFrozenEntry
    ]
    | None = None,
) -> tuple[dict[str, float], tuple[dict[str, object], ...]]:
    """Evaluate per-vehicle reward; retain joint/simulator diagnostics only."""

    _validate_raw_reward_config(reward_config)
    planner = trainer.planner
    proxy_backend = JointTrajectoryProxyReward(reward_config)
    vehicle_backend = VehicleModeCounterfactualReward(
        VehicleModeRewardConfig(
            trajectories_per_mode=trainer.config.trajectories_per_mode
        )
    )
    single_vehicle_backend = VehicleModeCounterfactualReward(
        VehicleModeRewardConfig(trajectories_per_mode=1)
    )
    evaluator = JointSimulatorBranchEvaluator(reward_config)
    trajectory_optimizer = KinematicTrajectoryOptimizer()
    cache = {} if frozen_cache is None else frozen_cache
    performance: defaultdict[str, float] = defaultdict(float)
    validation_started = time.perf_counter()
    cache_hits = 0
    cache_misses = 0
    raw_rewards: list[float] = []
    raw_unsafe_count = 0
    raw_collision_count = 0
    raw_out_count = 0
    vehicle_rewards: list[float] = []
    pretrain_vehicle_rewards: list[float] = []
    macro_context_vehicle_rewards: list[float] = []
    macro_context_pretrain_rewards: list[float] = []
    paired_n48_reward_means: list[float] = []
    paired_n48_frozen_reward_means: list[float] = []
    paired_n48_gain_means: list[float] = []
    vehicle_unsafe_count = 0
    vehicle_collision_count = 0
    vehicle_out_count = 0
    role_rewards: list[list[float]] = [[], [], []]
    role_unsafe_counts = [0, 0, 0]
    role_collision_counts = [0, 0, 0]
    role_out_counts = [0, 0, 0]
    simulator_rewards: list[float] = []
    simulator_unsafe_count = 0
    simulator_collision_count = 0
    simulator_out_count = 0
    simulator_errors: list[dict[str, object]] = []
    selected_vehicle_rewards: list[float] = []
    selected_pretrain_vehicle_rewards: list[float] = []
    scenario_selected_rewards: dict[str, list[float]] = defaultdict(list)
    scenario_selected_pretrain_rewards: dict[str, list[float]] = defaultdict(list)
    scenario_road_penalties: dict[str, list[float]] = defaultdict(list)
    scenario_pretrain_road_penalties: dict[str, list[float]] = defaultdict(list)
    scenario_road_margins: dict[str, list[float]] = defaultdict(list)
    scenario_pretrain_road_margins: dict[str, list[float]] = defaultdict(list)
    scenario_simulator_rewards: dict[str, list[float]] = defaultdict(list)
    scenario_simulator_collisions: dict[str, int] = defaultdict(int)
    scenario_simulator_outs: dict[str, int] = defaultdict(int)
    for scenario_index, scenario in enumerate(scenarios):
        for seed in seeds:
            cache_key = (tuple(scenario), int(seed))
            record = validation_state_bank.get(cache_key)
            if record is None:
                raise OnlineGRPOError(
                    "validation state bank is missing a scenario/seed record"
                )
            replay_started = time.perf_counter()
            (
                env,
                rule_maker,
                condition,
                values,
                execution_mask,
                batch,
                noise_bundle,
                _,
                reference_pose,
            ) = _replay_fixed_validation_state(
                record,
                scenario=tuple(scenario),
                seed=int(seed),
                device=device,
                trajectory_optimizer=trajectory_optimizer,
            )
            performance["perf/validation/state_replay_seconds"] += (
                time.perf_counter() - replay_started
            )
            try:
                current_inference_started = time.perf_counter()
                output = planner_forward_from_batch(
                    planner, batch, ddim_noise_bundle=noise_bundle
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
                current_all_modes = (
                    output["trajectory_candidates"][0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                performance["perf/validation/current_inference_seconds"] += (
                    time.perf_counter() - current_inference_started
                )
                cached = cache.get(cache_key)
                if cached is None:
                    cache_misses += 1
                    frozen_inference_started = time.perf_counter()
                    frozen = trainer.infer_frozen_pretrain_from_inputs(
                        batch, noise_bundle=noise_bundle
                    )
                    frozen_all_modes = (
                        frozen["all_mode_trajectories"][0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    frozen_argmax = (
                        frozen["selected_trajectory"][0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    frozen_selected_modes = (
                        frozen["selected_mode"][0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int64, copy=False)
                    )
                    performance["perf/validation/frozen_inference_seconds"] += (
                        time.perf_counter() - frozen_inference_started
                    )
                    pretrain_reward_started = time.perf_counter()
                    pretrain_local = vehicle_backend.score_pretrain(
                        env,
                        values,
                        frozen_all_modes,
                        frozen_argmax,
                        execution_mask,
                    )
                    performance["perf/validation/pretrain_reward_seconds"] += (
                        time.perf_counter() - pretrain_reward_started
                    )
                else:
                    cache_hits += 1
                    frozen_all_modes = cached.all_mode_trajectories
                    frozen_argmax = cached.selected_trajectory
                    frozen_selected_modes = cached.selected_mode
                    pretrain_local = cached.pretrain_reward
                paired_initial, paired_transition = _diffusion_attempt_generators(
                    device=device,
                    training_seed=10_000_019 + 1_009 * int(seed),
                    live_state_index=scenario_index,
                    retry_index=0,
                )
                paired_sampling_started = time.perf_counter()
                if cached is None:
                    paired_rollout = trainer.sample_groups(
                        batch,
                        generator=paired_initial,
                        transition_generator=paired_transition,
                        noise_bundle_identity=(
                            10_000_019 + 1_009 * int(seed),
                            int(scenario_index),
                            0,
                        ),
                    )
                    paired_current_candidates = (
                        paired_rollout.candidate_trajectories[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    paired_frozen_candidates = (
                        paired_rollout.frozen_candidate_trajectories[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                else:
                    paired_current_candidates = (
                        trainer.sample_current_groups(
                            batch,
                            generator=paired_initial,
                            transition_generator=paired_transition,
                        )[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    paired_frozen_candidates = cached.paired_candidates
                performance["perf/validation/paired_sampling_seconds"] += (
                    time.perf_counter() - paired_sampling_started
                )
                current_reward_started = time.perf_counter()
                paired_current = vehicle_backend.score_candidates(
                    env,
                    values,
                    paired_current_candidates,
                    frozen_argmax,
                    execution_mask,
                    pretrain_local,
                )
                performance["perf/validation/current_n48_reward_seconds"] += (
                    time.perf_counter() - current_reward_started
                )
                if cached is None:
                    frozen_reward_started = time.perf_counter()
                    paired_frozen = vehicle_backend.score_candidates(
                        env,
                        values,
                        paired_frozen_candidates,
                        frozen_argmax,
                        execution_mask,
                        pretrain_local,
                    )
                    performance["perf/validation/frozen_n48_reward_seconds"] += (
                        time.perf_counter() - frozen_reward_started
                    )
                    cached = _FixedValidationFrozenEntry(
                        all_mode_trajectories=np.array(
                            frozen_all_modes, dtype=np.float32, copy=True
                        ),
                        selected_trajectory=np.array(
                            frozen_argmax, dtype=np.float32, copy=True
                        ),
                        selected_mode=np.array(
                            frozen_selected_modes, dtype=np.int64, copy=True
                        ),
                        pretrain_reward=pretrain_local,
                        paired_candidates=np.array(
                            paired_frozen_candidates, dtype=np.float32, copy=True
                        ),
                        paired_reward=paired_frozen,
                    )
                    cache[cache_key] = cached
                else:
                    paired_frozen = cached.paired_reward
                paired_valid = np.asarray(
                    paired_current.valid_mode_mask, dtype=np.bool_
                )
                role_current = [
                    float(paired_current.rewards[role][paired_valid[role]].mean())
                    for role in range(3)
                ]
                role_frozen = [
                    float(paired_frozen.rewards[role][paired_valid[role]].mean())
                    for role in range(3)
                ]
                paired_n48_reward_means.append(float(np.mean(role_current)))
                paired_n48_frozen_reward_means.append(float(np.mean(role_frozen)))
                paired_n48_gain_means.append(
                    float(np.mean(np.asarray(role_current) - np.asarray(role_frozen)))
                )
                n1_reward_started = time.perf_counter()
                current_local = single_vehicle_backend.score_candidates(
                    env,
                    values,
                    current_all_modes[:, :, None],
                    frozen_argmax,
                    execution_mask,
                    pretrain_local,
                )
                performance["perf/validation/current_n1_reward_seconds"] += (
                    time.perf_counter() - n1_reward_started
                )
                valid = np.asarray(current_local.valid_mode_mask, dtype=np.bool_)
                scenario_key = str(scenario[0]).split("_", 1)[0]
                for role in range(3):
                    current_mode = int(selected_modes[0, role])
                    frozen_mode = int(frozen_selected_modes[role])
                    current_selected_reward = float(
                        current_local.rewards[role, current_mode, 0]
                    )
                    frozen_selected_reward = float(
                        pretrain_local.rewards[role, frozen_mode]
                    )
                    selected_vehicle_rewards.append(current_selected_reward)
                    selected_pretrain_vehicle_rewards.append(frozen_selected_reward)
                    scenario_selected_rewards[scenario_key].append(
                        current_selected_reward
                    )
                    scenario_selected_pretrain_rewards[scenario_key].append(
                        frozen_selected_reward
                    )
                    scenario_road_penalties[scenario_key].append(
                        float(
                            current_local.components["road_penalty"][
                                role, current_mode, 0
                            ]
                        )
                    )
                    scenario_pretrain_road_penalties[scenario_key].append(
                        float(
                            pretrain_local.components["road_penalty"][
                                role, frozen_mode
                            ]
                        )
                    )
                    scenario_road_margins[scenario_key].append(
                        float(
                            current_local.components["minimum_road_margin_m"][
                                role, current_mode, 0
                            ]
                        )
                    )
                    scenario_pretrain_road_margins[scenario_key].append(
                        float(
                            pretrain_local.components["minimum_road_margin_m"][
                                role, frozen_mode
                            ]
                        )
                    )
                per_mode_reward = np.asarray(current_local.rewards).mean(axis=-1)
                per_mode_unsafe = np.asarray(current_local.unsafe)[..., 0]
                per_mode_collision = np.asarray(current_local.collision)[..., 0]
                per_mode_out = np.asarray(current_local.out_of_drivable)[..., 0]
                context_role_rewards = []
                context_role_pretrain_rewards = []
                for role in range(3):
                    role_valid = valid[role]
                    context_role_rewards.append(
                        float(per_mode_reward[role][role_valid].mean())
                    )
                    context_role_pretrain_rewards.append(
                        float(
                            np.asarray(current_local.pretrain_rewards)[role][
                                role_valid
                            ].mean()
                        )
                    )
                macro_context_vehicle_rewards.append(
                    float(np.mean(context_role_rewards))
                )
                macro_context_pretrain_rewards.append(
                    float(np.mean(context_role_pretrain_rewards))
                )
                vehicle_rewards.extend(per_mode_reward[valid].tolist())
                pretrain_vehicle_rewards.extend(
                    np.asarray(current_local.pretrain_rewards)[valid].tolist()
                )
                vehicle_unsafe_count += int(per_mode_unsafe[valid].sum())
                vehicle_collision_count += int(per_mode_collision[valid].sum())
                vehicle_out_count += int(per_mode_out[valid].sum())
                for role in range(3):
                    role_valid = valid[role]
                    role_rewards[role].extend(
                        per_mode_reward[role][role_valid].tolist()
                    )
                    role_unsafe_counts[role] += int(
                        per_mode_unsafe[role][role_valid].sum()
                    )
                    role_collision_counts[role] += int(
                        per_mode_collision[role][role_valid].sum()
                    )
                    role_out_counts[role] += int(per_mode_out[role][role_valid].sum())
                joint_reward_started = time.perf_counter()
                raw_proxy = proxy_backend.score(env, values, raw_candidate)
                performance["perf/validation/joint_reward_seconds"] += (
                    time.perf_counter() - joint_reward_started
                )
                raw_rewards.append(float(raw_proxy.rewards[0]))
                raw_unsafe_count += int(raw_proxy.unsafe[0])
                raw_collision_count += int(raw_proxy.collision[0])
                raw_out_count += int(raw_proxy.out_of_drivable[0])
                validation_optimization = optimize_selected_model_trajectories(
                    values,
                    raw_candidate,
                    selected_modes,
                    optimizer=trajectory_optimizer,
                )
                validation_optimization, _, _, _ = _finalize_online_rule_action(
                    rule_maker,
                    condition,
                    env=env,
                    scenario=tuple(scenario),
                    values=values,
                    selected_modes=selected_modes[0],
                    optimization=validation_optimization,
                    optimizer=trajectory_optimizer,
                )
                command_candidate = validation_optimization.optimized_trajectories
                spec = JointEpisodeSpec(
                    scenario_id=str(scenario[0]),
                    local_route=str(scenario[1]),
                    seed=int(seed),
                    reference_pose_global=reference_pose,
                )
                simulator_started = time.perf_counter()
                try:
                    branch = evaluator.evaluate_from_replayed_env(
                        spec, env, command_candidate
                    )
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
                    branch = None
                finally:
                    performance["perf/validation/simulator_seconds"] += (
                        time.perf_counter() - simulator_started
                    )
            finally:
                env.close()
            if branch is None:
                continue
            simulator_rewards.append(float(branch.reward.rewards[0]))
            scenario_key = str(scenario[0]).split("_", 1)[0]
            scenario_simulator_rewards[scenario_key].append(
                float(branch.reward.rewards[0])
            )
            simulator_unsafe_count += int(branch.reward.unsafe[0])
            simulator_collision_count += int(branch.reward.collision[0])
            simulator_out_count += int(branch.reward.out_of_drivable[0])
            scenario_simulator_collisions[scenario_key] += int(
                branch.reward.collision[0]
            )
            scenario_simulator_outs[scenario_key] += int(
                branch.reward.out_of_drivable[0]
            )
    if not raw_rewards or not vehicle_rewards:
        raise OnlineGRPOError("fixed validation produced no raw proxy rewards")
    metrics = {
        "validation/vehicle_reward_mean": float(
            np.mean(macro_context_vehicle_rewards)
        ),
        "validation/vehicle_reward_flat_mean": float(np.mean(vehicle_rewards)),
        "validation/same_mode_pretrain_reward_mean": float(
            np.mean(macro_context_pretrain_rewards)
        ),
        "validation/vehicle_reward_gain": float(
            np.mean(macro_context_vehicle_rewards)
            - np.mean(macro_context_pretrain_rewards)
        ),
        "validation/paired_n48_vehicle_reward_mean": float(
            np.mean(paired_n48_reward_means)
        ),
        "validation/paired_n48_frozen_reward_mean": float(
            np.mean(paired_n48_frozen_reward_means)
        ),
        "validation/paired_n48_reward_gain": float(
            np.mean(paired_n48_gain_means)
        ),
        "validation/vehicle_unsafe_count": float(vehicle_unsafe_count),
        "validation/vehicle_collision_count": float(vehicle_collision_count),
        "validation/vehicle_out_of_drivable_count": float(vehicle_out_count),
        "validation/raw_proxy_reward_mean": float(np.mean(raw_rewards)),
        "validation/raw_proxy_unsafe_count": float(raw_unsafe_count),
        "validation/raw_proxy_collision_count": float(raw_collision_count),
        "validation/raw_proxy_out_of_road_count": float(raw_out_count),
        "validation/simulator_available": float(not simulator_errors),
        "validation/simulator_success_count": float(len(simulator_rewards)),
        "validation/simulator_failure_count": float(len(simulator_errors)),
        "validation/selected_vehicle_reward_mean": float(
            np.mean(selected_vehicle_rewards)
        ),
        "validation/selected_pretrain_vehicle_reward_mean": float(
            np.mean(selected_pretrain_vehicle_rewards)
        ),
        "validation/selected_vehicle_reward_gain": float(
            np.mean(selected_vehicle_rewards)
            - np.mean(selected_pretrain_vehicle_rewards)
        ),
    }
    for scenario_key in sorted(scenario_selected_rewards):
        current_selected = scenario_selected_rewards[scenario_key]
        frozen_selected = scenario_selected_pretrain_rewards[scenario_key]
        metrics.update(
            {
                f"validation/{scenario_key}/selected_vehicle_reward_mean": float(
                    np.mean(current_selected)
                ),
                f"validation/{scenario_key}/selected_vehicle_reward_gain": float(
                    np.mean(current_selected) - np.mean(frozen_selected)
                ),
                f"validation/{scenario_key}/road_penalty_mean": float(
                    np.mean(scenario_road_penalties[scenario_key])
                ),
                f"validation/{scenario_key}/pretrain_road_penalty_mean": float(
                    np.mean(scenario_pretrain_road_penalties[scenario_key])
                ),
                f"validation/{scenario_key}/minimum_road_margin_m": float(
                    np.min(scenario_road_margins[scenario_key])
                ),
                f"validation/{scenario_key}/pretrain_minimum_road_margin_m": float(
                    np.min(scenario_pretrain_road_margins[scenario_key])
                ),
            }
        )
    for role in range(3):
        if not role_rewards[role]:
            raise OnlineGRPOError(
                f"fixed validation produced no valid modes for vehicle {role}"
            )
        metrics.update(
            {
                f"validation/vehicle_{role}_reward_mean": float(
                    np.mean(role_rewards[role])
                ),
                f"validation/vehicle_{role}_unsafe_count": float(
                    role_unsafe_counts[role]
                ),
                f"validation/vehicle_{role}_collision_count": float(
                    role_collision_counts[role]
                ),
                f"validation/vehicle_{role}_out_of_drivable_count": float(
                    role_out_counts[role]
                ),
            }
        )
    if simulator_rewards:
        metrics.update(
            {
                "validation/simulator_reward_mean": float(np.mean(simulator_rewards)),
                "validation/unsafe_count": float(simulator_unsafe_count),
                "validation/collision_count": float(simulator_collision_count),
                "validation/out_of_road_count": float(simulator_out_count),
            }
        )
        for scenario_key in sorted(scenario_simulator_rewards):
            metrics.update(
                {
                    f"validation/{scenario_key}/simulator_reward_mean": float(
                        np.mean(scenario_simulator_rewards[scenario_key])
                    ),
                    f"validation/{scenario_key}/simulator_collision_count": float(
                        scenario_simulator_collisions[scenario_key]
                    ),
                    f"validation/{scenario_key}/simulator_out_count": float(
                        scenario_simulator_outs[scenario_key]
                    ),
                }
            )
    performance["perf/validation/total_seconds"] = (
        time.perf_counter() - validation_started
    )
    metrics.update(performance)
    metrics["perf/validation/frozen_cache_hits"] = float(cache_hits)
    metrics["perf/validation/frozen_cache_misses"] = float(cache_misses)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise OnlineGRPOError("fixed validation metrics must be finite")
    return metrics, tuple(simulator_errors)


def _validation_reward_comparison_metrics(
    current_validation: Mapping[str, object],
    pretrain_validation: Mapping[str, object],
) -> dict[str, float]:
    reward_tag = "validation/vehicle_reward_mean"
    if reward_tag not in current_validation:
        raise OnlineGRPOError(
            f"fixed validation is missing required metric {reward_tag}"
        )
    try:
        current_reward = float(current_validation[reward_tag])
        baseline_reward = float(pretrain_validation[reward_tag])
    except (TypeError, ValueError) as exc:
        raise OnlineGRPOError(
            "validation current and pretrain rewards must be finite scalars"
        ) from exc
    if not math.isfinite(current_reward) or not math.isfinite(baseline_reward):
        raise OnlineGRPOError(
            "validation current and pretrain rewards must be finite scalars"
        )
    metrics = {
        "validation/fixed_pretrain_vehicle_reward": baseline_reward,
        "validation/fixed_pretrain_vehicle_reward_gain": (
            current_reward - baseline_reward
        ),
    }
    comparison_tags = (
        "selected_vehicle_reward_mean",
        "simulator_reward_mean",
        "collision_count",
        "out_of_road_count",
        "S7/simulator_collision_count",
        "S7/simulator_out_count",
        "S7/road_penalty_mean",
        "S7/minimum_road_margin_m",
    )
    for suffix in comparison_tags:
        tag = f"validation/{suffix}"
        if tag not in current_validation or tag not in pretrain_validation:
            continue
        current_value = float(current_validation[tag])
        frozen_value = float(pretrain_validation[tag])
        if not math.isfinite(current_value) or not math.isfinite(frozen_value):
            raise OnlineGRPOError(
                f"validation comparison metric {tag} must be finite"
            )
        metrics[f"validation/fixed_pretrain/{suffix}"] = frozen_value
        metrics[f"validation/gain/{suffix}"] = current_value - frozen_value
    aliases = {
        "simulator_reward_mean": "validation/simulator_reward_gain",
        "selected_vehicle_reward_mean": "validation/selected_reward_gain",
        "S7/simulator_out_count": "validation/S7/out_delta",
        "S7/simulator_collision_count": "validation/S7/collision_delta",
        "S7/road_penalty_mean": "validation/S7/road_penalty_delta",
        "S7/minimum_road_margin_m": "validation/S7/road_margin_delta",
    }
    for suffix, alias in aliases.items():
        source = f"validation/gain/{suffix}"
        if source in metrics:
            metrics[alias] = metrics[source]
    return metrics


def _validation_vehicle_reward(validation: Mapping[str, object]) -> float:
    """Return the sole best-checkpoint objective as a finite scalar."""

    reward_tag = "validation/vehicle_reward_mean"
    if reward_tag not in validation:
        raise OnlineGRPOError(
            f"fixed validation is missing required metric {reward_tag}"
        )
    try:
        reward = float(validation[reward_tag])
    except (TypeError, ValueError) as exc:
        raise OnlineGRPOError(
            "validation vehicle reward must be a finite scalar"
        ) from exc
    if not math.isfinite(reward):
        raise OnlineGRPOError("validation vehicle reward must be a finite scalar")
    return reward


def _validation_is_safety_eligible(
    validation: Mapping[str, object],
    pretrain_validation: Mapping[str, object],
) -> bool:
    """Apply the frozen-baseline safety constraints for checkpoint selection."""

    if float(validation.get("validation/simulator_available", 0.0)) != 1.0:
        return False
    constrained_tags = (
        "validation/collision_count",
        "validation/out_of_road_count",
        "validation/S7/simulator_out_count",
    )
    for tag in constrained_tags:
        if tag not in validation or tag not in pretrain_validation:
            return False
        current = float(validation[tag])
        frozen = float(pretrain_validation[tag])
        if not math.isfinite(current) or not math.isfinite(frozen) or current > frozen:
            return False
    return True


def _append_validation_selection_event(
    history: list[dict[str, float]],
    *,
    accepted_update_state: int,
    validation: Mapping[str, object],
    pretrain_validation: Mapping[str, object],
) -> tuple[bool, tuple[float, float] | None]:
    """Append one validation and return its safety-constrained trailing score."""

    simulator_gain = float(validation["validation/simulator_reward_gain"])
    selected_gain = float(validation["validation/selected_reward_gain"])
    s7_out_delta = float(validation["validation/S7/out_delta"])
    if not all(
        math.isfinite(value)
        for value in (simulator_gain, selected_gain, s7_out_delta)
    ):
        raise OnlineGRPOError("checkpoint selection gains must be finite")
    safety_eligible = _validation_is_safety_eligible(
        validation, pretrain_validation
    )
    history.append(
        {
            "accepted_update_state": float(accepted_update_state),
            "simulator_reward_gain": simulator_gain,
            "selected_reward_gain": selected_gain,
            "s7_out_delta": s7_out_delta,
            "safety_eligible": float(safety_eligible),
        }
    )
    if accepted_update_state < 60 or len(history) < 3:
        return safety_eligible, None
    trailing = history[-3:]
    return safety_eligible, (
        float(np.mean([value["simulator_reward_gain"] for value in trailing])),
        float(np.mean([value["selected_reward_gain"] for value in trailing])),
    )


def _resume_best_checkpoint_anchor(
    resume_checkpoint: Path,
    resume_payload: Mapping[str, object],
) -> tuple[Path | None, tuple[float, float] | None, list[dict[str, float]]]:
    """Resolve the safety-constrained best checkpoint and selection history."""

    history = _validated_selection_history(
        resume_payload.get("validation_selection_history")
    )
    raw_best_reward = resume_payload.get("best_validation_reward")
    raw_best_selected = resume_payload.get("best_selected_reward_gain")
    best_sha = resume_payload.get("best_checkpoint_sha256")
    if raw_best_reward is None:
        if raw_best_selected is not None or best_sha is not None:
            raise OnlineGRPOError("resume checkpoint absent best score conflicts")
        return None, None, history
    best_score = (float(raw_best_reward), float(raw_best_selected))
    if best_sha is None:
        return Path(resume_checkpoint), best_score, history

    best_path = Path(resume_checkpoint).with_name("best.pt")
    if _checkpoint_file_sha256(best_path) != best_sha:
        raise OnlineGRPOError("resume best checkpoint SHA256 mismatch")
    try:
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
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
        "training_candidate_domain",
        "environment_action_source",
        "execution_input_domain",
        "best_checkpoint_metric",
        "joint_reward_role",
        "tracking_expansion_enabled",
        "calibration_required",
        "policy_update_contract",
        "policy_update_contract_sha256",
        "rollout_collection_contract",
        "scenario_contract_sha256",
        "trajectory_optimizer_sha256",
    )
    if any(
        best_payload.get(field) != resume_payload.get(field) for field in binding_fields
    ):
        raise OnlineGRPOError("resume best checkpoint contract binding mismatch")
    raw_best_reward = best_payload.get("best_validation_reward")
    raw_best_selected = best_payload.get("best_selected_reward_gain")
    if (
        isinstance(raw_best_reward, bool)
        or not isinstance(raw_best_reward, (int, float))
        or not math.isfinite(float(raw_best_reward))
        or isinstance(raw_best_selected, bool)
        or not isinstance(raw_best_selected, (int, float))
        or not math.isfinite(float(raw_best_selected))
    ):
        raise OnlineGRPOError("resume best checkpoint score binding mismatch")
    if (
        best_payload.get("best_checkpoint_sha256") is not None
        or (float(raw_best_reward), float(raw_best_selected)) != best_score
    ):
        raise OnlineGRPOError("resume best checkpoint score binding mismatch")
    return best_path, best_score, history


def run_joint_grpo_training(
    training_config: JointGRPOTrainingConfig,
    *,
    run_dir: Path,
) -> dict[str, object]:
    if not isinstance(training_config, JointGRPOTrainingConfig):
        raise OnlineGRPOError("training_config must be a JointGRPOTrainingConfig")
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise OnlineGRPOError(f"run_dir must already exist: {run_dir}")
    if not run_dir.is_dir():
        raise OnlineGRPOError(f"run_dir must be a directory: {run_dir}")
    unexpected_entries = sorted(
        child.name for child in run_dir.iterdir() if child.name != "training.log"
    )
    if unexpected_entries:
        raise OnlineGRPOError(
            "run_dir must be reserved and may contain only training.log; "
            f"found: {', '.join(unexpected_entries)}"
        )
    training_log = run_dir / "training.log"
    if training_log.exists() and not training_log.is_file():
        raise OnlineGRPOError(f"run_dir training.log must be a file: {training_log}")
    (run_dir / "checkpoints").mkdir()

    started_at = time.monotonic()
    performance_totals: defaultdict[str, float] = defaultdict(float)
    validation_calls = 0
    fixed_validation_cache: dict[
        tuple[tuple[str, str], int], _FixedValidationFrozenEntry
    ] = {}
    config = training_config.online
    variant = training_config.variant
    run_mode = training_config.run_mode
    source_checkpoint = training_config.source_checkpoint
    target_accepted_update_states = config.total_rollout_groups
    max_sampling_attempts = (
        config.max_sampling_attempts_multiplier * target_accepted_update_states
    )

    reward_config = VehicleModeRewardConfig(
        trajectories_per_mode=config.trajectories_per_mode
    )
    joint_diagnostic_reward_config = JointRewardConfig()
    _validate_raw_reward_config(joint_diagnostic_reward_config)
    try:
        scenario_contract_sha = validate_primary_scenario_contract(
            primary_scenario_contract(config.scenarios)
        )
    except BEVScenarioContractError as exc:
        raise OnlineGRPOError(str(exc)) from exc
    torch_device = _device(config.device)
    grpo_config = JointGRPOConfig(trajectories_per_mode=config.trajectories_per_mode)
    trainer, source_payload, source_sha = _load_trainer(
        variant,
        Path(source_checkpoint),
        torch_device,
        grpo_config=grpo_config,
        allow_diagnostic_source=run_mode == "smoke",
    )
    if (
        run_mode == "formal"
        and source_payload.get("eligible_for_formal_training") is not True
    ):
        raise OnlineGRPOError("formal GRPO requires an eligible Stage 1 source")

    implementation_commit = _implementation_commit()
    validation_state_bank, validation_state_bank_sha256 = (
        _load_grpo_validation_state_bank(
            config.validation_state_bank,
            scenarios=config.scenarios,
            seeds=HOLDOUT_SEEDS,
        )
    )
    pretrain_validation, pretrain_simulator_errors = (
        _fixed_raw_proxy_and_simulator_validation(
            trainer,
            device=torch_device,
            reward_config=joint_diagnostic_reward_config,
            scenarios=config.scenarios,
            seeds=HOLDOUT_SEEDS,
            validation_state_bank=validation_state_bank,
            frozen_cache=fixed_validation_cache,
        )
    )
    validation_calls += 1
    for name, value in pretrain_validation.items():
        if name.startswith("perf/") and name.endswith("_seconds"):
            performance_totals[name] += float(value)
    checkpoint_loader = (
        load_grpo_checkpoint if variant == "A" else load_grpo_b_checkpoint
    )
    training_buckets = _round_robin_training_buckets(
        config.scenarios, config.scenario_seeds
    )
    bucket_target_counts = _balanced_bucket_targets(
        target_accepted_update_states, len(training_buckets)
    )
    collection_contract = rollout_collection_contract(config)
    rollout_start_generator = torch.Generator(device=torch_device)
    rollout_start_generator.manual_seed(config.seed)
    environment_steps = 0
    warmup_environment_steps = 0
    accepted_update_states = 0
    sampling_attempts = 0
    rejected_sampling_attempts = 0
    stability_guard_rejections = 0
    exhausted_states = 0
    baseline_execution_steps = 0
    zero_signal_epochs = 0
    bucket_accepted_update_counts = [0 for _ in training_buckets]
    bucket_sampling_attempt_counts = [0 for _ in training_buckets]
    bucket_rejected_sampling_attempt_counts = [0 for _ in training_buckets]
    bucket_stability_guard_rejection_counts = [0 for _ in training_buckets]
    bucket_exhausted_state_counts = [0 for _ in training_buckets]
    bucket_baseline_execution_step_counts = [0 for _ in training_buckets]
    bucket_zero_signal_epoch_counts = [0 for _ in training_buckets]
    bucket_optimizer_step_counts = [0 for _ in training_buckets]
    bucket_episode_counts = [0 for _ in training_buckets]
    next_bucket_index = 0
    current_visit_progress = 0
    last_validated_update_state = -1
    advantage_vector_record_count = 0
    rollout_start_offsets_this_run: list[int] = []
    rollout_start_accepted_count = 0
    rollout_start_rejected_count = 0
    temporary_start_offset_upper_bound: int | None = None
    rule_diagnostics = {
        "conditioned_rollouts": 0,
        "proposal_match_attempts": 0,
        "proposal_matches": 0,
        "condition_failures": 0,
        "forced_safe_stops": 0,
        "s7_feedback_exception_hits": 0,
        "commitment_conditioned_rollouts": 0,
        "commitment_feedback_incompatible": 0,
    }
    last_metrics: dict[str, float] = {
        str(name): float(value) for name, value in pretrain_validation.items()
    }
    resume_best_path: Path | None = None
    best_reward: float | None = None
    best_selected_reward_gain: float | None = None
    best_unconstrained_score: tuple[float, float] | None = None
    validation_selection_history: list[dict[str, float]] = []
    if config.resume_checkpoint is not None:
        resume_payload = checkpoint_loader(
            config.resume_checkpoint,
            trainer,
            expected_source_stage1_sha256=source_sha,
        )
        restored_sampler_state = _validate_online_checkpoint_metadata(
            resume_payload,
            run_mode=run_mode,
            reward_config=reward_config,
            scenario_contract_sha=scenario_contract_sha,
            scenario_seeds=config.scenario_seeds,
            collection_contract=collection_contract,
            bucket_count=len(training_buckets),
            bucket_target_counts=bucket_target_counts,
            rollout_groups_per_bucket_visit=(config.rollout_groups_per_bucket_visit),
            optimizer_step=trainer.optimizer_step,
            max_sampling_attempts=max_sampling_attempts,
        )
        environment_steps = int(resume_payload["environment_steps"])
        warmup_environment_steps = int(
            restored_sampler_state["warmup_environment_steps"]
        )
        accepted_update_states = int(restored_sampler_state["accepted_update_states"])
        sampling_attempts = int(restored_sampler_state["sampling_attempts"])
        rejected_sampling_attempts = int(
            restored_sampler_state["rejected_sampling_attempts"]
        )
        stability_guard_rejections = int(
            restored_sampler_state["stability_guard_rejections"]
        )
        exhausted_states = int(restored_sampler_state["exhausted_states"])
        baseline_execution_steps = int(
            restored_sampler_state["baseline_execution_steps"]
        )
        zero_signal_epochs = int(restored_sampler_state["zero_signal_epochs"])
        bucket_accepted_update_counts = list(
            restored_sampler_state["bucket_accepted_update_counts"]
        )
        bucket_sampling_attempt_counts = list(
            restored_sampler_state["bucket_sampling_attempt_counts"]
        )
        bucket_rejected_sampling_attempt_counts = list(
            restored_sampler_state["bucket_rejected_sampling_attempt_counts"]
        )
        bucket_stability_guard_rejection_counts = list(
            restored_sampler_state["bucket_stability_guard_rejection_counts"]
        )
        bucket_exhausted_state_counts = list(
            restored_sampler_state["bucket_exhausted_state_counts"]
        )
        bucket_baseline_execution_step_counts = list(
            restored_sampler_state["bucket_baseline_execution_step_counts"]
        )
        bucket_zero_signal_epoch_counts = list(
            restored_sampler_state["bucket_zero_signal_epoch_counts"]
        )
        bucket_optimizer_step_counts = list(
            restored_sampler_state["bucket_optimizer_step_counts"]
        )
        bucket_episode_counts = list(restored_sampler_state["bucket_episode_counts"])
        next_bucket_index = int(restored_sampler_state["next_bucket_index"])
        current_visit_progress = int(restored_sampler_state["current_visit_progress"])
        last_validated_update_state = int(
            restored_sampler_state["last_validated_update_state"]
        )
        generator_states = restored_sampler_state["generator_states"]
        rollout_start_generator.set_state(generator_states["rollout_start"])
        last_metrics = {
            str(name): float(value) for name, value in resume_payload["metrics"].items()
        }
        resume_best_path, resumed_best_score, validation_selection_history = (
            _resume_best_checkpoint_anchor(
            Path(config.resume_checkpoint), resume_payload
            )
        )
        if resumed_best_score is not None:
            best_reward, best_selected_reward_gain = resumed_best_score
        if accepted_update_states >= target_accepted_update_states:
            raise OnlineGRPOError(
                "resume checkpoint already reached requested accepted rollout " "groups"
            )

    frozen_pretrain_reward_logging = _frozen_pretrain_reward_logging_metadata(
        trainer.planner
    )
    run_start_optimizer_step = trainer.optimizer_step
    run_start_accepted_update_state = accepted_update_states
    run_start_sampling_attempt = sampling_attempts
    online_config = dataclasses.asdict(config)
    online_config["resume_checkpoint"] = (
        str(config.resume_checkpoint) if config.resume_checkpoint is not None else None
    )
    online_config["validation_state_bank"] = str(config.validation_state_bank)
    frozen = {
        "format": "bev_joint_grpo_online_config_v13",
        "implementation_commit": implementation_commit,
        "variant": variant,
        "run_mode": run_mode,
        "diagnostic_only": run_mode != "formal",
        "eligible_for_formal_training": run_mode == "formal",
        "online_config": online_config,
        "grpo_config": _grpo_config_artifact_payload(grpo_config),
        "source_stage1_sha256": source_sha,
        "policy_update_contract": joint_grpo_optimizer_contract(),
        "policy_update_contract_sha256": joint_grpo_optimizer_contract_sha256(),
        "rollout_collection_contract": collection_contract,
        "reward_contract_version": _reward_contract_version(),
        "reward_contract": VEHICLE_MODE_REWARD_CONTRACT,
        "reward_contract_sha256": VEHICLE_MODE_REWARD_CONTRACT_SHA256,
        "reward_config": dataclasses.asdict(reward_config),
        "reward_config_sha256": vehicle_mode_reward_config_sha256(reward_config),
        "reward_application_contract": GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
        "reward_application_contract_sha256": (
            GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
        ),
        "reward_input_domain": "tau_d",
        "training_candidate_domain": "tau_d_all_vehicle_modes",
        "environment_action_source": "cached_frozen_stage1_argmax",
        "execution_input_domain": "tau_cmd",
        "frozen_pretrain_reward_logging": frozen_pretrain_reward_logging,
        "best_checkpoint_metric": BEST_CHECKPOINT_METRIC,
        "joint_reward_role": "historical_and_final_evaluation_only",
        "tracking_expansion_enabled": False,
        "calibration_required": False,
        "trajectory_optimizer_config": dataclasses.asdict(
            KinematicTrajectoryOptimizerConfig()
        ),
        "trajectory_optimizer_sha256": (KinematicTrajectoryOptimizerConfig().sha256()),
        "scenario_contract": primary_scenario_contract(config.scenarios),
        "scenario_contract_sha256": scenario_contract_sha,
        "validation_state_bank": {
            "path": str(config.validation_state_bank.resolve()),
            "sha256": validation_state_bank_sha256,
            "scenarios": [list(value) for value in config.scenarios],
            "seeds": [int(value) for value in HOLDOUT_SEEDS],
            "state": "earliest_history_ready_and_primary_scenario_ready",
            "noise_seed_formula": "10000019 + 1009 * seed + scenario_index",
            "common_random_numbers": True,
            "ddim_path": DEFAULT_DDIM_PATH.as_dict(),
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics_path = run_dir / "metrics.jsonl"
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    initial_performance = {
        name: float(value)
        for name, value in pretrain_validation.items()
        if name.startswith("perf/")
    }
    for metric_name, metric_value in initial_performance.items():
        writer.add_scalar(metric_name, metric_value, 0)
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "event": "initial_validation_performance",
                    "accepted_update_state": 0,
                    **initial_performance,
                },
                sort_keys=True,
            )
            + "\n"
        )
    vehicle_reward_backend = VehicleModeCounterfactualReward(reward_config)
    trajectory_optimizer = KinematicTrajectoryOptimizer()
    consecutive_empty_episodes = 0
    best_path = run_dir / "checkpoints" / "best.pt"
    best_safe_path = run_dir / "checkpoints" / "best_safe.pt"
    best_unconstrained_path = (
        run_dir / "checkpoints" / "best_simulator_unconstrained.pt"
    )
    milestone_200_path = run_dir / "checkpoints" / "milestone_200.pt"
    milestone_500_path = run_dir / "checkpoints" / "milestone_500.pt"
    last_path = run_dir / "checkpoints" / "last.pt"
    best_checkpoint_sha256: str | None = None
    if resume_best_path is not None:
        shutil.copyfile(resume_best_path, best_path)
        shutil.copyfile(resume_best_path, best_safe_path)
        best_checkpoint_sha256 = _checkpoint_file_sha256(best_path)
    simulator_diagnostic_errors: list[dict[str, object]] = [
        {"optimizer_step": 0, "stage1_baseline": True, **dict(value)}
        for value in pretrain_simulator_errors
    ]

    _reset_cuda_peak_memory(torch_device)
    attempt_budget_exhausted = False
    diagnostic_early_stop = False
    stability_guard_rejections_this_run = 0
    stability_guard_diagnostic: dict[str, object] | None = None
    try:
        while (
            accepted_update_states < target_accepted_update_states
            and not diagnostic_early_stop
            and stability_guard_diagnostic is None
        ):
            bucket_index = next_bucket_index
            if (
                bucket_accepted_update_counts[bucket_index]
                >= bucket_target_counts[bucket_index]
            ):
                raise OnlineGRPOError(
                    "training bucket cursor points to a completed target"
                )
            scenario, seed = training_buckets[bucket_index]
            accepted_at_episode_start = accepted_update_states
            baseline_steps_at_episode_start = baseline_execution_steps
            env = _new_env(scenario, seed)
            bucket_episode_counts[bucket_index] += 1
            builder = JointBEVSampleBuilder(AGENT_IDS)
            builder.reset()
            rule_maker = _new_online_rule_maker(trainer.planner, env)
            committed_execution_id: int | None = None
            committed_plan_actions: dict[str, int] | None = None
            dt_s = simulator_decision_dt_s(env)
            episode_step = 0
            start_ready_step: int | None = None
            start_offset: int | None = None
            start_upper_bound: int | None = None
            start_target_step: int | None = None
            start_accepted = False
            retryable_start_rejection = False
            scenario_window_closed = False
            try:
                while (
                    accepted_update_states < target_accepted_update_states
                    and not diagnostic_early_stop
                    and stability_guard_diagnostic is None
                    and bucket_accepted_update_counts[bucket_index]
                    < bucket_target_counts[bucket_index]
                    and current_visit_progress < config.rollout_groups_per_bucket_visit
                    and episode_step < config.environment_steps_per_episode
                ):
                    builder.capture_state(env, episode_step * dt_s)
                    history_ready = builder.history_ready()
                    if history_ready:
                        scenario_summary = _scenario_summary(env)
                        ready_now = _scenario_ready_from_summary(scenario_summary)
                        window_closed_now = (
                            _scenario_sampling_window_closed_from_summary(
                                scenario_summary
                            )
                        )
                    else:
                        ready_now = False
                        window_closed_now = False
                    scenario_window_closed = bool(
                        scenario_window_closed or window_closed_now
                    )
                    if scenario_window_closed:
                        if start_target_step is not None and not start_accepted:
                            assert start_ready_step is not None
                            assert start_offset is not None
                            assert start_upper_bound is not None
                            last_open_offset = episode_step - start_ready_step - 1
                            if last_open_offset >= 0:
                                temporary_start_offset_upper_bound = min(
                                    last_open_offset,
                                    (
                                        temporary_start_offset_upper_bound
                                        if temporary_start_offset_upper_bound
                                        is not None
                                        else last_open_offset
                                    ),
                                )
                                retryable_start_rejection = True
                            rollout_start_rejected_count += 1
                            _write_rollout_start_event(
                                metrics_path,
                                optimizer_step=trainer.optimizer_step,
                                rollout_group=accepted_update_states,
                                bucket_index=bucket_index,
                                scenario=scenario,
                                seed=seed,
                                bucket_episode=(bucket_episode_counts[bucket_index]),
                                ready_step=start_ready_step,
                                upper_bound=start_upper_bound,
                                sampled_offset=start_offset,
                                target_step=start_target_step,
                                status=("rejected_window_closed_before_target"),
                            )
                        break
                    if start_ready_step is None and ready_now:
                        start_ready_step = episode_step
                        start_offset, start_upper_bound = _sample_rollout_start_offset(
                            config,
                            start_ready_step,
                            generator=rollout_start_generator,
                            temporary_max_offset_steps=(
                                temporary_start_offset_upper_bound
                            ),
                        )
                        start_target_step = start_ready_step + start_offset
                        rollout_start_offsets_this_run.append(start_offset)
                    accepted_this_step = False
                    baseline_this_step = False
                    exhausted_this_step = False
                    baseline_step_result: tuple[object, ...] | None = None
                    if start_target_step is None or episode_step < start_target_step:
                        action = route_following_warmup_actions(env, builder)
                    else:
                        if not start_accepted:
                            assert start_ready_step is not None
                            assert start_offset is not None
                            assert start_upper_bound is not None
                            start_accepted = True
                            temporary_start_offset_upper_bound = None
                            rollout_start_accepted_count += 1
                            _write_rollout_start_event(
                                metrics_path,
                                optimizer_step=trainer.optimizer_step,
                                rollout_group=accepted_update_states + 1,
                                bucket_index=bucket_index,
                                scenario=scenario,
                                seed=seed,
                                bucket_episode=(bucket_episode_counts[bucket_index]),
                                ready_step=start_ready_step,
                                upper_bound=start_upper_bound,
                                sampled_offset=start_offset,
                                target_step=start_target_step,
                                status="accepted",
                            )
                        values = builder.build_model_inputs(env)
                        condition = _condition_online_model_inputs(
                            rule_maker,
                            env,
                            builder,
                            values,
                            committed_execution_id=committed_execution_id,
                            committed_plan_actions=committed_plan_actions,
                        )
                        values = condition.model_inputs
                        execution_mask = execution_mode_valid_mask(
                            values, optimizer=trajectory_optimizer
                        )
                        batch = model_inputs_to_batch(
                            values,
                            torch_device,
                            mode_valid_mask=execution_mask,
                        )
                        accepted_attempt: (
                            tuple[
                                object,
                                object,
                                object,
                                np.ndarray,
                                dict[str, float],
                            ]
                            | None
                        ) = None
                        state_performance: defaultdict[str, float] = defaultdict(float)
                        frozen_raw_candidate: np.ndarray | None = None
                        frozen_all_mode_candidates: np.ndarray | None = None
                        frozen_modes: np.ndarray | None = None
                        pretrain_score: object | None = None
                        attempts_for_state = 0
                        for retry_index in range(
                            1, config.max_sampling_attempts_per_state + 1
                        ):
                            if retry_index == 1 and _attempt_budget_is_exhausted(
                                accepted_update_states=accepted_update_states,
                                target_accepted_update_states=(
                                    target_accepted_update_states
                                ),
                                sampling_attempts=sampling_attempts,
                                max_sampling_attempts=(max_sampling_attempts),
                            ):
                                attempt_budget_exhausted = True
                                break
                            initial_noise_generator, transition_noise_generator = (
                                _diffusion_attempt_generators(
                                    device=torch_device,
                                    training_seed=config.seed,
                                    live_state_index=baseline_execution_steps,
                                    retry_index=retry_index - 1,
                                )
                            )
                            sampling_started = time.perf_counter()
                            rollout = trainer.sample_groups(
                                batch,
                                generator=initial_noise_generator,
                                transition_generator=transition_noise_generator,
                                noise_bundle_identity=(
                                    int(config.seed),
                                    int(baseline_execution_steps),
                                    int(retry_index - 1),
                                ),
                            )
                            raw_candidates = (
                                rollout.candidate_trajectories[0]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float32, copy=False)
                            )
                            frozen_paired_candidates = (
                                rollout.frozen_candidate_trajectories[0]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float32, copy=False)
                            )
                            state_performance["perf/train/n48_sampling_seconds"] += (
                                time.perf_counter() - sampling_started
                            )
                            attempts_for_state += 1
                            if frozen_raw_candidate is None:
                                frozen_inference_started = time.perf_counter()
                                frozen_pretrain = trainer.infer_frozen_pretrain(rollout)
                                frozen_raw_candidate = (
                                    frozen_pretrain["selected_trajectory"]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32, copy=False)
                                )
                                frozen_modes = (
                                    frozen_pretrain["selected_mode"]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.int64, copy=False)
                                )
                                frozen_all_mode_candidates = (
                                    frozen_pretrain["all_mode_trajectories"]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32, copy=False)
                                )
                                state_performance[
                                    "perf/train/frozen_inference_seconds"
                                ] += (time.perf_counter() - frozen_inference_started)
                                pretrain_reward_started = time.perf_counter()
                                pretrain_score = vehicle_reward_backend.score_pretrain(
                                    env,
                                    values,
                                    frozen_all_mode_candidates[0],
                                    frozen_raw_candidate[0],
                                    execution_mask,
                                )
                                state_performance[
                                    "perf/train/pretrain_reward_seconds"
                                ] += (time.perf_counter() - pretrain_reward_started)
                            assert frozen_raw_candidate is not None
                            assert frozen_all_mode_candidates is not None
                            assert pretrain_score is not None
                            current_reward_started = time.perf_counter()
                            proxy = vehicle_reward_backend.score_candidates(
                                env,
                                values,
                                raw_candidates,
                                frozen_raw_candidate[0],
                                execution_mask,
                                pretrain_score,
                            )
                            state_performance[
                                "perf/train/current_reward_seconds"
                            ] += (time.perf_counter() - current_reward_started)
                            frozen_reward_started = time.perf_counter()
                            frozen_proxy = vehicle_reward_backend.score_candidates(
                                env,
                                values,
                                frozen_paired_candidates,
                                frozen_raw_candidate[0],
                                execution_mask,
                                pretrain_score,
                            )
                            state_performance[
                                "perf/train/frozen_reward_seconds"
                            ] += (time.perf_counter() - frozen_reward_started)
                            centered_rewards, advantages, signal_mode_mask = (
                                _fixed_scale_reward_signals(
                                    proxy.rewards,
                                    frozen_proxy.rewards,
                                    proxy.collision,
                                    proxy.out_of_drivable,
                                    proxy.valid_mode_mask,
                                )
                            )
                            sampling_attempts += 1
                            bucket_sampling_attempt_counts[bucket_index] += 1
                            attempt_metrics = _write_dynamic_sampling_attempt_event(
                                metrics_path,
                                optimizer_step=trainer.optimizer_step,
                                accepted_update_state=(accepted_update_states + 1),
                                sampling_attempt=sampling_attempts,
                                retry_index=retry_index,
                                bucket_index=bucket_index,
                                scenario=scenario,
                                seed=seed,
                                reward_result=proxy,
                                paired_frozen_rewards=frozen_proxy.rewards,
                                centered_rewards=centered_rewards,
                                advantages=advantages,
                                hard_valid_mode_mask=np.asarray(
                                    values.mode_valid_mask, dtype=np.bool_
                                ),
                                signal_mode_mask=signal_mode_mask,
                                reward_config=reward_config,
                            )
                            writer.add_scalar(
                                "dynamic_sampling/accepted",
                                float(signal_mode_mask.any()),
                                sampling_attempts,
                            )
                            writer.add_scalar(
                                "dynamic_sampling/signal_mode_count",
                                attempt_metrics["signal_mode_count"],
                                sampling_attempts,
                            )
                            if signal_mode_mask.any():
                                accepted_attempt = (
                                    rollout,
                                    proxy,
                                    frozen_proxy,
                                    signal_mode_mask,
                                    attempt_metrics,
                                )
                                break
                            rejected_sampling_attempts += 1
                            bucket_rejected_sampling_attempt_counts[bucket_index] += 1

                        if attempt_budget_exhausted:
                            break
                        assert frozen_raw_candidate is not None
                        assert frozen_modes is not None
                        assert pretrain_score is not None
                        accepted_this_step = accepted_attempt is not None
                        exhausted_this_step = accepted_attempt is None
                        baseline_this_step = True
                        if exhausted_this_step:
                            exhausted_states += 1
                            bucket_exhausted_state_counts[bucket_index] += 1

                        if accepted_attempt is not None:
                            (
                                rollout,
                                proxy,
                                frozen_proxy,
                                signal_mode_mask,
                                accepted_attempt_metrics,
                            ) = accepted_attempt
                            rollout = rollout.with_reward_signals(
                                current_rewards=torch.from_numpy(
                                    np.asarray(proxy.rewards, dtype=np.float32)
                                ).unsqueeze(0).to(torch_device),
                                frozen_rewards=torch.from_numpy(
                                    np.asarray(
                                        frozen_proxy.rewards, dtype=np.float32
                                    )
                                ).unsqueeze(0).to(torch_device),
                                collision_mask=torch.from_numpy(
                                    np.asarray(proxy.collision, dtype=np.bool_)
                                ).unsqueeze(0).to(torch_device),
                                out_of_drivable_mask=torch.from_numpy(
                                    np.asarray(
                                        proxy.out_of_drivable, dtype=np.bool_
                                    )
                                ).unsqueeze(0).to(torch_device),
                                valid_executable_mode_mask=torch.from_numpy(
                                    np.asarray(
                                        proxy.valid_mode_mask, dtype=np.bool_
                                    )
                                ).unsqueeze(0).to(torch_device),
                            )
                            optimizer_step_before = trainer.optimizer_step
                            update_started = time.perf_counter()
                            update = trainer.update(rollout)
                            state_performance["perf/train/update_seconds"] += (
                                time.perf_counter() - update_started
                            )
                            if update.stability_guard_rejected:
                                stability_guard_rejections += 1
                                stability_guard_rejections_this_run += 1
                                bucket_stability_guard_rejection_counts[
                                    bucket_index
                                ] += 1
                                stability_guard_diagnostic = {
                                    "last_accepted_update_state": int(
                                        accepted_update_states
                                    ),
                                    "guard_rejected_sampling_attempt": int(
                                        sampling_attempts
                                    ),
                                    "frozen_baseline_executed": True,
                                    "optimizer_rollback_applied": True,
                                    "noise_bundle_identity": list(
                                        rollout.noise_bundle_identity or ()
                                    ),
                                    "post_update_reference_kl": float(
                                        update.post_update_reference_kl
                                    ),
                                    "adapter_relative_drifts": list(
                                        update.adapter_relative_drifts
                                    ),
                                    "trigger_modes": list(
                                        update.stability_guard_trigger_modes
                                    ),
                                }
                                accepted_this_step = False
                            optimizer_steps_for_state = (
                                trainer.optimizer_step - optimizer_step_before
                            )
                            bucket_optimizer_step_counts[
                                bucket_index
                            ] += optimizer_steps_for_state
                            zero_signal_epochs += int(update.zero_signal)
                            bucket_zero_signal_epoch_counts[bucket_index] += int(
                                update.zero_signal
                            )
                            next_update_state = accepted_update_states + int(
                                not update.stability_guard_rejected
                            )
                            rollout_advantages = update.loss.advantages
                            _, optimizer_metrics = _split_loss_metrics_by_step_axis(
                                update.loss.scalar_metrics()
                            )
                            optimizer_metrics.update(
                                {
                                    "gradient_total": float(
                                        update.total_gradient_norm
                                    ),
                                    "zero_signal_epoch": float(
                                        update.zero_signal
                                    ),
                                    "policy/post_update_reference_kl": float(
                                        update.post_update_reference_kl
                                    ),
                                    "policy/adapter_drift_max": float(
                                        max(update.adapter_relative_drifts)
                                    ),
                                    "stability_guard/rejected": float(
                                        update.stability_guard_rejected
                                    ),
                                    "optimizer_step": float(
                                        update.optimizer_step
                                    ),
                                    "accepted_update_state": float(
                                        next_update_state
                                    ),
                                }
                            )
                            for name, value in update.gradient_norms.items():
                                optimizer_metrics[f"gradient_pre_clip/{name}"] = float(
                                    value
                                )
                            for name, value in update.clipped_gradient_norms.items():
                                optimizer_metrics[f"gradient_post_clip/{name}"] = float(
                                    value
                                )
                            for mode, drift in enumerate(
                                update.adapter_relative_drifts
                            ):
                                optimizer_metrics[
                                    f"policy/mode_{mode}/adapter_drift"
                                ] = float(drift)
                            for metric_name, metric_value in optimizer_metrics.items():
                                writer.add_scalar(
                                    metric_name, metric_value, next_update_state
                                )
                            if not update.stability_guard_rejected and _write_advantage_vector_summary(
                                writer,
                                rollout_advantages,
                                next_update_state,
                                target_accepted_update_states,
                                config.advantage_vector_log_interval_rollouts,
                                trajectories_per_mode=(
                                    trainer.config.trajectories_per_mode
                                ),
                            ):
                                advantage_vector_record_count += 1
                            valid = np.asarray(proxy.valid_mode_mask, dtype=np.bool_)
                            rollout_metrics = {
                                "environment_steps": float(environment_steps + 1),
                                "accepted_update_states": float(next_update_state),
                                "sampling_attempts": float(sampling_attempts),
                                "rejected_sampling_attempts": float(
                                    rejected_sampling_attempts
                                ),
                                "stability_guard_rejections": float(
                                    stability_guard_rejections
                                ),
                                "exhausted_states": float(exhausted_states),
                                "baseline_execution_steps": float(
                                    baseline_execution_steps + 1
                                ),
                                "zero_signal_epochs": float(zero_signal_epochs),
                                "dynamic_sampling_attempts_for_state": float(
                                    attempts_for_state
                                ),
                                "training_bucket_index": float(bucket_index),
                                "vehicle_reward_mean": accepted_attempt_metrics[
                                    "train/valid_all/vehicle_reward_mean"
                                ],
                                "same_mode_pretrain_reward_mean": (
                                    accepted_attempt_metrics[
                                        "train/valid_all/"
                                        "same_mode_pretrain_reward_mean"
                                    ]
                                ),
                                FROZEN_PRETRAIN_RAW_PROXY_REWARD_TAG: float(
                                    accepted_attempt_metrics[
                                        "train/valid_all/"
                                        "same_mode_pretrain_reward_mean"
                                    ]
                                ),
                                "same_mode_reward_gain_mean": (
                                    accepted_attempt_metrics[
                                        "train/valid_all/reward_gain_mean"
                                    ]
                                ),
                                "signal_mode_count": float(signal_mode_mask.sum()),
                                "signal_vehicle_count": float(
                                    np.any(signal_mode_mask, axis=1).sum()
                                ),
                                "vehicle_unsafe_rate": float(
                                    np.asarray(proxy.unsafe)[valid].mean()
                                ),
                                "vehicle_collision_rate": float(
                                    np.asarray(proxy.collision)[valid].mean()
                                ),
                                "vehicle_out_of_drivable_rate": float(
                                    np.asarray(proxy.out_of_drivable)[valid].mean()
                                ),
                                **{
                                    f"vehicle_{role}/reward_mean": float(
                                        np.asarray(proxy.rewards)[role][
                                            valid[role]
                                        ].mean()
                                    )
                                    for role in range(3)
                                },
                                **_advantage_scalar_metrics(rollout_advantages),
                            }
                            rollout_metrics.update(accepted_attempt_metrics)
                            for metric_name, metric_value in rollout_metrics.items():
                                writer.add_scalar(
                                    metric_name,
                                    metric_value,
                                    next_update_state,
                                )
                            with metrics_path.open("a", encoding="utf-8") as stream:
                                stream.write(
                                    json.dumps(
                                        {
                                            "event": (
                                                "stability_guard_rejection"
                                                if update.stability_guard_rejected
                                                else "update_state"
                                            ),
                                            **optimizer_metrics,
                                            **rollout_metrics,
                                        },
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                            last_metrics.update(rollout_metrics)
                            last_metrics.update(optimizer_metrics)

                        baseline_started = time.perf_counter()
                        try:
                            (
                                baseline_step_result,
                                optimization,
                                rule_event,
                                committed_execution_id,
                                committed_plan_actions,
                            ) = execute_cached_frozen_baseline(
                                env=env,
                                rule_maker=rule_maker,
                                condition=condition,
                                scenario=scenario,
                                model_inputs=values,
                                frozen_raw_trajectories=(frozen_raw_candidate),
                                frozen_selected_modes=frozen_modes,
                                optimizer=trajectory_optimizer,
                            )
                            state_performance[
                                "perf/train/baseline_step_seconds"
                            ] += (time.perf_counter() - baseline_started)
                        except TrajectoryOptimizationError:
                            np.savez_compressed(
                                run_dir / "trajectory_optimizer_failure.npz",
                                raw_trajectories=frozen_raw_candidate,
                                frozen_selected_modes=frozen_modes,
                                coarse_trajectories=values.coarse_trajectories,
                                current_speeds_mps=values.ego_state[:, 0],
                                mode_valid_mask=values.mode_valid_mask,
                                candidate_source=np.asarray(["frozen_stage1_argmax"]),
                                environment_steps=np.asarray(
                                    [environment_steps], dtype=np.int64
                                ),
                                episode_steps=np.asarray(
                                    [episode_step], dtype=np.int64
                                ),
                            )
                            raise
                        for name, value in rule_event.items():
                            rule_diagnostics[name] += int(value)
                        baseline_execution_steps += 1
                        bucket_baseline_execution_step_counts[bucket_index] += 1
                        for metric_name, metric_value in state_performance.items():
                            performance_totals[metric_name] += float(metric_value)
                            writer.add_scalar(
                                metric_name,
                                float(metric_value),
                                baseline_execution_steps,
                            )
                        last_metrics.update(state_performance)
                        pretrain_values = np.asarray(pretrain_score.rewards)[
                            execution_mask
                        ]
                        _write_baseline_execution_event(
                            metrics_path,
                            optimizer_step=trainer.optimizer_step,
                            accepted_update_states=(
                                accepted_update_states + int(accepted_this_step)
                            ),
                            rejected_sampling_attempts=(rejected_sampling_attempts),
                            stability_guard_rejections=(
                                stability_guard_rejections
                            ),
                            exhausted_states=exhausted_states,
                            baseline_execution_steps=(baseline_execution_steps),
                            environment_steps=environment_steps + 1,
                            bucket_index=bucket_index,
                            scenario=scenario,
                            seed=seed,
                            pretrain_reward_mean=float(pretrain_values.mean()),
                            performance=state_performance,
                        )

                    stepped_from = episode_step
                    if baseline_step_result is None:
                        _, _, terminated, truncated, info = env.step(action)
                    else:
                        _, _, terminated, truncated, info = baseline_step_result
                    environment_steps += 1
                    episode_step += 1
                    if accepted_this_step:
                        accepted_update_states += 1
                        bucket_accepted_update_counts[bucket_index] += 1
                        current_visit_progress += 1
                    if stability_guard_diagnostic is not None:
                        break
                    if not baseline_this_step:
                        warmup_environment_steps += 1
                    if _attempt_budget_is_exhausted(
                        accepted_update_states=accepted_update_states,
                        target_accepted_update_states=(target_accepted_update_states),
                        sampling_attempts=sampling_attempts,
                        max_sampling_attempts=(max_sampling_attempts),
                    ):
                        attempt_budget_exhausted = True
                        break
                    if episode_has_ended(terminated, truncated, info):
                        if start_target_step is not None and not start_accepted:
                            assert start_ready_step is not None
                            assert start_offset is not None
                            assert start_upper_bound is not None
                            last_open_offset = stepped_from - start_ready_step
                            if last_open_offset >= 0:
                                temporary_start_offset_upper_bound = min(
                                    last_open_offset,
                                    (
                                        temporary_start_offset_upper_bound
                                        if temporary_start_offset_upper_bound
                                        is not None
                                        else last_open_offset
                                    ),
                                )
                                retryable_start_rejection = True
                            rollout_start_rejected_count += 1
                            _write_rollout_start_event(
                                metrics_path,
                                optimizer_step=trainer.optimizer_step,
                                rollout_group=accepted_update_states,
                                bucket_index=bucket_index,
                                scenario=scenario,
                                seed=seed,
                                bucket_episode=(bucket_episode_counts[bucket_index]),
                                ready_step=start_ready_step,
                                upper_bound=start_upper_bound,
                                sampled_offset=start_offset,
                                target_step=start_target_step,
                                status=("rejected_episode_ended_before_target"),
                            )
                        break
                    if accepted_update_states > accepted_at_episode_start and (
                        _bucket_visit_is_complete(
                            bucket_index=bucket_index,
                            bucket_sample_counts=(bucket_accepted_update_counts),
                            bucket_target_counts=bucket_target_counts,
                            current_visit_progress=current_visit_progress,
                            rollout_groups_per_bucket_visit=(
                                config.rollout_groups_per_bucket_visit
                            ),
                        )
                        or accepted_update_states == target_accepted_update_states
                        or accepted_update_states % config.validation_interval_rollouts
                        == 0
                    ):
                        break
            finally:
                env.close()
            if (
                accepted_update_states == accepted_at_episode_start
                and baseline_execution_steps == baseline_steps_at_episode_start
            ):
                if retryable_start_rejection:
                    consecutive_empty_episodes = 0
                else:
                    consecutive_empty_episodes += 1
                    if consecutive_empty_episodes >= 3:
                        raise OnlineGRPOError(
                            "three consecutive episodes produced no feasible "
                            "online state rollout for training bucket "
                            f"{bucket_index}: {scenario[0]}/{scenario[1]} "
                            f"seed={seed}"
                        )
            else:
                consecutive_empty_episodes = 0
            if _bucket_visit_is_complete(
                bucket_index=bucket_index,
                bucket_sample_counts=bucket_accepted_update_counts,
                bucket_target_counts=bucket_target_counts,
                current_visit_progress=current_visit_progress,
                rollout_groups_per_bucket_visit=(
                    config.rollout_groups_per_bucket_visit
                ),
            ):
                current_visit_progress = 0
                unfinished_bucket = _next_unfinished_bucket_index(
                    bucket_accepted_update_counts,
                    bucket_target_counts,
                    start_index=(bucket_index + 1) % len(training_buckets),
                )
                next_bucket_index = (
                    0 if unfinished_bucket is None else unfinished_bucket
                )
                temporary_start_offset_upper_bound = None
            else:
                next_bucket_index = bucket_index
            if attempt_budget_exhausted:
                break
            if stability_guard_diagnostic is not None:
                break
            if accepted_update_states != last_validated_update_state and (
                accepted_update_states == target_accepted_update_states
                or (
                    accepted_update_states > 0
                    and accepted_update_states % config.validation_interval_rollouts
                    == 0
                )
            ):
                validation, simulator_errors = (
                    _fixed_raw_proxy_and_simulator_validation(
                        trainer,
                        device=torch_device,
                        reward_config=joint_diagnostic_reward_config,
                        scenarios=config.scenarios,
                        seeds=HOLDOUT_SEEDS,
                        validation_state_bank=validation_state_bank,
                        frozen_cache=fixed_validation_cache,
                    )
                )
                validation_calls += 1
                for name, value in validation.items():
                    if name.startswith("perf/") and name.endswith("_seconds"):
                        performance_totals[name] += float(value)
                validation.update(
                    _validation_reward_comparison_metrics(
                        validation, pretrain_validation
                    )
                )
                for value in simulator_errors:
                    error_record = {
                        "optimizer_step": int(trainer.optimizer_step),
                        "accepted_update_state": int(accepted_update_states),
                        "stage1_baseline": False,
                        **dict(value),
                    }
                    simulator_diagnostic_errors.append(error_record)
                    writer.add_text(
                        "validation/simulator_error",
                        json.dumps(error_record, sort_keys=True),
                        accepted_update_states,
                    )
                last_metrics.update(validation)
                for metric_name, metric_value in validation.items():
                    writer.add_scalar(metric_name, metric_value, accepted_update_states)
                last_validated_update_state = accepted_update_states
                current_sampler_state = _sampler_state(
                    accepted_update_states=accepted_update_states,
                    sampling_attempts=sampling_attempts,
                    rejected_sampling_attempts=rejected_sampling_attempts,
                    stability_guard_rejections=stability_guard_rejections,
                    exhausted_states=exhausted_states,
                    baseline_execution_steps=baseline_execution_steps,
                    zero_signal_epochs=zero_signal_epochs,
                    warmup_environment_steps=warmup_environment_steps,
                    bucket_target_counts=bucket_target_counts,
                    bucket_accepted_update_counts=(bucket_accepted_update_counts),
                    bucket_sampling_attempt_counts=(bucket_sampling_attempt_counts),
                    bucket_rejected_sampling_attempt_counts=(
                        bucket_rejected_sampling_attempt_counts
                    ),
                    bucket_stability_guard_rejection_counts=(
                        bucket_stability_guard_rejection_counts
                    ),
                    bucket_exhausted_state_counts=(bucket_exhausted_state_counts),
                    bucket_baseline_execution_step_counts=(
                        bucket_baseline_execution_step_counts
                    ),
                    bucket_zero_signal_epoch_counts=(bucket_zero_signal_epoch_counts),
                    bucket_optimizer_step_counts=(bucket_optimizer_step_counts),
                    bucket_episode_counts=bucket_episode_counts,
                    next_bucket_index=next_bucket_index,
                    current_visit_progress=current_visit_progress,
                    rollout_start_generator_state=(
                        rollout_start_generator.get_state()
                    ),
                    last_validated_update_state=last_validated_update_state,
                    rollout_groups_per_bucket_visit=(
                        config.rollout_groups_per_bucket_visit
                    ),
                    optimizer_step=trainer.optimizer_step,
                    environment_steps=environment_steps,
                    max_sampling_attempts=max_sampling_attempts,
                )
                safety_eligible, selection_score = (
                    _append_validation_selection_event(
                        validation_selection_history,
                        accepted_update_state=accepted_update_states,
                        validation=validation,
                        pretrain_validation=pretrain_validation,
                    )
                )
                validation["validation/checkpoint_safety_eligible"] = float(
                    safety_eligible
                )
                if selection_score is not None:
                    validation[
                        "validation/simulator_reward_gain_trailing3"
                    ] = selection_score[0]
                    validation[
                        "validation/selected_reward_gain_trailing3"
                    ] = selection_score[1]
                last_metrics.update(validation)
                for metric_name in (
                    "validation/checkpoint_safety_eligible",
                    "validation/simulator_reward_gain_trailing3",
                    "validation/selected_reward_gain_trailing3",
                ):
                    if metric_name in validation:
                        writer.add_scalar(
                            metric_name,
                            validation[metric_name],
                            accepted_update_states,
                        )
                checkpoint_started = time.perf_counter()
                if selection_score is not None:
                    if (
                        best_unconstrained_score is None
                        or selection_score > best_unconstrained_score
                    ):
                        best_unconstrained_score = selection_score
                        unconstrained_checkpoint = _checkpoint_payload(
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
                            best_selected_reward_gain=(
                                best_selected_reward_gain
                            ),
                            best_checkpoint_sha256=best_checkpoint_sha256,
                            validation_selection_history=(
                                validation_selection_history
                            ),
                            collection_contract=collection_contract,
                            sampler_state=current_sampler_state,
                        )
                        save_grpo_checkpoint(
                            best_unconstrained_path, unconstrained_checkpoint
                        )
                if (
                    safety_eligible
                    and selection_score is not None
                    and selection_score[0] >= 0.0
                    and (
                    best_reward is None
                    or selection_score
                    > (best_reward, float(best_selected_reward_gain))
                    )
                ):
                    best_reward, best_selected_reward_gain = selection_score
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
                        best_selected_reward_gain=best_selected_reward_gain,
                        best_checkpoint_sha256=None,
                        validation_selection_history=validation_selection_history,
                        collection_contract=collection_contract,
                        sampler_state=current_sampler_state,
                    )
                    save_grpo_checkpoint(best_path, best_checkpoint)
                    save_grpo_checkpoint(best_safe_path, best_checkpoint)
                    best_checkpoint_sha256 = _checkpoint_file_sha256(best_path)
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
                    best_selected_reward_gain=best_selected_reward_gain,
                    best_checkpoint_sha256=best_checkpoint_sha256,
                    validation_selection_history=validation_selection_history,
                    collection_contract=collection_contract,
                    sampler_state=current_sampler_state,
                )
                save_grpo_checkpoint(last_path, checkpoint)
                if accepted_update_states in (200, 500):
                    save_grpo_checkpoint(
                        milestone_200_path
                        if accepted_update_states == 200
                        else milestone_500_path,
                        checkpoint,
                    )
                checkpoint_seconds = time.perf_counter() - checkpoint_started
                checkpoint_metric = "perf/validation/checkpoint_seconds"
                validation[checkpoint_metric] = float(checkpoint_seconds)
                performance_totals[checkpoint_metric] += float(checkpoint_seconds)
                last_metrics[checkpoint_metric] = float(checkpoint_seconds)
                writer.add_scalar(
                    checkpoint_metric,
                    float(checkpoint_seconds),
                    accepted_update_states,
                )
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "event": "validation_performance",
                                "accepted_update_state": int(
                                    accepted_update_states
                                ),
                                **{
                                    name: float(value)
                                    for name, value in validation.items()
                                    if name.startswith("perf/")
                                },
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                if accepted_update_states >= 200 and len(
                    validation_selection_history
                ) >= 3:
                    trailing = validation_selection_history[-3:]
                    diagnostic_early_stop = all(
                        value["simulator_reward_gain"] < 0.0
                        and value["selected_reward_gain"] < 0.0
                        and value["s7_out_delta"] > 0.0
                        for value in trailing
                    )
    finally:
        writer.close()

    if stability_guard_diagnostic is not None:
        guard_sampler_state = _sampler_state(
            accepted_update_states=accepted_update_states,
            sampling_attempts=sampling_attempts,
            rejected_sampling_attempts=rejected_sampling_attempts,
            stability_guard_rejections=stability_guard_rejections,
            exhausted_states=exhausted_states,
            baseline_execution_steps=baseline_execution_steps,
            zero_signal_epochs=zero_signal_epochs,
            warmup_environment_steps=warmup_environment_steps,
            bucket_target_counts=bucket_target_counts,
            bucket_accepted_update_counts=bucket_accepted_update_counts,
            bucket_sampling_attempt_counts=bucket_sampling_attempt_counts,
            bucket_rejected_sampling_attempt_counts=(
                bucket_rejected_sampling_attempt_counts
            ),
            bucket_stability_guard_rejection_counts=(
                bucket_stability_guard_rejection_counts
            ),
            bucket_exhausted_state_counts=bucket_exhausted_state_counts,
            bucket_baseline_execution_step_counts=(
                bucket_baseline_execution_step_counts
            ),
            bucket_zero_signal_epoch_counts=bucket_zero_signal_epoch_counts,
            bucket_optimizer_step_counts=bucket_optimizer_step_counts,
            bucket_episode_counts=bucket_episode_counts,
            next_bucket_index=next_bucket_index,
            current_visit_progress=current_visit_progress,
            rollout_start_generator_state=rollout_start_generator.get_state(),
            last_validated_update_state=last_validated_update_state,
            rollout_groups_per_bucket_visit=config.rollout_groups_per_bucket_visit,
            optimizer_step=trainer.optimizer_step,
            environment_steps=environment_steps,
            max_sampling_attempts=max_sampling_attempts,
        )
        guard_payload = _checkpoint_payload(
            variant=variant,
            trainer=trainer,
            source_sha=source_sha,
            source_payload=source_payload,
            metrics=last_metrics,
            diagnostic_only=True,
            run_mode=run_mode,
            reward_config=reward_config,
            scenario_contract_sha=scenario_contract_sha,
            scenario_seeds=config.scenario_seeds,
            environment_steps=environment_steps,
            best_validation_reward=best_reward,
            best_selected_reward_gain=best_selected_reward_gain,
            best_checkpoint_sha256=best_checkpoint_sha256,
            validation_selection_history=validation_selection_history,
            collection_contract=collection_contract,
            sampler_state=guard_sampler_state,
        )
        guard_payload["training_status"] = "stability_guard_rejected"
        guard_path = save_grpo_checkpoint(
            run_dir / "checkpoints" / "stability_guard.pt", guard_payload
        )
        guard_record = {
            "format": "stage2_grpo_stability_guard_diagnostic_v2",
            "training_status": "stability_guard_rejected",
            "stability_guard_rejections": stability_guard_rejections,
            "stability_guard_rejections_this_run": (
                stability_guard_rejections_this_run
            ),
            "optimizer_steps": trainer.optimizer_step,
            "accepted_update_states": accepted_update_states,
            "baseline_execution_steps": baseline_execution_steps,
            "environment_steps": environment_steps,
            "checkpoint": str(guard_path.resolve()),
            **stability_guard_diagnostic,
        }
        (run_dir / "stability_guard.json").write_text(
            json.dumps(guard_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = {
            "format": "bev_joint_grpo_online_report_v14",
            "training_status": "stability_guard_rejected",
            "diagnostic_only": True,
            "eligible_for_formal_training": False,
            "accepted_update_states": accepted_update_states,
            "target_accepted_update_states": target_accepted_update_states,
            "optimizer_steps": trainer.optimizer_step,
            "sampling_attempts": sampling_attempts,
            "rejected_sampling_attempts": rejected_sampling_attempts,
            "exhausted_states": exhausted_states,
            "baseline_execution_steps": baseline_execution_steps,
            "environment_steps": environment_steps,
            "stability_guard_rejections": stability_guard_rejections,
            "stability_guard_rejections_this_run": (
                stability_guard_rejections_this_run
            ),
            "stability_guard": guard_record,
            "last_checkpoint": str(guard_path.resolve()),
            "wall_time_seconds": time.monotonic() - started_at,
            "performance": _performance_summary(
                performance_totals, validation_calls=validation_calls
            ),
        }
        (run_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report

    if attempt_budget_exhausted:
        incomplete_sampler_state = _sampler_state(
            accepted_update_states=accepted_update_states,
            sampling_attempts=sampling_attempts,
            rejected_sampling_attempts=rejected_sampling_attempts,
            stability_guard_rejections=stability_guard_rejections,
            exhausted_states=exhausted_states,
            baseline_execution_steps=baseline_execution_steps,
            zero_signal_epochs=zero_signal_epochs,
            warmup_environment_steps=warmup_environment_steps,
            bucket_target_counts=bucket_target_counts,
            bucket_accepted_update_counts=bucket_accepted_update_counts,
            bucket_sampling_attempt_counts=bucket_sampling_attempt_counts,
            bucket_rejected_sampling_attempt_counts=(
                bucket_rejected_sampling_attempt_counts
            ),
            bucket_stability_guard_rejection_counts=(
                bucket_stability_guard_rejection_counts
            ),
            bucket_exhausted_state_counts=bucket_exhausted_state_counts,
            bucket_baseline_execution_step_counts=(
                bucket_baseline_execution_step_counts
            ),
            bucket_zero_signal_epoch_counts=(bucket_zero_signal_epoch_counts),
            bucket_optimizer_step_counts=bucket_optimizer_step_counts,
            bucket_episode_counts=bucket_episode_counts,
            next_bucket_index=next_bucket_index,
            current_visit_progress=current_visit_progress,
            rollout_start_generator_state=rollout_start_generator.get_state(),
            last_validated_update_state=last_validated_update_state,
            rollout_groups_per_bucket_visit=(config.rollout_groups_per_bucket_visit),
            optimizer_step=trainer.optimizer_step,
            environment_steps=environment_steps,
            max_sampling_attempts=max_sampling_attempts,
        )
        incomplete_checkpoint = _checkpoint_payload(
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
            best_selected_reward_gain=best_selected_reward_gain,
            best_checkpoint_sha256=best_checkpoint_sha256,
            validation_selection_history=validation_selection_history,
            collection_contract=collection_contract,
            sampler_state=incomplete_sampler_state,
        )
        incomplete_checkpoint["training_status"] = "incomplete_attempt_budget_exhausted"
        last_path = save_grpo_checkpoint(last_path, incomplete_checkpoint)
        incomplete_report = {
            "format": "bev_joint_grpo_online_report_v14",
            "training_status": "incomplete_attempt_budget_exhausted",
            "variant": variant,
            "run_mode": run_mode,
            "accepted_update_states": accepted_update_states,
            "accepted_update_states_this_run": (
                accepted_update_states - run_start_accepted_update_state
            ),
            "target_accepted_update_states": (target_accepted_update_states),
            "sampling_attempts": sampling_attempts,
            "sampling_attempts_this_run": (
                sampling_attempts - run_start_sampling_attempt
            ),
            "max_sampling_attempts": max_sampling_attempts,
            "rejected_sampling_attempts": rejected_sampling_attempts,
            "exhausted_states": exhausted_states,
            "baseline_execution_steps": baseline_execution_steps,
            "zero_signal_epochs": zero_signal_epochs,
            "stability_guard_rejections": stability_guard_rejections,
            "stability_guard_rejections_this_run": (
                stability_guard_rejections_this_run
            ),
            "warmup_environment_steps": warmup_environment_steps,
            "environment_steps": environment_steps,
            "optimizer_steps": trainer.optimizer_step,
            "wall_time_seconds": time.monotonic() - started_at,
            "performance": _performance_summary(
                performance_totals, validation_calls=validation_calls
            ),
            "cuda_peak_memory_bytes": _cuda_peak_memory_bytes(torch_device),
            "rollout_collection_contract": collection_contract,
            "training_bucket_counters": [
                {
                    "scenario": scenario[0],
                    "route": scenario[1],
                    "seed": seed,
                    "target_accepted_updates": bucket_target_counts[index],
                    "accepted_update_states": (bucket_accepted_update_counts[index]),
                    "sampling_attempts": (bucket_sampling_attempt_counts[index]),
                    "rejected_sampling_attempts": (
                        bucket_rejected_sampling_attempt_counts[index]
                    ),
                    "stability_guard_rejections": (
                        bucket_stability_guard_rejection_counts[index]
                    ),
                    "exhausted_states": (bucket_exhausted_state_counts[index]),
                    "baseline_execution_steps": (
                        bucket_baseline_execution_step_counts[index]
                    ),
                    "zero_signal_epochs": bucket_zero_signal_epoch_counts[index],
                    "environment_episodes": bucket_episode_counts[index],
                    "optimizer_steps": bucket_optimizer_step_counts[index],
                }
                for index, (scenario, seed) in enumerate(training_buckets)
            ],
            "last_checkpoint": str(last_path.resolve()),
        }
        incomplete_report_path = run_dir / "report.json"
        incomplete_report_path.write_text(
            json.dumps(incomplete_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise OnlineGRPOError(
            "dynamic sampling exhausted max_sampling_attempts before reaching "
            "target_accepted_update_states; incomplete checkpoint "
            f"and report saved at {last_path} and {incomplete_report_path}"
        )

    if not validation_selection_history:
        raise OnlineGRPOError("online GRPO never completed fixed validation")
    final_sampler_state = _sampler_state(
        accepted_update_states=accepted_update_states,
        sampling_attempts=sampling_attempts,
        rejected_sampling_attempts=rejected_sampling_attempts,
        stability_guard_rejections=stability_guard_rejections,
        exhausted_states=exhausted_states,
        baseline_execution_steps=baseline_execution_steps,
        zero_signal_epochs=zero_signal_epochs,
        warmup_environment_steps=warmup_environment_steps,
        bucket_target_counts=bucket_target_counts,
        bucket_accepted_update_counts=bucket_accepted_update_counts,
        bucket_sampling_attempt_counts=bucket_sampling_attempt_counts,
        bucket_rejected_sampling_attempt_counts=(
            bucket_rejected_sampling_attempt_counts
        ),
        bucket_stability_guard_rejection_counts=(
            bucket_stability_guard_rejection_counts
        ),
        bucket_exhausted_state_counts=bucket_exhausted_state_counts,
        bucket_baseline_execution_step_counts=(bucket_baseline_execution_step_counts),
        bucket_zero_signal_epoch_counts=(bucket_zero_signal_epoch_counts),
        bucket_optimizer_step_counts=bucket_optimizer_step_counts,
        bucket_episode_counts=bucket_episode_counts,
        next_bucket_index=next_bucket_index,
        current_visit_progress=current_visit_progress,
        rollout_start_generator_state=rollout_start_generator.get_state(),
        last_validated_update_state=last_validated_update_state,
        rollout_groups_per_bucket_visit=(config.rollout_groups_per_bucket_visit),
        optimizer_step=trainer.optimizer_step,
        environment_steps=environment_steps,
        max_sampling_attempts=max_sampling_attempts,
    )
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
        best_selected_reward_gain=best_selected_reward_gain,
        best_checkpoint_sha256=best_checkpoint_sha256,
        validation_selection_history=validation_selection_history,
        collection_contract=collection_contract,
        sampler_state=final_sampler_state,
    )
    last_path = save_grpo_checkpoint(last_path, payload)
    cuda_peak_memory_bytes = _cuda_peak_memory_bytes(torch_device)

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
        collection_contract=collection_contract,
        bucket_count=len(training_buckets),
        bucket_target_counts=bucket_target_counts,
        rollout_groups_per_bucket_visit=(config.rollout_groups_per_bucket_visit),
        optimizer_step=restored.optimizer_step,
        max_sampling_attempts=max_sampling_attempts,
    )

    plot_paths = generate_grpo_plots(
        run_dir / "tb",
        run_dir / "plots",
        require_frozen_pretrain_reward=True,
        rollout_axis_label=ACCEPTED_ROLLOUT_AXIS_LABEL,
    )
    if rollout_start_accepted_count + rollout_start_rejected_count != len(
        rollout_start_offsets_this_run
    ):
        raise OnlineGRPOError(
            "rollout start diagnostics contain an unresolved sampling attempt"
        )

    report = {
        "format": "bev_joint_grpo_online_report_v14",
        "implementation_commit": implementation_commit,
        "training_status": (
            "diagnostic_early_stop" if diagnostic_early_stop else "complete"
        ),
        "variant": variant,
        "run_mode": run_mode,
        "diagnostic_only": run_mode != "formal",
        "eligible_for_formal_training": run_mode == "formal",
        "grpo_config": _grpo_config_artifact_payload(grpo_config),
        "optimizer_steps": trainer.optimizer_step,
        "optimizer_steps_this_run": (trainer.optimizer_step - run_start_optimizer_step),
        "environment_steps": environment_steps,
        "environment_episode_count": sum(bucket_episode_counts),
        "warmup_environment_steps": warmup_environment_steps,
        "accepted_update_states": accepted_update_states,
        "accepted_update_states_this_run": (
            accepted_update_states - run_start_accepted_update_state
        ),
        "target_accepted_update_states": target_accepted_update_states,
        "sampling_attempts": sampling_attempts,
        "sampling_attempts_this_run": (sampling_attempts - run_start_sampling_attempt),
        "max_sampling_attempts": max_sampling_attempts,
        "rejected_sampling_attempts": rejected_sampling_attempts,
        "exhausted_states": exhausted_states,
        "baseline_execution_steps": baseline_execution_steps,
        "zero_signal_epochs": zero_signal_epochs,
        "stability_guard_rejections": stability_guard_rejections,
        "stability_guard_rejections_this_run": (
            stability_guard_rejections_this_run
        ),
        "rollout_start_diagnostics_this_run": {
            "attempt_count": len(rollout_start_offsets_this_run),
            "accepted_count": rollout_start_accepted_count,
            "rejected_count": rollout_start_rejected_count,
            "offset_min_steps": (
                min(rollout_start_offsets_this_run)
                if rollout_start_offsets_this_run
                else None
            ),
            "offset_mean_steps": (
                float(np.mean(rollout_start_offsets_this_run))
                if rollout_start_offsets_this_run
                else None
            ),
            "offset_max_steps": (
                max(rollout_start_offsets_this_run)
                if rollout_start_offsets_this_run
                else None
            ),
        },
        "wall_time_seconds": time.monotonic() - started_at,
        "performance": _performance_summary(
            performance_totals, validation_calls=validation_calls
        ),
        "cuda_peak_memory_bytes": cuda_peak_memory_bytes,
        "v2_rule_conditioning": {
            "enabled": True,
            **rule_diagnostics,
        },
        "reward_contract_version": _reward_contract_version(),
        "reward_contract_sha256": VEHICLE_MODE_REWARD_CONTRACT_SHA256,
        "reward_config_sha256": vehicle_mode_reward_config_sha256(reward_config),
        "reward_application_contract": GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
        "reward_application_contract_sha256": (
            GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
        ),
        "reward_input_domain": "tau_d",
        "training_candidate_domain": "tau_d_all_vehicle_modes",
        "environment_action_source": "cached_frozen_stage1_argmax",
        "execution_input_domain": "tau_cmd",
        "frozen_pretrain_reward_logging": frozen_pretrain_reward_logging,
        "best_checkpoint_metric": BEST_CHECKPOINT_METRIC,
        "joint_reward_role": "historical_and_final_evaluation_only",
        "tracking_expansion_enabled": False,
        "calibration_required": False,
        "policy_update_contract": joint_grpo_optimizer_contract(),
        "policy_update_contract_sha256": joint_grpo_optimizer_contract_sha256(),
        "rollout_collection_contract": collection_contract,
        "simulator_validation_role": "diagnostic_only",
        "simulator_diagnostic_errors": simulator_diagnostic_errors,
        "scenario_contract_sha256": scenario_contract_sha,
        "training_scenarios": [list(value) for value in config.scenarios],
        "training_seeds": [int(value) for value in config.scenario_seeds],
        "training_bucket_counters": [
            {
                "scenario": scenario[0],
                "route": scenario[1],
                "seed": seed,
                "target_accepted_updates": bucket_target_counts[index],
                "accepted_update_states": bucket_accepted_update_counts[index],
                "sampling_attempts": bucket_sampling_attempt_counts[index],
                "rejected_sampling_attempts": (
                    bucket_rejected_sampling_attempt_counts[index]
                ),
                "stability_guard_rejections": (
                    bucket_stability_guard_rejection_counts[index]
                ),
                "exhausted_states": (bucket_exhausted_state_counts[index]),
                "baseline_execution_steps": (
                    bucket_baseline_execution_step_counts[index]
                ),
                "zero_signal_epochs": bucket_zero_signal_epoch_counts[index],
                "environment_episodes": bucket_episode_counts[index],
                "optimizer_steps": bucket_optimizer_step_counts[index],
            }
            for index, (scenario, seed) in enumerate(training_buckets)
        ],
        "validation_seeds": list(HOLDOUT_SEEDS),
        "last_checkpoint": str(last_path.resolve()),
        "best_checkpoint": (
            str(best_path.resolve()) if best_checkpoint_sha256 is not None else None
        ),
        "best_safe_checkpoint": (
            str(best_safe_path.resolve())
            if best_checkpoint_sha256 is not None
            else None
        ),
        "best_simulator_unconstrained_checkpoint": (
            str(best_unconstrained_path.resolve())
            if best_unconstrained_path.exists()
            else None
        ),
        "checkpoint_selection_status": (
            "eligible" if best_checkpoint_sha256 is not None else "no_eligible_checkpoint"
        ),
        "best_validation_reward": best_reward,
        "best_selected_reward_gain": best_selected_reward_gain,
        "validation_selection_history": validation_selection_history,
        "metrics": last_metrics,
        "checkpoint_round_trip": True,
        "advantage_vector_logging": {
            "storage": "tensorboard_tensor",
            "tensorboard_tag": ADVANTAGE_VECTOR_TAG,
            "tensorboard_dir": str((run_dir / "tb").resolve()),
            "shape": [
                1,
                3,
                10,
                int(trainer.config.trajectories_per_mode),
            ],
            "interval_rollout_groups": (config.advantage_vector_log_interval_rollouts),
            "record_count": advantage_vector_record_count,
            "scope": "current_run_only",
            "run_start_optimizer_step": run_start_optimizer_step,
            "run_start_accepted_update_state": (run_start_accepted_update_state),
            "group_axis_semantics": "vehicle_mode_trajectory_sample",
            "heatmap": str(plot_paths["advantage_heatmap"].resolve()),
            "x_axis": "absolute_accepted_update_state",
        },
        "training_plots": {
            "reward_curve": str(plot_paths["reward_curve"].resolve()),
            "reward_tags": list(REWARD_CURVE_TAGS),
            "validation_reward_curve": str(
                plot_paths["validation_reward_curve"].resolve()
            ),
            "validation_reward_tags": list(VALIDATION_REWARD_CURVE_TAGS),
            "reward_domain": "raw_tau_d",
            "grpo_loss_curve": str(plot_paths["grpo_loss_curve"].resolve()),
            "grpo_loss_tags": [
                "loss/total",
                "loss/trajectory_pg",
            ],
            "kl_loss_curve": str(plot_paths["kl_loss_curve"].resolve()),
            "kl_loss_tags": [
                "loss/reference_kl",
                "loss/trajectory_reference_kl",
            ],
            "reference_kl_weighted": False,
            "policy_stability_curve": str(
                plot_paths["policy_stability_curve"].resolve()
            ),
            "reward_x_axis": "absolute_accepted_update_state",
            "validation_reward_x_axis": ("absolute_accepted_update_state"),
            "optimizer_x_axis": "absolute_accepted_update_state",
        },
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _config_from_yaml(path: Path) -> JointGRPOTrainingConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise OnlineGRPOError(f"unable to read online GRPO config: {path}") from exc
    if not isinstance(payload, Mapping):
        raise OnlineGRPOError("online GRPO YAML root must be a mapping")
    run = payload.get("run")
    if not isinstance(run, Mapping):
        raise OnlineGRPOError("online GRPO YAML requires a run mapping")
    required_run_fields = {"variant", "run_mode", "source_checkpoint"}
    if (
        any(not isinstance(name, str) for name in run)
        or set(run) != required_run_fields
    ):
        raise OnlineGRPOError(
            "online GRPO YAML run mapping must contain exactly variant, "
            "run_mode, and source_checkpoint"
        )
    variant = run["variant"]
    if not isinstance(variant, str) or variant not in ("A", "B"):
        raise OnlineGRPOError("online GRPO variant must be A or B")
    run_mode = run["run_mode"]
    if not isinstance(run_mode, str) or run_mode not in ("formal", "smoke"):
        raise OnlineGRPOError("run_mode must be formal or smoke")
    source_checkpoint = run["source_checkpoint"]
    if not isinstance(source_checkpoint, str) or not source_checkpoint.strip():
        raise OnlineGRPOError("source_checkpoint must be a non-empty path string")
    online = payload.get("online")
    if not isinstance(online, Mapping):
        raise OnlineGRPOError("online GRPO YAML requires an online mapping")
    legacy_fields = {
        "total_optimizer_steps",
        "validation_interval_steps",
        "advantage_vector_log_interval_steps",
        "clip_epsilon",
        "clip_epsilon_low",
        "clip_epsilon_high",
        "update_epochs",
        "group_size",
        "pretrain_improvement_margin",
        "max_candidate_groups_per_state",
        "max_attempted_groups_multiplier",
    }
    present_legacy = sorted(legacy_fields.intersection(online))
    if present_legacy:
        raise OnlineGRPOError(
            "online GRPO YAML contains legacy fields: " + ", ".join(present_legacy)
        )
    scenarios = tuple(
        (str(value["scenario"]), str(value["route"]))
        for value in online.get("scenarios", ())
        if isinstance(value, Mapping)
    )
    online_config = JointGRPOOnlineConfig(
        device=str(online.get("device", "cuda")),
        seed=online.get("seed", 17),
        trajectories_per_mode=online.get("trajectories_per_mode", 48),
        total_rollout_groups=online.get("total_rollout_groups", 100),
        resume_checkpoint=(
            Path(str(online["resume_checkpoint"]))
            if online.get("resume_checkpoint")
            else None
        ),
        validation_state_bank=Path(
            str(
                online.get(
                    "validation_state_bank",
                    "evaluation/artifacts/grpo_validation_state_bank_v1.pt",
                )
            )
        ),
        scenarios=scenarios or PRIMARY_S5_S9_SCENARIOS,
        scenario_seeds=tuple(
            int(value) for value in online.get("scenario_seeds", DEVELOPMENT_SEEDS)
        ),
        environment_steps_per_episode=int(
            online.get("environment_steps_per_episode", 100)
        ),
        rollout_groups_per_bucket_visit=online.get(
            "rollout_groups_per_bucket_visit", 10
        ),
        rollout_start_offset_max_steps=online.get(
            "rollout_start_offset_max_steps", 200
        ),
        rollout_start_min_remaining_steps=online.get(
            "rollout_start_min_remaining_steps", 10
        ),
        validation_interval_rollouts=online.get("validation_interval_rollouts", 20),
        advantage_vector_log_interval_rollouts=online.get(
            "advantage_vector_log_interval_rollouts", 10
        ),
        max_sampling_attempts_per_state=online.get(
            "max_sampling_attempts_per_state", 3
        ),
        max_sampling_attempts_multiplier=online.get(
            "max_sampling_attempts_multiplier", 3
        ),
    )
    return JointGRPOTrainingConfig(
        variant=variant,
        run_mode=run_mode,
        source_checkpoint=Path(source_checkpoint),
        online=online_config,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    training_config = _config_from_yaml(arguments.config)
    report = run_joint_grpo_training(
        training_config,
        run_dir=arguments.run_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = [
    "DEVELOPMENT_SEEDS",
    "HOLDOUT_SEEDS",
    "PRIMARY_S5_S9_SCENARIOS",
    "JointGRPOOnlineConfig",
    "JointGRPOTrainingConfig",
    "OnlineGRPOError",
    "constant_velocity_actions",
    "episode_has_ended",
    "execute_cached_frozen_baseline",
    "build_grpo_validation_state_bank",
    "joint_trajectory_action",
    "model_inputs_to_batch",
    "optimize_selected_model_trajectories",
    "run_joint_grpo_training",
]


if __name__ == "__main__":
    raise SystemExit(main())
