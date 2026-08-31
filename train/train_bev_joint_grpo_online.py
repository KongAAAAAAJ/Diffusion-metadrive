"""Online raw-diffusion proxy-reward joint GRPO training for Variants A/B."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from evaluation.plot_grpo import (
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
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256,
    JOINT_REWARD_CONTRACT,
    JOINT_REWARD_CONTRACT_SHA256,
    JointGRPOConfig,
    JointGRPOPolicyUpdateConfig,
    JointRewardConfig,
    JointRewardError,
    JointTrajectoryProxyReward,
    KinematicTrajectoryOptimizer,
    KinematicTrajectoryOptimizerConfig,
    TrajectoryOptimizationError,
    TrajectoryOptimizationResult,
    joint_reward_config_sha256,
    joint_grpo_optimizer_contract,
    joint_grpo_optimizer_contract_sha256,
    joint_grpo_transition_noise_contract,
    joint_grpo_transition_noise_contract_sha256,
    normalize_signed_advantages,
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
ROLLOUT_COLLECTION_CONTRACT_VERSION = (
    "stage2_joint_grpo_persistent_episode_v2"
)


class OnlineGRPOError(RuntimeError):
    """Raised when online raw-domain GRPO violates its contract."""


def _frozen_pretrain_reward_logging_metadata(
    planner: torch.nn.Module,
) -> dict[str, object]:
    """Describe the per-live-state frozen Stage1 reward diagnostic."""

    planner_config = planner.config
    return {
        "tensorboard_tag": FROZEN_PRETRAIN_RAW_PROXY_REWARD_TAG,
        "metrics_jsonl_field": FROZEN_PRETRAIN_RAW_PROXY_REWARD_TAG,
        "step_axis": "absolute_fresh_rollout_group",
        "same_live_state_as_current_exploration": True,
        "trajectory_count": 1,
        "trajectory_selection": "valid_mode_masked_argmax",
        "trajectory_domain": "raw_tau_d",
        "inference": "frozen_stage1_standard_deterministic_ddim",
        "inference_seed": int(planner_config.inference_seed),
        "inference_noise_timestep": int(
            planner_config.inference_noise_timestep
        ),
        "inference_denoise_steps": int(
            planner_config.inference_denoise_steps
        ),
        "fixed_inference_noise": True,
        "diagnostic_only": True,
        "affects_training_or_environment_action": False,
    }


@dataclass(frozen=True)
class JointGRPOOnlineConfig:
    device: str = "cuda"
    seed: int = 17
    group_size: int = 24
    total_rollout_groups: int = 100
    update_epochs: int = 4
    clip_epsilon: float = 0.2
    resume_checkpoint: Path | None = None
    scenarios: tuple[tuple[str, str], ...] = PRIMARY_S5_S9_SCENARIOS
    scenario_seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    environment_steps_per_episode: int = 100
    rollout_groups_per_bucket_visit: int = 10
    rollout_start_offset_max_steps: int = 200
    rollout_start_min_remaining_steps: int = 10
    validation_interval_rollouts: int = 20
    advantage_vector_log_interval_rollouts: int = 20

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
            "total_rollout_groups",
            "update_epochs",
            "environment_steps_per_episode",
            "rollout_groups_per_bucket_visit",
            "validation_interval_rollouts",
            "advantage_vector_log_interval_rollouts",
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
                "rollout_start_min_remaining_steps must be a non-negative "
                "integer"
            )
        if self.environment_steps_per_episode < 10:
            raise OnlineGRPOError(
                "environment_steps_per_episode must be at least 10"
            )
        if (
            self.rollout_start_min_remaining_steps
            < self.rollout_groups_per_bucket_visit
        ):
            raise OnlineGRPOError(
                "rollout_start_min_remaining_steps must be greater than or "
                "equal to rollout_groups_per_bucket_visit"
            )
        if (
            self.environment_steps_per_episode
            < self.rollout_start_min_remaining_steps
        ):
            raise OnlineGRPOError(
                "environment_steps_per_episode must be greater than or equal "
                "to rollout_start_min_remaining_steps"
            )
        if (
            isinstance(self.clip_epsilon, bool)
            or not isinstance(self.clip_epsilon, (int, float))
            or not math.isfinite(float(self.clip_epsilon))
            or not 0.0 < float(self.clip_epsilon) < 1.0
        ):
            raise OnlineGRPOError(
                "clip_epsilon must be a finite scalar strictly between 0 and 1"
            )
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
        "budget_unit": "fresh_rollout_group",
        "bucket_order": "ordered_scenario_then_seed",
        "bucket_target_assignment": (
            "balanced_floor_with_remainder_to_lower_bucket_indices"
        ),
        "rollout_groups_per_bucket_visit": int(
            config.rollout_groups_per_bucket_visit
        ),
        "environment_steps_per_episode": int(
            config.environment_steps_per_episode
        ),
        "validation_interval_rollouts": int(
            config.validation_interval_rollouts
        ),
        "selected_environment_steps_per_rollout": 1,
        "uninformative_rollout_execution": "selected_action_once_no_update",
        "episode_reuse": "persistent_within_bucket_visit",
        "interrupted_visit_resume": "same_bucket_same_visit_progress",
        "checkpoint_boundary": "closed_environment_only",
        "active_environment_serialized": False,
        "rule_maker_commitment_scope": "live_environment_episode",
        "rollout_start_ready_gate": (
            "history_ready_and_primary_scenario_ready"
        ),
        "rollout_start_offset_distribution": "inclusive_uniform_integer",
        "rollout_start_generator": "shared_training_torch_generator",
        "rollout_start_offset_max_steps": int(
            config.rollout_start_offset_max_steps
        ),
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
            "same_bucket_same_visit_with_temporary_observed_last_open_upper_"
            "bound"
        ),
        "partial_visit_policy": (
            "retain_completed_rollouts_and_updates_then_new_episode_fresh_"
            "offset"
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
        else {
            agent_id: int(committed_plan_actions[agent_id])
            for agent_id in AGENT_IDS
        }
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
                feedback_actions[agent_id]
                == int(condition.rule_actions[agent_id])
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
        diagnostics["proposal_matches"] = int(not condition.is_commitment)
        resolved = optimization

    _validate_online_trajectory_controls(
        env, resolved.optimized_trajectories[0]
    )
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
        raise OnlineGRPOError(
            "rollout_group cannot exceed target_rollout_groups"
        )
    return (
        rollout_group % interval_rollouts == 0
        or rollout_group == target_rollout_groups
    )


def _write_advantage_vector_summary(
    writer: SummaryWriter,
    advantages: torch.Tensor,
    rollout_group: int,
    target_rollout_groups: int,
    interval_rollouts: int,
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
        rollout_group, target_rollout_groups, interval_rollouts
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
        "advantage/std": float(
            advantages.std(unbiased=False).detach().cpu()
        ),
        "advantage/min": float(advantages.min().detach().cpu()),
        "advantage/max": float(advantages.max().detach().cpu()),
    }


def _reject_exhausted_uninformative_budget(
    *,
    sampled_rollouts: int,
    target_rollout_groups: int,
    optimizer_step: int,
    run_start_optimizer_step: int,
) -> None:
    if (
        sampled_rollouts >= target_rollout_groups
        and optimizer_step == run_start_optimizer_step
    ):
        raise OnlineGRPOError(
            "rollout budget was exhausted but every fresh group was "
            "uninformative; no optimizer update occurred"
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


def _balanced_bucket_targets(
    total_rollout_groups: int, bucket_count: int
) -> list[int]:
    for name, value in (
        ("total_rollout_groups", total_rollout_groups),
        ("bucket_count", bucket_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise OnlineGRPOError(f"{name} must be a positive integer")
    base, remainder = divmod(total_rollout_groups, bucket_count)
    return [
        base + int(bucket_index < remainder)
        for bucket_index in range(bucket_count)
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
        or bucket_sample_counts[bucket_index]
        >= bucket_target_counts[bucket_index]
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
        return bool(
            route.get("all_agents_continued_on_exit_ramp", False)
        ) and bool(conflict.get("formation_recovered_on_ramp", False))
    if scenario_id == "S9_narrow_channel_negotiation":
        return bool(
            route.get("all_agents_returned_to_original_lane", False)
        ) and bool(conflict.get("formation_recovered_after_return", False))
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


def _policy_update_config(
    config: JointGRPOOnlineConfig,
) -> JointGRPOPolicyUpdateConfig:
    return JointGRPOPolicyUpdateConfig(
        update_epochs=config.update_epochs,
        clip_epsilon=float(config.clip_epsilon),
    )


def _grpo_config_artifact_payload(
    config: JointGRPOConfig,
) -> dict[str, object]:
    return dataclasses.asdict(config)


def _sampler_state(
    *,
    sampled_rollouts: int,
    uninformative_rollouts: int,
    bucket_target_counts: Sequence[int],
    bucket_sample_counts: Sequence[int],
    bucket_optimizer_step_counts: Sequence[int],
    bucket_episode_counts: Sequence[int],
    next_bucket_index: int,
    current_visit_progress: int,
    generator_state: torch.Tensor,
    last_validated_rollout: int,
    rollout_groups_per_bucket_visit: int,
    update_epochs: int,
    optimizer_step: int,
) -> dict[str, object]:
    raw = {
        "sampled_rollouts": sampled_rollouts,
        "uninformative_rollouts": uninformative_rollouts,
        "bucket_target_counts": list(bucket_target_counts),
        "bucket_sample_counts": list(bucket_sample_counts),
        "bucket_optimizer_step_counts": list(bucket_optimizer_step_counts),
        "bucket_episode_counts": list(bucket_episode_counts),
        "next_bucket_index": next_bucket_index,
        "current_visit_progress": current_visit_progress,
        "generator_state": generator_state.detach().cpu().clone(),
        "last_validated_rollout": last_validated_rollout,
    }
    return _validate_sampler_state(
        raw,
        bucket_count=len(bucket_sample_counts),
        expected_bucket_target_counts=bucket_target_counts,
        rollout_groups_per_bucket_visit=rollout_groups_per_bucket_visit,
        update_epochs=update_epochs,
        optimizer_step=optimizer_step,
    )


def _validate_sampler_state(
    raw: object,
    *,
    bucket_count: int,
    expected_bucket_target_counts: Sequence[int],
    rollout_groups_per_bucket_visit: int,
    update_epochs: int,
    optimizer_step: int,
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise OnlineGRPOError("online GRPO checkpoint sampler_state is invalid")
    if (
        isinstance(bucket_count, bool)
        or not isinstance(bucket_count, int)
        or bucket_count <= 0
    ):
        raise OnlineGRPOError("training bucket count must be a positive integer")
    if (
        isinstance(rollout_groups_per_bucket_visit, bool)
        or not isinstance(rollout_groups_per_bucket_visit, int)
        or rollout_groups_per_bucket_visit <= 0
    ):
        raise OnlineGRPOError("rollout bucket visit quota is invalid")
    for name, minimum in (
        ("sampled_rollouts", 0),
        ("uninformative_rollouts", 0),
        ("next_bucket_index", 0),
        ("current_visit_progress", 0),
        ("last_validated_rollout", -1),
    ):
        value = raw.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
        ):
            raise OnlineGRPOError(
                f"online GRPO checkpoint sampler_state {name} is invalid"
            )
    sampled = int(raw["sampled_rollouts"])
    uninformative = int(raw["uninformative_rollouts"])
    next_bucket = int(raw["next_bucket_index"])
    current_visit_progress = int(raw["current_visit_progress"])
    last_validated = int(raw["last_validated_rollout"])
    if uninformative > sampled or last_validated > sampled:
        raise OnlineGRPOError("online GRPO checkpoint sampler counters conflict")
    if next_bucket >= bucket_count:
        raise OnlineGRPOError("online GRPO checkpoint bucket cursor is invalid")

    counts: dict[str, list[int]] = {}
    for name in (
        "bucket_target_counts",
        "bucket_sample_counts",
        "bucket_optimizer_step_counts",
        "bucket_episode_counts",
    ):
        values = raw.get(name)
        if (
            not isinstance(values, (list, tuple))
            or len(values) != bucket_count
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
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
    if sum(counts["bucket_sample_counts"]) != sampled:
        raise OnlineGRPOError("online GRPO checkpoint bucket samples conflict")
    if sum(counts["bucket_optimizer_step_counts"]) != optimizer_step:
        raise OnlineGRPOError("online GRPO checkpoint bucket updates conflict")
    if optimizer_step != (sampled - uninformative) * update_epochs:
        raise OnlineGRPOError(
            "online GRPO checkpoint rollout and optimizer counters conflict"
        )
    if any(
        sampled_count > target_count
        for sampled_count, target_count in zip(
            counts["bucket_sample_counts"],
            counts["bucket_target_counts"],
        )
    ):
        raise OnlineGRPOError("online GRPO checkpoint bucket target exceeded")
    if any(
        sampled_count > 0 and episode_count == 0
        for sampled_count, episode_count in zip(
            counts["bucket_sample_counts"],
            counts["bucket_episode_counts"],
        )
    ):
        raise OnlineGRPOError("online GRPO checkpoint bucket episodes conflict")

    unfinished = _next_unfinished_bucket_index(
        counts["bucket_sample_counts"],
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
            counts["bucket_sample_counts"][next_bucket]
            % rollout_groups_per_bucket_visit
        )
        if current_visit_progress != expected_progress:
            raise OnlineGRPOError(
                "online GRPO checkpoint current visit progress is invalid"
            )
    for index, (sample_count, target_count) in enumerate(
        zip(
            counts["bucket_sample_counts"],
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

    generator_state = raw.get("generator_state")
    if (
        not isinstance(generator_state, torch.Tensor)
        or generator_state.dtype != torch.uint8
        or generator_state.ndim != 1
        or generator_state.numel() == 0
    ):
        raise OnlineGRPOError(
            "online GRPO checkpoint generator_state is invalid"
        )
    return {
        "sampled_rollouts": sampled,
        "uninformative_rollouts": uninformative,
        **counts,
        "next_bucket_index": next_bucket,
        "current_visit_progress": current_visit_progress,
        "generator_state": generator_state.detach().cpu().clone(),
        "last_validated_rollout": last_validated,
    }


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
    policy_update: JointGRPOPolicyUpdateConfig,
    collection_contract: Mapping[str, object],
    sampler_state: Mapping[str, object],
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
            "policy_update_contract": joint_grpo_optimizer_contract(
                policy_update
            ),
            "policy_update_contract_sha256": (
                joint_grpo_optimizer_contract_sha256(policy_update)
            ),
            "transition_noise_contract": (
                joint_grpo_transition_noise_contract()
            ),
            "transition_noise_contract_sha256": (
                joint_grpo_transition_noise_contract_sha256()
            ),
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
    reward_config: JointRewardConfig,
    scenario_contract_sha: str,
    scenario_seeds: Sequence[int],
    policy_update: JointGRPOPolicyUpdateConfig,
    collection_contract: Mapping[str, object],
    bucket_count: int,
    bucket_target_counts: Sequence[int],
    rollout_groups_per_bucket_visit: int,
    optimizer_step: int,
) -> dict[str, object]:
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
        "policy_update_contract": joint_grpo_optimizer_contract(policy_update),
        "policy_update_contract_sha256": (
            joint_grpo_optimizer_contract_sha256(policy_update)
        ),
        "transition_noise_contract": joint_grpo_transition_noise_contract(),
        "transition_noise_contract_sha256": (
            joint_grpo_transition_noise_contract_sha256()
        ),
        "rollout_collection_contract": dict(collection_contract),
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
    return _validate_sampler_state(
        payload.get("sampler_state"),
        bucket_count=bucket_count,
        expected_bucket_target_counts=bucket_target_counts,
        rollout_groups_per_bucket_visit=rollout_groups_per_bucket_visit,
        update_epochs=policy_update.update_epochs,
        optimizer_step=optimizer_step,
    )


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
                condition = _condition_online_model_inputs(
                    rule_maker,
                    env,
                    builder,
                    values,
                    committed_execution_id=None,
                    committed_plan_actions=None,
                )
                values = condition.model_inputs
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
        "policy_update_contract",
        "policy_update_contract_sha256",
        "transition_noise_contract",
        "transition_noise_contract_sha256",
        "rollout_collection_contract",
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
    training_config: JointGRPOTrainingConfig,
    *,
    run_dir: Path,
) -> dict[str, object]:
    if not isinstance(training_config, JointGRPOTrainingConfig):
        raise OnlineGRPOError(
            "training_config must be a JointGRPOTrainingConfig"
        )
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise OnlineGRPOError(
            f"run_dir must already exist: {run_dir}"
        )
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
        raise OnlineGRPOError(
            f"run_dir training.log must be a file: {training_log}"
        )
    (run_dir / "checkpoints").mkdir()

    started_at = time.monotonic()
    config = training_config.online
    variant = training_config.variant
    run_mode = training_config.run_mode
    source_checkpoint = training_config.source_checkpoint
    target_rollout_groups = config.total_rollout_groups

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
    policy_update = _policy_update_config(config)
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
    training_buckets = _round_robin_training_buckets(
        config.scenarios, config.scenario_seeds
    )
    bucket_target_counts = _balanced_bucket_targets(
        target_rollout_groups, len(training_buckets)
    )
    collection_contract = rollout_collection_contract(config)
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(config.seed)
    environment_steps = 0
    sampled_rollouts = 0
    uninformative_rollouts = 0
    bucket_sample_counts = [0 for _ in training_buckets]
    bucket_optimizer_step_counts = [0 for _ in training_buckets]
    bucket_episode_counts = [0 for _ in training_buckets]
    next_bucket_index = 0
    current_visit_progress = 0
    last_validated_rollout = -1
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
    last_metrics: dict[str, float] = {}
    resume_best_path: Path | None = None
    best_reward: float | None = None
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
            policy_update=policy_update,
            collection_contract=collection_contract,
            bucket_count=len(training_buckets),
            bucket_target_counts=bucket_target_counts,
            rollout_groups_per_bucket_visit=(
                config.rollout_groups_per_bucket_visit
            ),
            optimizer_step=trainer.optimizer_step,
        )
        environment_steps = int(resume_payload["environment_steps"])
        sampled_rollouts = int(restored_sampler_state["sampled_rollouts"])
        uninformative_rollouts = int(
            restored_sampler_state["uninformative_rollouts"]
        )
        bucket_sample_counts = list(
            restored_sampler_state["bucket_sample_counts"]
        )
        bucket_optimizer_step_counts = list(
            restored_sampler_state["bucket_optimizer_step_counts"]
        )
        bucket_episode_counts = list(
            restored_sampler_state["bucket_episode_counts"]
        )
        next_bucket_index = int(restored_sampler_state["next_bucket_index"])
        current_visit_progress = int(
            restored_sampler_state["current_visit_progress"]
        )
        last_validated_rollout = int(
            restored_sampler_state["last_validated_rollout"]
        )
        generator.set_state(restored_sampler_state["generator_state"])
        last_metrics = {
            str(name): float(value)
            for name, value in resume_payload["metrics"].items()
        }
        resume_best_path, best_reward = _resume_best_checkpoint_anchor(
            Path(config.resume_checkpoint), resume_payload
        )
        if sampled_rollouts >= target_rollout_groups:
            raise OnlineGRPOError(
                "resume checkpoint already reached requested rollout groups"
            )

    frozen_pretrain_reward_logging = (
        _frozen_pretrain_reward_logging_metadata(trainer.planner)
    )
    run_start_optimizer_step = trainer.optimizer_step
    run_start_rollout_group = sampled_rollouts
    online_config = dataclasses.asdict(config)
    online_config["resume_checkpoint"] = (
        str(config.resume_checkpoint)
        if config.resume_checkpoint is not None
        else None
    )
    frozen = {
        "format": "bev_joint_grpo_online_config_v7",
        "variant": variant,
        "run_mode": run_mode,
        "diagnostic_only": run_mode != "formal",
        "eligible_for_formal_training": run_mode == "formal",
        "online_config": online_config,
        "grpo_config": _grpo_config_artifact_payload(grpo_config),
        "source_stage1_sha256": source_sha,
        "policy_update_contract": joint_grpo_optimizer_contract(policy_update),
        "policy_update_contract_sha256": (
            joint_grpo_optimizer_contract_sha256(policy_update)
        ),
        "transition_noise_contract": joint_grpo_transition_noise_contract(),
        "transition_noise_contract_sha256": (
            joint_grpo_transition_noise_contract_sha256()
        ),
        "rollout_collection_contract": collection_contract,
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
        "frozen_pretrain_reward_logging": frozen_pretrain_reward_logging,
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
    consecutive_empty_episodes = 0
    best_path = run_dir / "checkpoints" / "best.pt"
    last_path = run_dir / "checkpoints" / "last.pt"
    best_checkpoint_sha256: str | None = None
    if resume_best_path is not None:
        shutil.copyfile(resume_best_path, best_path)
        best_checkpoint_sha256 = _checkpoint_file_sha256(best_path)
    simulator_diagnostic_errors: list[dict[str, object]] = [
        {"optimizer_step": 0, "stage1_baseline": True, **dict(value)}
        for value in pretrain_simulator_errors
    ]

    _reset_cuda_peak_memory(torch_device)
    try:
        while sampled_rollouts < target_rollout_groups:
            bucket_index = next_bucket_index
            if (
                bucket_sample_counts[bucket_index]
                >= bucket_target_counts[bucket_index]
            ):
                raise OnlineGRPOError(
                    "training bucket cursor points to a completed target"
                )
            scenario, seed = training_buckets[bucket_index]
            samples_at_episode_start = sampled_rollouts
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
                    sampled_rollouts < target_rollout_groups
                    and bucket_sample_counts[bucket_index]
                    < bucket_target_counts[bucket_index]
                    and current_visit_progress
                    < config.rollout_groups_per_bucket_visit
                    and episode_step < config.environment_steps_per_episode
                ):
                    builder.capture_state(env, episode_step * dt_s)
                    history_ready = builder.history_ready()
                    if history_ready:
                        scenario_summary = _scenario_summary(env)
                        ready_now = _scenario_ready_from_summary(
                            scenario_summary
                        )
                        window_closed_now = (
                            _scenario_sampling_window_closed_from_summary(
                                scenario_summary
                            )
                        )
                    else:
                        ready_now = False
                        window_closed_now = False
                    scenario_window_closed = bool(
                        scenario_window_closed
                        or window_closed_now
                    )
                    if scenario_window_closed:
                        if start_target_step is not None and not start_accepted:
                            assert start_ready_step is not None
                            assert start_offset is not None
                            assert start_upper_bound is not None
                            last_open_offset = (
                                episode_step - start_ready_step - 1
                            )
                            if last_open_offset >= 0:
                                temporary_start_offset_upper_bound = min(
                                    last_open_offset,
                                    temporary_start_offset_upper_bound
                                    if temporary_start_offset_upper_bound
                                    is not None
                                    else last_open_offset,
                                )
                                retryable_start_rejection = True
                            rollout_start_rejected_count += 1
                            _write_rollout_start_event(
                                metrics_path,
                                optimizer_step=trainer.optimizer_step,
                                rollout_group=sampled_rollouts,
                                bucket_index=bucket_index,
                                scenario=scenario,
                                seed=seed,
                                bucket_episode=(
                                    bucket_episode_counts[bucket_index]
                                ),
                                ready_step=start_ready_step,
                                upper_bound=start_upper_bound,
                                sampled_offset=start_offset,
                                target_step=start_target_step,
                                status=(
                                    "rejected_window_closed_before_target"
                                ),
                            )
                        break
                    if start_ready_step is None and ready_now:
                        start_ready_step = episode_step
                        start_offset, start_upper_bound = (
                            _sample_rollout_start_offset(
                                config,
                                start_ready_step,
                                generator=generator,
                                temporary_max_offset_steps=(
                                    temporary_start_offset_upper_bound
                                ),
                            )
                        )
                        start_target_step = start_ready_step + start_offset
                        rollout_start_offsets_this_run.append(start_offset)
                    if (
                        start_target_step is None
                        or episode_step < start_target_step
                    ):
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
                                rollout_group=sampled_rollouts + 1,
                                bucket_index=bucket_index,
                                scenario=scenario,
                                seed=seed,
                                bucket_episode=(
                                    bucket_episode_counts[bucket_index]
                                ),
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
                        committed_execution_id = (
                            condition.committed_execution_id
                        )
                        committed_plan_actions = (
                            condition.committed_plan_actions
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
                        frozen_pretrain = trainer.infer_frozen_pretrain(rollout)
                        frozen_raw_candidate = (
                            frozen_pretrain["selected_trajectory"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                        frozen_proxy = proxy_backend.score(
                            env, values, frozen_raw_candidate
                        )
                        frozen_rewards = np.asarray(frozen_proxy.rewards)
                        if frozen_rewards.shape != (1,):
                            raise OnlineGRPOError(
                                "frozen Stage1 inference must produce exactly "
                                "one joint raw trajectory reward"
                            )
                        frozen_pretrain_reward = float(frozen_rewards[0])
                        if not math.isfinite(frozen_pretrain_reward):
                            raise OnlineGRPOError(
                                "frozen Stage1 raw proxy reward must be finite"
                            )
                        (
                            optimization,
                            rule_event,
                            committed_execution_id,
                            committed_plan_actions,
                        ) = _finalize_online_rule_action(
                            rule_maker,
                            condition,
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
                        current_visit_progress += 1
                        informative = _joint_rewards_are_informative(
                            proxy.rewards,
                            group_size=trainer.config.group_size,
                        )
                        rewards = torch.from_numpy(
                            proxy.rewards.reshape(1, -1)
                        ).to(torch_device)
                        if informative:
                            update = trainer.update(
                                rollout,
                                rewards,
                                policy_update=policy_update,
                            )
                            if len(update.epoch_results) != config.update_epochs:
                                raise OnlineGRPOError(
                                    "joint GRPO update did not complete every "
                                    "configured epoch"
                                )
                            bucket_optimizer_step_counts[bucket_index] += len(
                                update.epoch_results
                            )
                            rollout_advantages = update.loss.advantages
                            for epoch in update.epoch_results:
                                _, optimizer_metrics = (
                                    _split_loss_metrics_by_step_axis(
                                        epoch.loss.scalar_metrics()
                                    )
                                )
                                optimizer_metrics.update(
                                    {
                                        "optimizer_step": float(
                                            epoch.optimizer_step
                                        ),
                                        "rollout_group": float(sampled_rollouts),
                                        "epoch_in_rollout": float(
                                            epoch.epoch_in_rollout
                                        ),
                                        "gradient_total": float(
                                            epoch.total_gradient_norm
                                        ),
                                    }
                                )
                                for name, value in epoch.gradient_norms.items():
                                    optimizer_metrics[f"gradient/{name}"] = float(
                                        value
                                    )
                                for metric_name, metric_value in (
                                    optimizer_metrics.items()
                                ):
                                    writer.add_scalar(
                                        metric_name,
                                        metric_value,
                                        epoch.optimizer_step,
                                    )
                                with metrics_path.open(
                                    "a", encoding="utf-8"
                                ) as stream:
                                    stream.write(
                                        json.dumps(
                                            {
                                                "event": "optimizer_epoch",
                                                **optimizer_metrics,
                                            },
                                            sort_keys=True,
                                        )
                                        + "\n"
                                    )
                                last_metrics.update(optimizer_metrics)
                        else:
                            uninformative_rollouts += 1
                            rollout_advantages = normalize_signed_advantages(
                                rewards,
                                group_size=trainer.config.group_size,
                                eps=trainer.config.advantage_eps,
                            ).detach()
                        if _write_advantage_vector_summary(
                            writer,
                            rollout_advantages,
                            sampled_rollouts,
                            target_rollout_groups,
                            config.advantage_vector_log_interval_rollouts,
                            group_size=trainer.config.group_size,
                        ):
                            advantage_vector_record_count += 1
                        rollout_advantage_metrics = _advantage_scalar_metrics(
                            rollout_advantages
                        )
                        rollout_metrics = {
                            "environment_steps": float(environment_steps),
                            "sampled_rollouts": float(sampled_rollouts),
                            "uninformative_rollouts": float(
                                uninformative_rollouts
                            ),
                            "training_bucket_index": float(bucket_index),
                            "raw_proxy_reward_mean": float(
                                proxy.rewards.mean()
                            ),
                            "raw_proxy_reward_max": float(proxy.rewards.max()),
                            FROZEN_PRETRAIN_RAW_PROXY_REWARD_TAG: (
                                frozen_pretrain_reward
                            ),
                            "raw_proxy_unsafe_rate": float(proxy.unsafe.mean()),
                            "selected_raw_candidate_index": float(
                                selected_index
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
                            **rollout_advantage_metrics,
                        }
                        for metric_name, metric_value in rollout_metrics.items():
                            writer.add_scalar(
                                metric_name,
                                metric_value,
                                sampled_rollouts,
                            )
                        with metrics_path.open("a", encoding="utf-8") as stream:
                            stream.write(
                                json.dumps(
                                    {"event": "rollout", **rollout_metrics},
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                        last_metrics.update(rollout_metrics)
                        action = joint_trajectory_action(
                            optimization.optimized_trajectories[0]
                        )

                    stepped_from = episode_step
                    _, _, terminated, truncated, info = env.step(action)
                    environment_steps += 1
                    episode_step += 1
                    if episode_has_ended(terminated, truncated, info):
                        if start_target_step is not None and not start_accepted:
                            assert start_ready_step is not None
                            assert start_offset is not None
                            assert start_upper_bound is not None
                            last_open_offset = stepped_from - start_ready_step
                            if last_open_offset >= 0:
                                temporary_start_offset_upper_bound = min(
                                    last_open_offset,
                                    temporary_start_offset_upper_bound
                                    if temporary_start_offset_upper_bound
                                    is not None
                                    else last_open_offset,
                                )
                                retryable_start_rejection = True
                            rollout_start_rejected_count += 1
                            _write_rollout_start_event(
                                metrics_path,
                                optimizer_step=trainer.optimizer_step,
                                rollout_group=sampled_rollouts,
                                bucket_index=bucket_index,
                                scenario=scenario,
                                seed=seed,
                                bucket_episode=(
                                    bucket_episode_counts[bucket_index]
                                ),
                                ready_step=start_ready_step,
                                upper_bound=start_upper_bound,
                                sampled_offset=start_offset,
                                target_step=start_target_step,
                                status=(
                                    "rejected_episode_ended_before_target"
                                ),
                            )
                        break
                    if sampled_rollouts > samples_at_episode_start and (
                        _bucket_visit_is_complete(
                            bucket_index=bucket_index,
                            bucket_sample_counts=bucket_sample_counts,
                            bucket_target_counts=bucket_target_counts,
                            current_visit_progress=current_visit_progress,
                            rollout_groups_per_bucket_visit=(
                                config.rollout_groups_per_bucket_visit
                            ),
                        )
                        or sampled_rollouts == target_rollout_groups
                        or sampled_rollouts
                        % config.validation_interval_rollouts
                        == 0
                    ):
                        break
            finally:
                env.close()
            if sampled_rollouts == samples_at_episode_start:
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
                bucket_sample_counts=bucket_sample_counts,
                bucket_target_counts=bucket_target_counts,
                current_visit_progress=current_visit_progress,
                rollout_groups_per_bucket_visit=(
                    config.rollout_groups_per_bucket_visit
                ),
            ):
                current_visit_progress = 0
                unfinished_bucket = _next_unfinished_bucket_index(
                    bucket_sample_counts,
                    bucket_target_counts,
                    start_index=(bucket_index + 1) % len(training_buckets),
                )
                next_bucket_index = (
                    0 if unfinished_bucket is None else unfinished_bucket
                )
                temporary_start_offset_upper_bound = None
            else:
                next_bucket_index = bucket_index
            _reject_exhausted_uninformative_budget(
                sampled_rollouts=sampled_rollouts,
                target_rollout_groups=target_rollout_groups,
                optimizer_step=trainer.optimizer_step,
                run_start_optimizer_step=run_start_optimizer_step,
            )
            if sampled_rollouts != last_validated_rollout and (
                sampled_rollouts == target_rollout_groups
                or (
                    sampled_rollouts > 0
                    and sampled_rollouts
                    % config.validation_interval_rollouts
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
                        "rollout_group": int(sampled_rollouts),
                        "stage1_baseline": False,
                        **dict(value),
                    }
                    simulator_diagnostic_errors.append(error_record)
                    writer.add_text(
                        "validation/simulator_error",
                        json.dumps(error_record, sort_keys=True),
                        sampled_rollouts,
                    )
                last_metrics.update(validation)
                for metric_name, metric_value in validation.items():
                    writer.add_scalar(
                        metric_name, metric_value, sampled_rollouts
                    )
                last_validated_rollout = sampled_rollouts
                if trainer.optimizer_step == run_start_optimizer_step:
                    continue
                current_sampler_state = _sampler_state(
                    sampled_rollouts=sampled_rollouts,
                    uninformative_rollouts=uninformative_rollouts,
                    bucket_target_counts=bucket_target_counts,
                    bucket_sample_counts=bucket_sample_counts,
                    bucket_optimizer_step_counts=(
                        bucket_optimizer_step_counts
                    ),
                    bucket_episode_counts=bucket_episode_counts,
                    next_bucket_index=next_bucket_index,
                    current_visit_progress=current_visit_progress,
                    generator_state=generator.get_state(),
                    last_validated_rollout=last_validated_rollout,
                    rollout_groups_per_bucket_visit=(
                        config.rollout_groups_per_bucket_visit
                    ),
                    update_epochs=config.update_epochs,
                    optimizer_step=trainer.optimizer_step,
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
                        policy_update=policy_update,
                        collection_contract=collection_contract,
                        sampler_state=current_sampler_state,
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
                    policy_update=policy_update,
                    collection_contract=collection_contract,
                    sampler_state=current_sampler_state,
                )
                save_grpo_checkpoint(last_path, checkpoint)
    finally:
        writer.close()

    if best_reward is None or best_checkpoint_sha256 is None:
        raise OnlineGRPOError("online GRPO never completed fixed validation")
    final_sampler_state = _sampler_state(
        sampled_rollouts=sampled_rollouts,
        uninformative_rollouts=uninformative_rollouts,
        bucket_target_counts=bucket_target_counts,
        bucket_sample_counts=bucket_sample_counts,
        bucket_optimizer_step_counts=bucket_optimizer_step_counts,
        bucket_episode_counts=bucket_episode_counts,
        next_bucket_index=next_bucket_index,
        current_visit_progress=current_visit_progress,
        generator_state=generator.get_state(),
        last_validated_rollout=last_validated_rollout,
        rollout_groups_per_bucket_visit=(
            config.rollout_groups_per_bucket_visit
        ),
        update_epochs=config.update_epochs,
        optimizer_step=trainer.optimizer_step,
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
        best_checkpoint_sha256=best_checkpoint_sha256,
        policy_update=policy_update,
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
        policy_update=policy_update,
        collection_contract=collection_contract,
        bucket_count=len(training_buckets),
        bucket_target_counts=bucket_target_counts,
        rollout_groups_per_bucket_visit=(
            config.rollout_groups_per_bucket_visit
        ),
        optimizer_step=restored.optimizer_step,
    )

    plot_paths = generate_grpo_plots(
        run_dir / "tb",
        run_dir / "plots",
        require_frozen_pretrain_reward=True,
    )
    if (
        rollout_start_accepted_count + rollout_start_rejected_count
        != len(rollout_start_offsets_this_run)
    ):
        raise OnlineGRPOError(
            "rollout start diagnostics contain an unresolved sampling attempt"
        )

    report = {
        "format": "bev_joint_grpo_online_report_v7",
        "variant": variant,
        "run_mode": run_mode,
        "diagnostic_only": run_mode != "formal",
        "eligible_for_formal_training": run_mode == "formal",
        "grpo_config": _grpo_config_artifact_payload(grpo_config),
        "optimizer_steps": trainer.optimizer_step,
        "optimizer_steps_this_run": (
            trainer.optimizer_step - run_start_optimizer_step
        ),
        "environment_steps": environment_steps,
        "environment_episode_count": sum(bucket_episode_counts),
        "warmup_environment_steps": environment_steps - sampled_rollouts,
        "sampled_rollouts": sampled_rollouts,
        "sampled_rollouts_this_run": (
            sampled_rollouts - run_start_rollout_group
        ),
        "target_rollout_groups": target_rollout_groups,
        "uninformative_rollouts": uninformative_rollouts,
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
        "cuda_peak_memory_bytes": cuda_peak_memory_bytes,
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
        "frozen_pretrain_reward_logging": frozen_pretrain_reward_logging,
        "best_checkpoint_metric": "validation/raw_proxy_reward_mean",
        "tracking_expansion_enabled": False,
        "calibration_required": False,
        "policy_update_contract": joint_grpo_optimizer_contract(policy_update),
        "policy_update_contract_sha256": (
            joint_grpo_optimizer_contract_sha256(policy_update)
        ),
        "transition_noise_contract": joint_grpo_transition_noise_contract(),
        "transition_noise_contract_sha256": (
            joint_grpo_transition_noise_contract_sha256()
        ),
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
                "target_rollouts": bucket_target_counts[index],
                "sampled_rollouts": bucket_sample_counts[index],
                "environment_episodes": bucket_episode_counts[index],
                "optimizer_steps": bucket_optimizer_step_counts[index],
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
            "interval_rollout_groups": (
                config.advantage_vector_log_interval_rollouts
            ),
            "record_count": advantage_vector_record_count,
            "scope": "current_run_only",
            "run_start_optimizer_step": run_start_optimizer_step,
            "run_start_rollout_group": run_start_rollout_group,
            "group_axis_semantics": (
                "independent_random_sample_slot_without_cross_step_identity"
            ),
            "heatmap": str(plot_paths["advantage_heatmap"].resolve()),
            "x_axis": "absolute_rollout_group",
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
            "policy_stability_curve": str(
                plot_paths["policy_stability_curve"].resolve()
            ),
            "reward_x_axis": "absolute_rollout_group",
            "validation_reward_x_axis": "absolute_rollout_group",
            "optimizer_x_axis": "absolute_optimizer_step",
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
        raise OnlineGRPOError(
            "source_checkpoint must be a non-empty path string"
        )
    online = payload.get("online")
    if not isinstance(online, Mapping):
        raise OnlineGRPOError("online GRPO YAML requires an online mapping")
    legacy_fields = {
        "total_optimizer_steps",
        "validation_interval_steps",
        "advantage_vector_log_interval_steps",
    }
    present_legacy = sorted(legacy_fields.intersection(online))
    if present_legacy:
        raise OnlineGRPOError(
            "online GRPO YAML contains legacy optimizer-step fields: "
            + ", ".join(present_legacy)
        )
    scenarios = tuple(
        (str(value["scenario"]), str(value["route"]))
        for value in online.get("scenarios", ())
        if isinstance(value, Mapping)
    )
    online_config = JointGRPOOnlineConfig(
        device=str(online.get("device", "cuda")),
        seed=online.get("seed", 17),
        group_size=online.get("group_size", 24),
        total_rollout_groups=online.get("total_rollout_groups", 100),
        update_epochs=online.get("update_epochs", 4),
        clip_epsilon=online.get("clip_epsilon", 0.2),
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
        rollout_groups_per_bucket_visit=online.get(
            "rollout_groups_per_bucket_visit", 10
        ),
        rollout_start_offset_max_steps=online.get(
            "rollout_start_offset_max_steps", 200
        ),
        rollout_start_min_remaining_steps=online.get(
            "rollout_start_min_remaining_steps", 10
        ),
        validation_interval_rollouts=online.get(
            "validation_interval_rollouts", 20
        ),
        advantage_vector_log_interval_rollouts=online.get(
            "advantage_vector_log_interval_rollouts", 20
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
    "joint_trajectory_action",
    "model_inputs_to_batch",
    "optimize_selected_model_trajectories",
    "run_joint_grpo_training",
]


if __name__ == "__main__":
    raise SystemExit(main())
