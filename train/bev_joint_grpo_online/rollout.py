"""GRPO sampling, reward-signal construction, and bucket scheduling."""
from __future__ import annotations
import dataclasses, hashlib, json, math, shutil, subprocess, time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Mapping, Sequence
import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from evaluation.plot_grpo import ACCEPTED_ROLLOUT_AXIS_LABEL, ADVANTAGE_VECTOR_TAG, FROZEN_PRETRAIN_RAW_PROXY_REWARD_TAG, REWARD_CURVE_TAGS, VALIDATION_REWARD_CURVE_TAGS, generate_grpo_plots
from evaluation.joint_simulator_branch import JointEpisodeSpec, JointSimulatorBranchEvaluator, capture_joint_pose_global
from expert_dataset.collect_joint_bev import JointBEVSampleBuilder, SensorlessJointBEVPlatoonEnv, simulator_decision_dt_s
from models.bev_planner import DDIMNoiseBundle, DDIMTransitionError, DEFAULT_DDIM_PATH, GRPO_OPEN_REWARD_APPLICATION_CONTRACT, GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256, JointGRPOConfig, JointRewardConfig, JointTrajectoryProxyReward, KinematicTrajectoryOptimizer, KinematicTrajectoryOptimizerConfig, TrajectoryOptimizationError, TrajectoryOptimizationResult, joint_grpo_optimizer_contract, joint_grpo_optimizer_contract_sha256
from models.bev_planner.joint_reward import VEHICLE_MODE_REWARD_CONTRACT, VEHICLE_MODE_REWARD_CONTRACT_SHA256, VehicleModeCounterfactualReward, VehicleModePretrainRewardResult, VehicleModeRewardResult, VehicleModeRewardConfig, vehicle_mode_reward_config_sha256
from models.bev_planner.mode_contract import ModeIndex
from models.decisioner.rule_decisioner import LaneChangeCommitmentError, diffusion_mode_feedback_actions, hard_valid_modes_by_rule_action, joint_proposal_actions, make_rule_maker, match_joint_action_proposal
from train.bev_joint_grpo import grpo_b_checkpoint_payload, grpo_checkpoint_payload, load_grpo_b_checkpoint, load_grpo_checkpoint, load_stage1_a_for_grpo, load_stage1_b_for_grpo, save_grpo_checkpoint
from train.train_bev_diffusion_stage1 import planner_forward_from_batch
from scenarios.bev_round13_contract import DEVELOPMENT_SEEDS, HOLDOUT_SEEDS, PRIMARY_S5_S9_SCENARIOS, BEVScenarioContractError, deterministic_initial_speed_km_h, primary_scenario_contract, validate_primary_scenario_contract
from .config import OnlineGRPOError

def _load_trainer(variant: str, source_checkpoint: Path, device: torch.device, *, grpo_config: JointGRPOConfig, allow_diagnostic_source: bool):
    loader = load_stage1_a_for_grpo if variant == 'A' else load_stage1_b_for_grpo
    return loader(source_checkpoint, device=device, config=grpo_config, allow_diagnostic_source=allow_diagnostic_source)

def _fixed_scale_reward_signals(current_rewards: np.ndarray, frozen_rewards: np.ndarray, collision: np.ndarray, out_of_drivable: np.ndarray, valid_mode_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror the fixed-scale torch advantage contract for online gating/logging."""
    values = np.asarray(current_rewards)
    paired = np.asarray(frozen_rewards)
    collision_values = np.asarray(collision)
    out_values = np.asarray(out_of_drivable)
    valid = np.asarray(valid_mode_mask)
    if values.ndim != 3 or values.shape[:2] != (3, 10):
        raise OnlineGRPOError('vehicle-mode rewards must have shape [3,10,N]')
    if values.shape[2] < 2 or values.dtype not in (np.float32, np.float64):
        raise OnlineGRPOError('vehicle-mode rewards must be float [3,10,N>=2]')
    if paired.shape != values.shape or paired.dtype not in (np.float32, np.float64):
        raise OnlineGRPOError('paired frozen rewards must be float [3,10,N]')
    if collision_values.shape != values.shape or collision_values.dtype != np.bool_ or out_values.shape != values.shape or (out_values.dtype != np.bool_):
        raise OnlineGRPOError('paired safety masks must be bool [3,10,N]')
    if valid.shape != (3, 10) or valid.dtype != np.bool_:
        raise OnlineGRPOError('valid mode mask must be bool [3,10]')
    if not np.isfinite(values).all() or not np.isfinite(paired).all():
        raise OnlineGRPOError('paired vehicle-mode rewards must be finite')
    centered = values - values.mean(axis=-1, keepdims=True)
    unsafe = collision_values | out_values
    advantages = np.where(unsafe, -1.0, np.where(values >= paired - 1e-06, np.maximum(centered, 0.0), 0.0)).astype(np.float32, copy=False)
    advantages *= valid[..., None]
    signal = valid & np.any(advantages != 0.0, axis=-1)
    return (centered.astype(np.float32, copy=False), advantages, signal)

def _diffusion_attempt_generators(*, device: torch.device, training_seed: int, live_state_index: int, retry_index: int) -> tuple[torch.Generator, torch.Generator]:
    """Create independent reproducible initial/transition noise streams."""
    if min(live_state_index, retry_index) < 0:
        raise OnlineGRPOError('diffusion RNG indices must be non-negative')
    modulus = 2 ** 63 - 1
    state_key = (int(training_seed) + 1000003 * int(live_state_index) + 10007 * int(retry_index)) % modulus
    initial = torch.Generator(device=device)
    transition = torch.Generator(device=device)
    initial.manual_seed((state_key + 2000033) % modulus)
    transition.manual_seed((state_key + 4000037) % modulus)
    return (initial, transition)

def _attempt_budget_is_exhausted(*, accepted_update_states: int, target_accepted_update_states: int, sampling_attempts: int, max_sampling_attempts: int) -> bool:
    """Whether collection hit its hard attempt cap before its accepted target."""
    for name, value in (('accepted_update_states', accepted_update_states), ('target_accepted_update_states', target_accepted_update_states), ('sampling_attempts', sampling_attempts), ('max_sampling_attempts', max_sampling_attempts)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise OnlineGRPOError(f'{name} must be a non-negative integer')
    if target_accepted_update_states <= 0:
        raise OnlineGRPOError('target_accepted_update_states must be a positive integer')
    if max_sampling_attempts <= 0:
        raise OnlineGRPOError('max_sampling_attempts must be a positive integer')
    if accepted_update_states > target_accepted_update_states:
        raise OnlineGRPOError('accepted rollout target was exceeded')
    if sampling_attempts < accepted_update_states:
        raise OnlineGRPOError('attempted rollout count is below accepted count')
    return accepted_update_states < target_accepted_update_states and sampling_attempts >= max_sampling_attempts

def _round_robin_training_buckets(scenarios: Sequence[tuple[str, str]], seeds: Sequence[int]) -> tuple[tuple[tuple[str, str], int], ...]:
    buckets = tuple((((str(scenario[0]), str(scenario[1])), int(seed)) for scenario in scenarios for seed in seeds))
    if not buckets:
        raise OnlineGRPOError('online GRPO training schedule is empty')
    return buckets

def _balanced_bucket_targets(total_rollout_groups: int, bucket_count: int) -> list[int]:
    for name, value in (('total_rollout_groups', total_rollout_groups), ('bucket_count', bucket_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise OnlineGRPOError(f'{name} must be a positive integer')
    base, remainder = divmod(total_rollout_groups, bucket_count)
    return [base + int(bucket_index < remainder) for bucket_index in range(bucket_count)]

def _next_unfinished_bucket_index(bucket_sample_counts: Sequence[int], bucket_target_counts: Sequence[int], *, start_index: int) -> int | None:
    if len(bucket_sample_counts) != len(bucket_target_counts) or not bucket_sample_counts:
        raise OnlineGRPOError('training bucket counters are invalid')
    bucket_count = len(bucket_sample_counts)
    if isinstance(start_index, bool) or not isinstance(start_index, int) or (not 0 <= start_index < bucket_count):
        raise OnlineGRPOError('training bucket start cursor is invalid')
    for offset in range(bucket_count):
        index = (start_index + offset) % bucket_count
        if bucket_sample_counts[index] < bucket_target_counts[index]:
            return index
    return None

def _bucket_visit_is_complete(*, bucket_index: int, bucket_sample_counts: Sequence[int], bucket_target_counts: Sequence[int], current_visit_progress: int, rollout_groups_per_bucket_visit: int) -> bool:
    return current_visit_progress >= rollout_groups_per_bucket_visit or bucket_sample_counts[bucket_index] >= bucket_target_counts[bucket_index]
