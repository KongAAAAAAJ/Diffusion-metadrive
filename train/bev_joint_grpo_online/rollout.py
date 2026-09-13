"""GRPO trainer loading, reward gating, and rollout scheduling helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from models.bev_planner import JointGRPOConfig
from train.bev_joint_grpo import load_stage1_a_for_grpo, load_stage1_b_for_grpo

from .config import OnlineGRPOError

def _load_trainer(variant: str, source_checkpoint: Path, device: torch.device, *, grpo_config: JointGRPOConfig, allow_diagnostic_source: bool):
    loader = load_stage1_a_for_grpo if variant == 'A' else load_stage1_b_for_grpo
    return loader(source_checkpoint, device=device, config=grpo_config, allow_diagnostic_source=allow_diagnostic_source)

def _standard_grpo_reward_signals(
    current_rewards: np.ndarray,
    valid_mode_mask: np.ndarray,
    *,
    epsilon: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror the standard torch GRPO advantage contract for gating/logging."""
    values = np.asarray(current_rewards)
    valid = np.asarray(valid_mode_mask)
    if values.ndim != 3 or values.shape[:2] != (3, 10):
        raise OnlineGRPOError('vehicle-mode rewards must have shape [3,10,N]')
    if values.shape[2] < 2 or values.dtype not in (np.float32, np.float64):
        raise OnlineGRPOError('vehicle-mode rewards must be float [3,10,N>=2]')
    if valid.shape != (3, 10) or valid.dtype != np.bool_:
        raise OnlineGRPOError('valid mode mask must be bool [3,10]')
    if not np.isfinite(values).all():
        raise OnlineGRPOError('vehicle-mode rewards must be finite')
    eps = float(epsilon)
    if not np.isfinite(eps) or eps <= 0.0:
        raise OnlineGRPOError('epsilon must be positive and finite')
    centered = values - values.mean(axis=-1, keepdims=True)
    population_std = np.sqrt(np.mean(np.square(centered), axis=-1, keepdims=True))
    advantages = (centered / (population_std + eps)).astype(np.float32, copy=False)
    advantages *= valid[..., None]
    signal = valid & np.any(advantages != 0.0, axis=-1)
    return centered.astype(np.float32, copy=False), advantages, signal


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
