"""TensorBoard and JSONL logging helpers."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from evaluation.plot_grpo import ADVANTAGE_VECTOR_TAG
from models.bev_planner.vehicle_mode_reward import VehicleModeRewardConfig, VehicleModeRewardResult

from .config import OnlineGRPOError
from .validation import _validation_is_safety_eligible

def _should_record_advantage_vector(rollout_group: int, target_rollout_groups: int, interval_rollouts: int) -> bool:
    for name, value in (('rollout_group', rollout_group), ('target_rollout_groups', target_rollout_groups), ('interval_rollouts', interval_rollouts)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise OnlineGRPOError(f'{name} must be a positive integer')
    if rollout_group > target_rollout_groups:
        raise OnlineGRPOError('rollout_group cannot exceed target_rollout_groups')
    return rollout_group % interval_rollouts == 0 or rollout_group == target_rollout_groups

def _write_advantage_vector_summary(writer: SummaryWriter, advantages: torch.Tensor, rollout_group: int, target_rollout_groups: int, interval_rollouts: int, *, trajectories_per_mode: int) -> bool:
    if isinstance(trajectories_per_mode, bool) or not isinstance(trajectories_per_mode, int) or trajectories_per_mode < 2:
        raise OnlineGRPOError('trajectories_per_mode must be an integer greater than or equal to 2')
    if not _should_record_advantage_vector(rollout_group, target_rollout_groups, interval_rollouts):
        return False
    vector = advantages.detach().cpu()
    if vector.dtype != torch.float32 or tuple(vector.shape) != (1, 3, 10, trajectories_per_mode):
        raise OnlineGRPOError(f'advantage tensor must be float32 with shape [1,3,10,{trajectories_per_mode}]')
    if not bool(torch.isfinite(vector).all()):
        raise OnlineGRPOError('advantage vector must be finite')
    writer.add_tensor(ADVANTAGE_VECTOR_TAG, vector, rollout_group)
    return True

def _split_loss_metrics_by_step_axis(metrics: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    rollout_metrics = {name: float(value) for name, value in metrics.items() if name.startswith('advantage/')}
    optimizer_metrics = {name: float(value) for name, value in metrics.items() if not name.startswith('advantage/')}
    return (rollout_metrics, optimizer_metrics)

def _advantage_scalar_metrics(advantages: torch.Tensor) -> dict[str, float]:
    return {'advantage/mean': float(advantages.mean().detach().cpu()), 'advantage/std': float(advantages.std(unbiased=False).detach().cpu()), 'advantage/min': float(advantages.min().detach().cpu()), 'advantage/max': float(advantages.max().detach().cpu())}

def _write_rollout_start_event(path: Path, *, optimizer_step: int, rollout_group: int, bucket_index: int, scenario: tuple[str, str], seed: int, bucket_episode: int, ready_step: int, upper_bound: int, sampled_offset: int, target_step: int, status: str) -> None:
    record = {'event': 'rollout_start', 'optimizer_step': int(optimizer_step), 'rollout_group': int(rollout_group), 'training_bucket_index': int(bucket_index), 'scenario': str(scenario[0]), 'route': str(scenario[1]), 'seed': int(seed), 'bucket_episode': int(bucket_episode), 't_ready': int(ready_step), 'feasible_upper_bound': int(upper_bound), 'sampled_offset': int(sampled_offset), 'target_step': int(target_step), 'status': str(status)}
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')

def _write_dynamic_sampling_attempt_event(path: Path, *, optimizer_step: int, accepted_update_state: int, sampling_attempt: int, retry_index: int, bucket_index: int, scenario: tuple[str, str], seed: int, reward_result: VehicleModeRewardResult, paired_frozen_rewards: np.ndarray, centered_rewards: np.ndarray, advantages: np.ndarray, hard_valid_mode_mask: np.ndarray, signal_mode_mask: np.ndarray, reward_config: VehicleModeRewardConfig) -> dict[str, float]:
    if not isinstance(reward_result, VehicleModeRewardResult):
        raise OnlineGRPOError('sampling diagnostics require vehicle-mode rewards')
    values = np.asarray(reward_result.rewards, dtype=np.float64)
    baseline = np.asarray(reward_result.pretrain_rewards, dtype=np.float64)
    valid = np.asarray(reward_result.valid_mode_mask, dtype=np.bool_)
    hard_valid = np.asarray(hard_valid_mode_mask, dtype=np.bool_)
    paired = np.asarray(paired_frozen_rewards, dtype=np.float64)
    centered_all = np.asarray(centered_rewards, dtype=np.float64)
    advantages_all = np.asarray(advantages, dtype=np.float64)
    active = np.asarray(signal_mode_mask, dtype=np.bool_)
    if values.ndim != 3 or values.shape[:2] != (3, 10):
        raise OnlineGRPOError('sampling rewards must have shape [3,10,N]')
    if baseline.shape != (3, 10) or paired.shape != values.shape or centered_all.shape != values.shape or (advantages_all.shape != values.shape) or (valid.shape != (3, 10)) or (hard_valid.shape != (3, 10)) or (active.shape != (3, 10)):
        raise OnlineGRPOError('sampling pretrain rewards and active mask must be [3,10]')
    if not valid.any() or np.any(valid & ~hard_valid) or np.any(active & ~valid):
        raise OnlineGRPOError('sampling masks contain no valid modes or conflict')
    weights = {'progress_score': float(reward_config.progress_weight), 'gap_penalty': -float(reward_config.gap_weight), 'ttc_penalty': -float(reward_config.ttc_weight), 'road_penalty': -float(reward_config.road_weight), 'comfort_penalty': -float(reward_config.comfort_weight)}
    reconstructed = sum((coefficient * np.asarray(reward_result.components[name], dtype=np.float64) for name, coefficient in weights.items()))
    reconstructed -= float(reward_config.collision_penalty) * np.asarray(reward_result.collision, dtype=np.float64)
    reconstructed -= float(reward_config.out_of_drivable_penalty) * np.asarray(reward_result.out_of_drivable, dtype=np.float64)
    if not np.allclose(reconstructed[valid], values[valid], atol=2e-05, rtol=1e-05):
        raise OnlineGRPOError('reward components do not reconstruct total reward')
    group_records: list[dict[str, object]] = []
    group_mean_gains = np.zeros((3, 10), dtype=np.float64)
    group_median_gains = np.zeros((3, 10), dtype=np.float64)
    group_fractions = np.zeros((3, 10), dtype=np.float64)
    positive_below_fractions = np.zeros((3, 10), dtype=np.float64)
    for role in range(3):
        for mode in range(10):
            if not valid[role, mode]:
                group_records.append({'vehicle': role, 'mode': mode, 'valid': False, 'active': False, 'inactive_reason': 'optimizer_inexecutable' if hard_valid[role, mode] else 'hard_invalid'})
                continue
            group = values[role, mode]
            reference = float(baseline[role, mode])
            paired_group = paired[role, mode]
            centered = centered_all[role, mode]
            advantage = advantages_all[role, mode]
            mean_gain = float(np.mean(group - paired_group))
            median_gain = float(np.median(group - paired_group))
            fraction_ge = float(np.mean(group >= paired_group))
            positive_below = float(np.mean((advantage > 0.0) & (group < paired_group - 1e-06)))
            group_mean_gains[role, mode] = mean_gain
            group_median_gains[role, mode] = median_gain
            group_fractions[role, mode] = fraction_ge
            positive_below_fractions[role, mode] = positive_below
            component_record = {name: {'raw_mean': float(np.asarray(reward_result.components[name])[role, mode].mean()), 'weighted_mean': float(coefficient * np.asarray(reward_result.components[name])[role, mode].mean())} for name, coefficient in weights.items()}
            component_record.update({'collision': {'raw_mean': float(np.asarray(reward_result.collision)[role, mode].mean()), 'weighted_mean': float(-reward_config.collision_penalty * np.asarray(reward_result.collision)[role, mode].mean())}, 'out_of_drivable': {'raw_mean': float(np.asarray(reward_result.out_of_drivable)[role, mode].mean()), 'weighted_mean': float(-reward_config.out_of_drivable_penalty * np.asarray(reward_result.out_of_drivable)[role, mode].mean())}})
            group_records.append({'vehicle': role, 'mode': mode, 'valid': True, 'active': bool(active[role, mode]), 'inactive_reason': None if active[role, mode] else 'no_nonzero_sample_advantage', 'pretrain_reward': reference, 'paired_frozen_reward_mean': float(paired_group.mean()), 'reward_mean': float(group.mean()), 'reward_std': float(group.std()), 'reward_min': float(group.min()), 'reward_max': float(group.max()), 'reward_p05': float(np.quantile(group, 0.05)), 'reward_p50': float(np.quantile(group, 0.5)), 'reward_p95': float(np.quantile(group, 0.95)), 'mean_gain': mean_gain, 'median_gain': median_gain, 'max_gain': float(np.max(group - paired_group)), 'fraction_reward_ge_pretrain': fraction_ge, 'positive_advantage_below_pretrain_fraction': positive_below, 'advantage_min': float(advantage.min()), 'advantage_max': float(advantage.max()), 'advantage_rms': float(np.sqrt(np.mean(advantage ** 2))), 'collision_rate': float(np.asarray(reward_result.collision)[role, mode].mean()), 'out_of_drivable_rate': float(np.asarray(reward_result.out_of_drivable)[role, mode].mean()), 'clearance_violation_rate': float(np.asarray(reward_result.clearance_violation)[role, mode].mean()), 'components': component_record, 'minimum_background_gap_m': float(np.asarray(reward_result.components['minimum_background_gap_m'])[role, mode].min()), 'minimum_teammate_gap_m': float(np.asarray(reward_result.components['minimum_teammate_gap_m'])[role, mode].min()), 'minimum_road_margin_m': float(np.asarray(reward_result.components['minimum_road_margin_m'])[role, mode].min()), 'minimum_ttc_s': float(np.asarray(reward_result.components['minimum_ttc_s'])[role, mode].min())})

    def hierarchical_mean(group_values: np.ndarray, group_mask: np.ndarray) -> float:
        role_means = [float(group_values[role][group_mask[role]].mean()) if bool(group_mask[role].any()) else 0.0 for role in range(3)]
        return float(np.mean(role_means))
    valid_values = values[valid]
    paired_valid = paired[valid]
    deltas = valid_values - paired_valid
    reward_group_means = values.mean(axis=-1)
    reward_group_gains = (values - paired).mean(axis=-1)
    metrics = {'train/valid_all/vehicle_reward_mean': hierarchical_mean(reward_group_means, valid), 'train/valid_all/same_mode_pretrain_reward_mean': hierarchical_mean(baseline, valid), 'train/valid_all/reward_gain_mean': hierarchical_mean(reward_group_gains, valid), 'train/valid_all/reward_gain_max': float(deltas.max()), 'train/valid_all/fraction_reward_ge_pretrain': float(np.mean(valid_values >= paired_valid)), 'train/active_only/group_mean_gain': hierarchical_mean(group_mean_gains, active), 'train/active_only/group_median_gain': hierarchical_mean(group_median_gains, active), 'train/active_only/fraction_reward_ge_pretrain': hierarchical_mean(group_fractions, active), 'train/active_only/positive_advantage_below_pretrain_fraction': hierarchical_mean(positive_below_fractions, active), 'signal_mode_count': float(active.sum()), 'no_signal_mode_count': float((valid & ~active).sum()), 'invalid_or_unexecutable_mode_count': float((~valid).sum()), 'signal_vehicle_count': float(np.any(active, axis=1).sum()), 'train/valid_all/paired_reward_gain_mean': float(deltas.mean()), 'train/valid_all/paired_reward_gain_median': float(np.median(deltas)), 'train/valid_all/paired_reward_gain_p05': float(np.quantile(deltas, 0.05)), 'train/valid_all/paired_reward_gain_p95': float(np.quantile(deltas, 0.95)), 'train/valid_all/positive_fraction': float(np.mean(advantages_all[valid] > 0.0)), 'train/valid_all/baseline_filtered_fraction': float(np.mean((centered_all[valid] > 0.0) & (values[valid] < paired_valid - 1e-06))), 'train/valid_all/collision_negative_fraction': float(np.mean(np.asarray(reward_result.collision)[valid])), 'train/valid_all/out_negative_fraction': float(np.mean(np.asarray(reward_result.out_of_drivable)[valid]))}
    active_samples = np.broadcast_to(active[..., None], values.shape)
    for name, coefficient in weights.items():
        component = np.asarray(reward_result.components[name], dtype=np.float64)
        component_group_means = component.mean(axis=-1)
        metrics[f'train/valid_all/component/{name}_raw'] = hierarchical_mean(component_group_means, valid)
        metrics[f'train/valid_all/component/{name}_weighted'] = coefficient * hierarchical_mean(component_group_means, valid)
        if active.any():
            metrics[f'train/active_only/component/{name}_raw'] = hierarchical_mean(component_group_means, active)
            metrics[f'train/active_only/component/{name}_weighted'] = coefficient * hierarchical_mean(component_group_means, active)
    for name, indicator, coefficient in (('collision', reward_result.collision, -reward_config.collision_penalty), ('out_of_drivable', reward_result.out_of_drivable, -reward_config.out_of_drivable_penalty)):
        component = np.asarray(indicator, dtype=np.float64)
        component_group_means = component.mean(axis=-1)
        metrics[f'train/valid_all/component/{name}_raw'] = hierarchical_mean(component_group_means, valid)
        metrics[f'train/valid_all/component/{name}_weighted'] = coefficient * hierarchical_mean(component_group_means, valid)
        if active.any():
            metrics[f'train/active_only/component/{name}_raw'] = hierarchical_mean(component_group_means, active)
            metrics[f'train/active_only/component/{name}_weighted'] = coefficient * hierarchical_mean(component_group_means, active)
    road_margin = np.asarray(reward_result.components['minimum_road_margin_m'], dtype=np.float64)
    metrics['train/valid_all/minimum_road_margin_m'] = float(road_margin[valid].min())
    if active.any():
        metrics['train/active_only/minimum_road_margin_m'] = float(road_margin[active_samples].min())
    record = {'event': 'dynamic_sampling_attempt', 'optimizer_step': int(optimizer_step), 'accepted_update_state': int(accepted_update_state), 'sampling_attempt': int(sampling_attempt), 'retry_index': int(retry_index), 'training_bucket_index': int(bucket_index), 'scenario': str(scenario[0]), 'route': str(scenario[1]), 'seed': int(seed), 'accepted': bool(active.any()), 'rejection_reason': None if active.any() else 'all_vehicle_modes_no_signal', 'vehicle_mode_groups': group_records, **metrics}
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')
    return metrics

def _write_baseline_execution_event(path: Path, *, optimizer_step: int, accepted_update_states: int, rejected_sampling_attempts: int, stability_guard_rejections: int, exhausted_states: int, baseline_execution_steps: int, environment_steps: int, bucket_index: int, scenario: tuple[str, str], seed: int, pretrain_reward_mean: float, performance: Mapping[str, float]) -> None:
    record = {'event': 'frozen_baseline_execution', 'optimizer_step': int(optimizer_step), 'accepted_update_states': int(accepted_update_states), 'rejected_sampling_attempts': int(rejected_sampling_attempts), 'stability_guard_rejections': int(stability_guard_rejections), 'exhausted_states': int(exhausted_states), 'baseline_execution_steps': int(baseline_execution_steps), 'environment_steps': int(environment_steps), 'training_bucket_index': int(bucket_index), 'scenario': str(scenario[0]), 'route': str(scenario[1]), 'seed': int(seed), 'same_mode_pretrain_reward_mean': float(pretrain_reward_mean), **{str(name): float(value) for name, value in performance.items()}}
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')

def _append_validation_selection_event(history: list[dict[str, float]], *, accepted_update_state: int, validation: Mapping[str, object], pretrain_validation: Mapping[str, object]) -> tuple[bool, tuple[float, float] | None]:
    """Append one vehicle-mode validation and return its trailing score."""
    vehicle_gain = float(validation['validation/vehicle_reward_gain_vs_fixed_pretrain'])
    selected_gain = float(validation['validation/selected_reward_gain'])
    s7_out_delta = float(validation['validation/S7/out_delta'])
    if not all(math.isfinite(value) for value in (vehicle_gain, selected_gain, s7_out_delta)):
        raise OnlineGRPOError('checkpoint selection gains must be finite')
    safety_eligible = _validation_is_safety_eligible(validation, pretrain_validation)
    history.append({
        'accepted_update_state': float(accepted_update_state),
        'vehicle_reward_gain': vehicle_gain,
        'selected_reward_gain': selected_gain,
        's7_out_delta': s7_out_delta,
        'safety_eligible': float(safety_eligible),
    })
    if accepted_update_state < 60 or len(history) < 3:
        return safety_eligible, None
    trailing = history[-3:]
    return safety_eligible, (
        float(np.mean([value['vehicle_reward_gain'] for value in trailing])),
        float(np.mean([value['selected_reward_gain'] for value in trailing])),
    )
