"""Experiment contracts, reproducibility metadata, and checkpoint state."""
from __future__ import annotations
import dataclasses
import hashlib
import math
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import torch

from models.bev_planner import (
    DEFAULT_DDIM_PATH,
    JointGRPOConfig,
    KinematicTrajectoryOptimizerConfig,
    joint_grpo_optimizer_contract,
    joint_grpo_optimizer_contract_sha256,
)
from models.bev_planner.vehicle_mode_reward import (
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256,
    VEHICLE_MODE_REWARD_CONTRACT,
    VEHICLE_MODE_REWARD_CONTRACT_SHA256,
    VehicleModeRewardConfig,
    vehicle_mode_reward_config_sha256,
)
from train.bev_joint_grpo import grpo_b_checkpoint_payload, grpo_checkpoint_payload

from .config import JointGRPOOnlineConfig, OnlineGRPOError
from .rollout import _attempt_budget_is_exhausted, _next_unfinished_bucket_index
ROLLOUT_COLLECTION_CONTRACT_VERSION = "stage2_joint_grpo_persistent_episode_v8"
BEST_CHECKPOINT_METRIC = "validation/safety_constrained_vehicle_reward_gain_trailing3"
VALIDATION_STATE_BANK_FORMAT = "bev_joint_grpo_validation_state_bank_v1"
MUTABLE_RUNTIME_CONFIG_PATH = "configs/train/bev_joint_grpo.yaml"
FROZEN_PRETRAIN_VEHICLE_REWARD_TAG = "reward/same_mode_pretrain_vehicle_reward_mean"
VALIDATION_REWARD_FAMILY = "vehicle_mode_counterfactual"

def _performance_summary(totals: Mapping[str, float], *, validation_calls: int) -> dict[str, object]:
    """Return accumulated wall timings without changing checkpoint payloads."""
    return {'timing_totals_seconds': {str(name): float(value) for name, value in sorted(totals.items())}, 'validation_calls': int(validation_calls)}

def _implementation_commit() -> str:
    """Return the committed code identity while allowing the runtime YAML."""
    try:
        commit = subprocess.run(('git', 'rev-parse', 'HEAD'), check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(('git', 'status', '--porcelain=v1', '--untracked-files=no'), check=True, capture_output=True, text=True).stdout.rstrip('\n')
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OnlineGRPOError('unable to resolve GRPO implementation commit') from exc
    disallowed = []
    for record in dirty.splitlines():
        status = record[:2]
        path = record[3:] if len(record) >= 4 else ''
        runtime_config_edit = path == MUTABLE_RUNTIME_CONFIG_PATH and status in {' M', 'M ', 'MM'}
        if not runtime_config_edit:
            disallowed.append(record)
    if len(commit) != 40 or disallowed:
        detail = ', '.join(disallowed) if disallowed else 'invalid HEAD'
        raise OnlineGRPOError(f'bounded GRPO diagnostics require committed implementation files; disallowed tracked changes: {detail}')
    return commit

def _frozen_pretrain_reward_logging_metadata(planner: torch.nn.Module) -> dict[str, object]:
    """Describe the per-live-state frozen Stage1 reward diagnostic."""
    planner_config = planner.config
    return {'tensorboard_tag': FROZEN_PRETRAIN_VEHICLE_REWARD_TAG, 'metrics_jsonl_field': FROZEN_PRETRAIN_VEHICLE_REWARD_TAG, 'step_axis': 'absolute_accepted_update_state', 'same_live_state_as_current_exploration': True, 'reward_baseline_trajectory_count': 30, 'reward_baseline_selection': 'all_valid_vehicle_modes', 'execution_trajectory_count': 3, 'execution_selection': 'valid_mode_masked_argmax_per_vehicle', 'trajectory_domain': 'raw_tau_d', 'inference': 'frozen_stage1_standard_deterministic_ddim', 'inference_seed': int(planner_config.inference_seed), 'ddim_path': DEFAULT_DDIM_PATH.as_dict(), 'fixed_inference_noise': True, 'advantage_formula_role': 'paired_sample_baseline_filter', 'active_mode_gate_role': 'none; signal is derived per sample', 'environment_action_role': 'sole_executed_policy'}

def rollout_collection_contract(config: JointGRPOOnlineConfig) -> dict[str, object]:
    """Return the exact persistent-episode collection protocol."""
    return {'version': ROLLOUT_COLLECTION_CONTRACT_VERSION, 'budget_unit': 'accepted_update_state', 'attempt_budget_unit': 'same_live_state_noise_resample', 'pretrain_baseline_domain': 'raw_tau_d', 'pretrain_inference_per_live_state': 1, 'comparison_unit': 'vehicle_mode', 'trajectories_per_mode': int(config.trajectories_per_mode), 'advantage': 'fixed_scale_baseline_relative_safety_truncation', 'signal_mode': 'at_least_one_nonzero_sample_advantage', 'max_sampling_attempts_per_state': int(config.max_sampling_attempts_per_state), 'max_sampling_attempts_multiplier': int(config.max_sampling_attempts_multiplier), 'max_consecutive_empty_episodes': int(config.max_consecutive_empty_episodes), 'retry_scope': 'same_live_state', 'partial_active_policy': 'merge_all_signal_modes_without_backfill', 'retry_condition': 'all_vehicle_modes_have_zero_signal', 'sampled_candidate_execution': False, 'environment_action': 'cached_frozen_stage1_argmax_only', 'bucket_order': 'ordered_scenario_then_seed', 'bucket_target_assignment': 'balanced_floor_with_remainder_to_lower_bucket_indices', 'rollout_groups_per_bucket_visit': int(config.rollout_groups_per_bucket_visit), 'environment_steps_per_episode': int(config.environment_steps_per_episode), 'validation_interval_rollouts': int(config.validation_interval_rollouts), 'baseline_environment_steps_per_live_state': 1, 'episode_reuse': 'persistent_within_bucket_visit', 'interrupted_visit_resume': 'same_bucket_same_visit_progress', 'checkpoint_boundary': 'closed_environment_only', 'active_environment_serialized': False, 'rule_maker_commitment_scope': 'live_environment_episode', 'rollout_start_ready_gate': 'history_ready_and_primary_scenario_ready', 'rollout_start_offset_distribution': 'inclusive_uniform_integer', 'rollout_start_generator': 'independent_rollout_start_torch_generator', 'rollout_initial_noise': 'stateless_generator_derived_from_training_seed_live_state_retry', 'ddim_transition_noise': 'independent_stateless_generator_derived_from_training_seed_live_state_retry', 'retry_sequence_invariance': 'later_live_state_noise_is_independent_of_prior_retry_count', 'rollout_start_offset_max_steps': int(config.rollout_start_offset_max_steps), 'rollout_start_min_remaining_steps': int(config.rollout_start_min_remaining_steps), 'rollout_start_rng_draws': 'exactly_one_per_feasible_live_training_episode_including_zero_upper_bound', 'rollout_start_upper_bound': 'min(configured_max,episode_cap_minus_t_ready_minus_min_remaining)', 'scenario_window_close': {'S5_hard_brake_lead': 'conflict_evidence.formation_recovered_after_hazard', 'S6_background_merge_in': 'conflict_evidence.formation_recovered_after_merge', 'S7_ego_merge_from_ramp': 'route_completion.all_agents_entered_mainline_and_conflict_evidence.formation_recovered_after_merge', 'S8_ego_exit_to_ramp': 'route_completion.all_agents_continued_on_exit_ramp_and_conflict_evidence.formation_recovered_on_ramp', 'S9_narrow_channel_negotiation': 'route_completion.all_agents_returned_to_original_lane_and_conflict_evidence.formation_recovered_after_return'}, 'late_target_retry': 'same_bucket_same_visit_with_temporary_observed_last_open_upper_bound', 'partial_visit_policy': 'retain_completed_rollouts_and_updates_then_new_episode_fresh_offset', 'validation_start': 'earliest_ready_without_random_offset'}

def _checkpoint_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    except OSError as exc:
        raise OnlineGRPOError(f'unable to read checkpoint: {path}') from exc
    return digest.hexdigest()

def _reward_contract_version() -> str:
    version = VEHICLE_MODE_REWARD_CONTRACT.get('version')
    if not isinstance(version, str) or not version:
        raise OnlineGRPOError('vehicle-mode reward contract version is invalid')
    return version

def _application_contract_version() -> str:
    version = GRPO_OPEN_REWARD_APPLICATION_CONTRACT.get('version')
    if not isinstance(version, str) or not version:
        raise OnlineGRPOError('GRPO-Open application contract version is invalid')
    return version

def _validate_vehicle_mode_reward_config(config: VehicleModeRewardConfig, *, trajectories_per_mode: int) -> None:
    if not isinstance(config, VehicleModeRewardConfig):
        raise OnlineGRPOError('training reward config must be VehicleModeRewardConfig')
    if config.trajectories_per_mode != trajectories_per_mode:
        raise OnlineGRPOError('vehicle-mode reward trajectory count does not match GRPO config')

def _grpo_config_artifact_payload(config: JointGRPOConfig) -> dict[str, object]:
    return dataclasses.asdict(config)

def _sampler_state(*, accepted_update_states: int, sampling_attempts: int, rejected_sampling_attempts: int, stability_guard_rejections: int, exhausted_states: int, baseline_execution_steps: int, zero_signal_epochs: int, warmup_environment_steps: int, bucket_target_counts: Sequence[int], bucket_accepted_update_counts: Sequence[int], bucket_sampling_attempt_counts: Sequence[int], bucket_rejected_sampling_attempt_counts: Sequence[int], bucket_stability_guard_rejection_counts: Sequence[int], bucket_exhausted_state_counts: Sequence[int], bucket_baseline_execution_step_counts: Sequence[int], bucket_zero_signal_epoch_counts: Sequence[int], bucket_optimizer_step_counts: Sequence[int], bucket_episode_counts: Sequence[int], next_bucket_index: int, current_visit_progress: int, rollout_start_generator_state: torch.Tensor, last_validated_update_state: int, rollout_groups_per_bucket_visit: int, optimizer_step: int, environment_steps: int, max_sampling_attempts: int) -> dict[str, object]:
    raw = {'accepted_update_states': accepted_update_states, 'sampling_attempts': sampling_attempts, 'rejected_sampling_attempts': rejected_sampling_attempts, 'stability_guard_rejections': stability_guard_rejections, 'exhausted_states': exhausted_states, 'baseline_execution_steps': baseline_execution_steps, 'zero_signal_epochs': zero_signal_epochs, 'warmup_environment_steps': warmup_environment_steps, 'bucket_target_counts': list(bucket_target_counts), 'bucket_accepted_update_counts': list(bucket_accepted_update_counts), 'bucket_sampling_attempt_counts': list(bucket_sampling_attempt_counts), 'bucket_rejected_sampling_attempt_counts': list(bucket_rejected_sampling_attempt_counts), 'bucket_stability_guard_rejection_counts': list(bucket_stability_guard_rejection_counts), 'bucket_exhausted_state_counts': list(bucket_exhausted_state_counts), 'bucket_baseline_execution_step_counts': list(bucket_baseline_execution_step_counts), 'bucket_zero_signal_epoch_counts': list(bucket_zero_signal_epoch_counts), 'bucket_optimizer_step_counts': list(bucket_optimizer_step_counts), 'bucket_episode_counts': list(bucket_episode_counts), 'next_bucket_index': next_bucket_index, 'current_visit_progress': current_visit_progress, 'generator_states': {'rollout_start': rollout_start_generator_state.detach().cpu().clone()}, 'last_validated_update_state': last_validated_update_state}
    return _validate_sampler_state(raw, bucket_count=len(bucket_accepted_update_counts), expected_bucket_target_counts=bucket_target_counts, rollout_groups_per_bucket_visit=rollout_groups_per_bucket_visit, optimizer_step=optimizer_step, environment_steps=environment_steps, max_sampling_attempts=max_sampling_attempts)

def _validate_sampler_state(raw: object, *, bucket_count: int, expected_bucket_target_counts: Sequence[int], rollout_groups_per_bucket_visit: int, optimizer_step: int, environment_steps: int, max_sampling_attempts: int) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise OnlineGRPOError('online GRPO checkpoint sampler_state is invalid')
    if isinstance(bucket_count, bool) or not isinstance(bucket_count, int) or bucket_count <= 0:
        raise OnlineGRPOError('training bucket count must be a positive integer')
    raw = dict(raw)
    raw.setdefault('stability_guard_rejections', 0)
    raw.setdefault('bucket_stability_guard_rejection_counts', [0 for _ in range(bucket_count)])
    if isinstance(rollout_groups_per_bucket_visit, bool) or not isinstance(rollout_groups_per_bucket_visit, int) or rollout_groups_per_bucket_visit <= 0:
        raise OnlineGRPOError('rollout bucket visit quota is invalid')
    for name, minimum in (('accepted_update_states', 0), ('sampling_attempts', 0), ('rejected_sampling_attempts', 0), ('stability_guard_rejections', 0), ('exhausted_states', 0), ('baseline_execution_steps', 0), ('zero_signal_epochs', 0), ('warmup_environment_steps', 0), ('next_bucket_index', 0), ('current_visit_progress', 0), ('last_validated_update_state', -1)):
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise OnlineGRPOError(f'online GRPO checkpoint sampler_state {name} is invalid')
    accepted = int(raw['accepted_update_states'])
    attempts = int(raw['sampling_attempts'])
    rejected = int(raw['rejected_sampling_attempts'])
    guard_rejected = int(raw['stability_guard_rejections'])
    exhausted = int(raw['exhausted_states'])
    baseline_steps = int(raw['baseline_execution_steps'])
    zero_signal = int(raw['zero_signal_epochs'])
    warmup_steps = int(raw['warmup_environment_steps'])
    next_bucket = int(raw['next_bucket_index'])
    current_visit_progress = int(raw['current_visit_progress'])
    last_validated = int(raw['last_validated_update_state'])
    if attempts != accepted + rejected + guard_rejected or last_validated > accepted:
        raise OnlineGRPOError('online GRPO checkpoint sampler counters conflict')
    if baseline_steps != accepted + exhausted + guard_rejected:
        raise OnlineGRPOError('online GRPO baseline execution counters conflict')
    if isinstance(environment_steps, bool) or not isinstance(environment_steps, int) or environment_steps < 0 or (environment_steps != warmup_steps + baseline_steps):
        raise OnlineGRPOError('online GRPO checkpoint environment step counters conflict')
    _attempt_budget_is_exhausted(accepted_update_states=accepted, target_accepted_update_states=sum((int(value) for value in expected_bucket_target_counts)), sampling_attempts=attempts, max_sampling_attempts=max_sampling_attempts)
    if next_bucket >= bucket_count:
        raise OnlineGRPOError('online GRPO checkpoint bucket cursor is invalid')
    counts: dict[str, list[int]] = {}
    for name in ('bucket_target_counts', 'bucket_accepted_update_counts', 'bucket_sampling_attempt_counts', 'bucket_rejected_sampling_attempt_counts', 'bucket_stability_guard_rejection_counts', 'bucket_exhausted_state_counts', 'bucket_baseline_execution_step_counts', 'bucket_zero_signal_epoch_counts', 'bucket_optimizer_step_counts', 'bucket_episode_counts'):
        values = raw.get(name)
        if not isinstance(values, (list, tuple)) or len(values) != bucket_count or any((isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)):
            raise OnlineGRPOError(f'online GRPO checkpoint sampler_state {name} is invalid')
        counts[name] = [int(value) for value in values]
    expected_targets = [int(value) for value in expected_bucket_target_counts]
    if len(expected_targets) != bucket_count or counts['bucket_target_counts'] != expected_targets:
        raise OnlineGRPOError('online GRPO checkpoint bucket targets mismatch')
    if sum(counts['bucket_accepted_update_counts']) != accepted:
        raise OnlineGRPOError('online GRPO checkpoint bucket accepts conflict')
    if sum(counts['bucket_sampling_attempt_counts']) != attempts:
        raise OnlineGRPOError('online GRPO checkpoint bucket attempts conflict')
    if sum(counts['bucket_rejected_sampling_attempt_counts']) != rejected:
        raise OnlineGRPOError('online GRPO checkpoint bucket rejections conflict')
    if sum(counts['bucket_stability_guard_rejection_counts']) != guard_rejected:
        raise OnlineGRPOError('online GRPO checkpoint bucket guard rejections conflict')
    if sum(counts['bucket_exhausted_state_counts']) != exhausted:
        raise OnlineGRPOError('online GRPO checkpoint bucket exhaustions conflict')
    if sum(counts['bucket_baseline_execution_step_counts']) != baseline_steps:
        raise OnlineGRPOError('online GRPO checkpoint baseline steps conflict')
    if sum(counts['bucket_zero_signal_epoch_counts']) != zero_signal:
        raise OnlineGRPOError('online GRPO checkpoint zero-signal epochs conflict')
    if sum(counts['bucket_optimizer_step_counts']) != optimizer_step:
        raise OnlineGRPOError('online GRPO checkpoint bucket updates conflict')
    if any((attempt_count != accepted_count + rejected_count + guard_rejected_count for attempt_count, accepted_count, rejected_count, guard_rejected_count in zip(counts['bucket_sampling_attempt_counts'], counts['bucket_accepted_update_counts'], counts['bucket_rejected_sampling_attempt_counts'], counts['bucket_stability_guard_rejection_counts']))):
        raise OnlineGRPOError('online GRPO checkpoint bucket attempt counters conflict')
    if any((baseline_count != accepted_count + exhausted_count + guard_rejected_count for baseline_count, accepted_count, exhausted_count, guard_rejected_count in zip(counts['bucket_baseline_execution_step_counts'], counts['bucket_accepted_update_counts'], counts['bucket_exhausted_state_counts'], counts['bucket_stability_guard_rejection_counts']))):
        raise OnlineGRPOError('online GRPO bucket baseline counters conflict')
    if any((sampled_count > target_count for sampled_count, target_count in zip(counts['bucket_accepted_update_counts'], counts['bucket_target_counts']))):
        raise OnlineGRPOError('online GRPO checkpoint bucket target exceeded')
    if any(((attempt_count > 0 or baseline_count > 0) and episode_count == 0 for attempt_count, baseline_count, episode_count in zip(counts['bucket_sampling_attempt_counts'], counts['bucket_baseline_execution_step_counts'], counts['bucket_episode_counts']))):
        raise OnlineGRPOError('online GRPO checkpoint bucket episodes conflict')
    unfinished = _next_unfinished_bucket_index(counts['bucket_accepted_update_counts'], counts['bucket_target_counts'], start_index=next_bucket)
    if unfinished is None:
        if next_bucket != 0 or current_visit_progress != 0:
            raise OnlineGRPOError('online GRPO checkpoint completed bucket cursor is invalid')
    else:
        if unfinished != next_bucket:
            raise OnlineGRPOError('online GRPO checkpoint bucket cursor is invalid')
        expected_progress = counts['bucket_accepted_update_counts'][next_bucket] % rollout_groups_per_bucket_visit
        if current_visit_progress != expected_progress:
            raise OnlineGRPOError('online GRPO checkpoint current visit progress is invalid')
    for index, (sample_count, target_count) in enumerate(zip(counts['bucket_accepted_update_counts'], counts['bucket_target_counts'])):
        if index != next_bucket and sample_count < target_count and (sample_count % rollout_groups_per_bucket_visit != 0):
            raise OnlineGRPOError('online GRPO checkpoint has multiple partial bucket visits')
    generator_states = raw.get('generator_states')
    if not isinstance(generator_states, Mapping) or set(generator_states) != {'rollout_start'}:
        raise OnlineGRPOError('online GRPO checkpoint generator_states is invalid')
    for generator_state in generator_states.values():
        if not isinstance(generator_state, torch.Tensor) or generator_state.dtype != torch.uint8 or generator_state.ndim != 1 or (generator_state.numel() == 0):
            raise OnlineGRPOError('online GRPO checkpoint generator_states is invalid')
    return {'accepted_update_states': accepted, 'sampling_attempts': attempts, 'rejected_sampling_attempts': rejected, 'stability_guard_rejections': guard_rejected, 'exhausted_states': exhausted, 'baseline_execution_steps': baseline_steps, 'zero_signal_epochs': zero_signal, 'warmup_environment_steps': warmup_steps, **counts, 'next_bucket_index': next_bucket, 'current_visit_progress': current_visit_progress, 'generator_states': {name: value.detach().cpu().clone() for name, value in generator_states.items()}, 'last_validated_update_state': last_validated}

def _validated_selection_history(raw: object) -> list[dict[str, float]]:
    if not isinstance(raw, (list, tuple)):
        raise OnlineGRPOError('validation selection history must be a sequence')
    expected = {'accepted_update_state', 'vehicle_reward_gain', 'selected_reward_gain', 's7_out_delta', 'safety_eligible'}
    checked: list[dict[str, float]] = []
    previous_state = -1.0
    for entry in raw:
        if not isinstance(entry, Mapping) or set(entry) != expected:
            raise OnlineGRPOError('validation selection history entry is invalid')
        values = {name: float(entry[name]) for name in expected}
        if not all((math.isfinite(value) for value in values.values())):
            raise OnlineGRPOError('validation selection history must be finite')
        if values['accepted_update_state'] <= previous_state or values['safety_eligible'] not in (0.0, 1.0):
            raise OnlineGRPOError('validation selection history order is invalid')
        previous_state = values['accepted_update_state']
        checked.append(values)
    return checked

def _checkpoint_payload(*, variant: str, trainer: object, source_sha: str, source_payload: Mapping[str, object], metrics: Mapping[str, float], diagnostic_only: bool, run_mode: str, reward_config: VehicleModeRewardConfig, scenario_contract_sha: str, scenario_seeds: Sequence[int], environment_steps: int, best_validation_reward: float | None, best_selected_reward_gain: float | None, best_checkpoint_sha256: str | None, validation_selection_history: Sequence[Mapping[str, float]], collection_contract: Mapping[str, object], sampler_state: Mapping[str, object]) -> dict[str, object]:
    _validate_vehicle_mode_reward_config(reward_config, trajectories_per_mode=trainer.config.trajectories_per_mode)
    if (best_validation_reward is None) is not (best_selected_reward_gain is None):
        raise OnlineGRPOError('best checkpoint score must be complete or absent')
    if best_validation_reward is not None and (not math.isfinite(float(best_validation_reward)) or not math.isfinite(float(best_selected_reward_gain))):
        raise OnlineGRPOError('best checkpoint score must be finite')
    if best_checkpoint_sha256 is not None and best_validation_reward is None:
        raise OnlineGRPOError('best checkpoint SHA requires an eligible score')
    checked_history = _validated_selection_history(validation_selection_history)
    if best_checkpoint_sha256 is not None:
        if len(best_checkpoint_sha256) != 64 or best_checkpoint_sha256 != best_checkpoint_sha256.lower():
            raise OnlineGRPOError('best checkpoint SHA256 is invalid')
        try:
            int(best_checkpoint_sha256, 16)
        except ValueError as exc:
            raise OnlineGRPOError('best checkpoint SHA256 is invalid') from exc
    builder = grpo_checkpoint_payload if variant == 'A' else grpo_b_checkpoint_payload
    payload = builder(trainer=trainer, source_stage1_sha256=source_sha, source_stage1_payload=source_payload, metrics=metrics, diagnostic_only=diagnostic_only)
    payload.update({'run_mode': run_mode, 'reward_contract_version': _reward_contract_version(), 'reward_contract_sha256': VEHICLE_MODE_REWARD_CONTRACT_SHA256, 'reward_config': dataclasses.asdict(reward_config), 'reward_config_sha256': vehicle_mode_reward_config_sha256(reward_config), 'reward_application_contract': GRPO_OPEN_REWARD_APPLICATION_CONTRACT, 'reward_application_contract_sha256': GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256, 'reward_input_domain': 'tau_d', 'training_candidate_domain': 'tau_d_all_vehicle_modes', 'environment_action_source': 'cached_frozen_stage1_argmax', 'execution_input_domain': 'tau_cmd', 'best_checkpoint_metric': BEST_CHECKPOINT_METRIC, 'validation_reward_family': VALIDATION_REWARD_FAMILY, 'tracking_expansion_enabled': False, 'calibration_required': False, 'scenario_contract_sha256': scenario_contract_sha, 'scenario_seeds': [int(value) for value in scenario_seeds], 'environment_steps': int(environment_steps), 'best_validation_reward': None if best_validation_reward is None else float(best_validation_reward), 'best_selected_reward_gain': None if best_selected_reward_gain is None else float(best_selected_reward_gain), 'best_checkpoint_sha256': best_checkpoint_sha256, 'validation_selection_history': checked_history, 'policy_update_contract': joint_grpo_optimizer_contract(), 'policy_update_contract_sha256': joint_grpo_optimizer_contract_sha256(), 'rollout_collection_contract': dict(collection_contract), 'sampler_state': dict(sampler_state), 'trajectory_optimizer_config': dataclasses.asdict(KinematicTrajectoryOptimizerConfig()), 'trajectory_optimizer_sha256': KinematicTrajectoryOptimizerConfig().sha256()})
    return payload

def _validate_online_checkpoint_metadata(payload: Mapping[str, object], *, run_mode: str, reward_config: VehicleModeRewardConfig, scenario_contract_sha: str, scenario_seeds: Sequence[int], collection_contract: Mapping[str, object], bucket_count: int, bucket_target_counts: Sequence[int], rollout_groups_per_bucket_visit: int, optimizer_step: int, max_sampling_attempts: int | None=None) -> dict[str, object]:
    _validate_vehicle_mode_reward_config(reward_config, trajectories_per_mode=int(collection_contract.get('trajectories_per_mode', -1)))
    legacy_fields = ('calibration_report_sha256', 'calibration_gate_bypassed', 'calibration_report_passed', 'calibration_blockers', 'joint_reward_role', 'simulator_validation_role')
    if any((name in payload for name in legacy_fields)):
        raise OnlineGRPOError('online GRPO checkpoint contains legacy calibration semantics')
    expected = {'run_mode': run_mode, 'diagnostic_only': run_mode != 'formal', 'eligible_for_formal_training': run_mode == 'formal', 'reward_contract_version': _reward_contract_version(), 'reward_contract_sha256': VEHICLE_MODE_REWARD_CONTRACT_SHA256, 'reward_config': dataclasses.asdict(reward_config), 'reward_config_sha256': vehicle_mode_reward_config_sha256(reward_config), 'reward_application_contract': GRPO_OPEN_REWARD_APPLICATION_CONTRACT, 'reward_application_contract_sha256': GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256, 'reward_input_domain': 'tau_d', 'training_candidate_domain': 'tau_d_all_vehicle_modes', 'environment_action_source': 'cached_frozen_stage1_argmax', 'execution_input_domain': 'tau_cmd', 'best_checkpoint_metric': BEST_CHECKPOINT_METRIC, 'validation_reward_family': VALIDATION_REWARD_FAMILY, 'tracking_expansion_enabled': False, 'calibration_required': False, 'policy_update_contract': joint_grpo_optimizer_contract(), 'policy_update_contract_sha256': joint_grpo_optimizer_contract_sha256(), 'rollout_collection_contract': dict(collection_contract), 'scenario_contract_sha256': scenario_contract_sha, 'scenario_seeds': [int(value) for value in scenario_seeds], 'trajectory_optimizer_config': dataclasses.asdict(KinematicTrajectoryOptimizerConfig()), 'trajectory_optimizer_sha256': KinematicTrajectoryOptimizerConfig().sha256()}
    for name, value in expected.items():
        if payload.get(name) != value:
            raise OnlineGRPOError(f'online GRPO checkpoint {name} mismatch')
    environment_steps = payload.get('environment_steps')
    if isinstance(environment_steps, bool) or not isinstance(environment_steps, int) or environment_steps < 0:
        raise OnlineGRPOError('online GRPO checkpoint environment_steps is invalid')
    best_reward = payload.get('best_validation_reward')
    best_selected = payload.get('best_selected_reward_gain')
    if (best_reward is None) is not (best_selected is None):
        raise OnlineGRPOError('online GRPO checkpoint best score is invalid')
    if best_reward is not None and (isinstance(best_reward, bool) or not isinstance(best_reward, (int, float)) or (not math.isfinite(float(best_reward))) or isinstance(best_selected, bool) or (not isinstance(best_selected, (int, float))) or (not math.isfinite(float(best_selected)))):
        raise OnlineGRPOError('online GRPO checkpoint best score is invalid')
    best_sha = payload.get('best_checkpoint_sha256')
    if best_sha is not None:
        if best_reward is None:
            raise OnlineGRPOError('online GRPO checkpoint best SHA has no score')
        if not isinstance(best_sha, str) or len(best_sha) != 64 or best_sha != best_sha.lower():
            raise OnlineGRPOError('online GRPO checkpoint best_checkpoint_sha256 is invalid')
        try:
            int(best_sha, 16)
        except ValueError as exc:
            raise OnlineGRPOError('online GRPO checkpoint best_checkpoint_sha256 is invalid') from exc
    _validated_selection_history(payload.get('validation_selection_history'))
    if max_sampling_attempts is None:
        multiplier = collection_contract.get('max_sampling_attempts_multiplier')
        if isinstance(multiplier, bool) or not isinstance(multiplier, int):
            raise OnlineGRPOError('online GRPO collection attempt multiplier is invalid')
        max_sampling_attempts = multiplier * sum((int(value) for value in bucket_target_counts))
    return _validate_sampler_state(payload.get('sampler_state'), bucket_count=bucket_count, expected_bucket_target_counts=bucket_target_counts, rollout_groups_per_bucket_visit=rollout_groups_per_bucket_visit, optimizer_step=optimizer_step, environment_steps=environment_steps, max_sampling_attempts=max_sampling_attempts)
