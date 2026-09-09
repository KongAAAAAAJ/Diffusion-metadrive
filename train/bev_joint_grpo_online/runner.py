from __future__ import annotations
import dataclasses, json, shutil, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from torch.utils.tensorboard import SummaryWriter
from evaluation.plot_grpo import ACCEPTED_ROLLOUT_AXIS_LABEL, ADVANTAGE_VECTOR_TAG, generate_grpo_plots
from expert_dataset.collect_joint_bev import JointBEVSampleBuilder, simulator_decision_dt_s
from models.bev_planner import DEFAULT_DDIM_PATH, JointGRPOConfig, KinematicTrajectoryOptimizer, KinematicTrajectoryOptimizerConfig, TrajectoryOptimizationError, joint_grpo_optimizer_contract, joint_grpo_optimizer_contract_sha256
from models.bev_planner.vehicle_mode_reward import GRPO_OPEN_REWARD_APPLICATION_CONTRACT, GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256, VEHICLE_MODE_REWARD_CONTRACT, VEHICLE_MODE_REWARD_CONTRACT_SHA256, VehicleModeCounterfactualReward, VehicleModeRewardConfig, vehicle_mode_reward_config_sha256
from train.bev_joint_grpo import load_grpo_b_checkpoint, load_grpo_checkpoint, save_grpo_checkpoint
from scenarios.bev_round13_contract import HOLDOUT_SEEDS, BEVScenarioContractError, primary_scenario_contract, validate_primary_scenario_contract
from .config import AGENT_IDS, JointGRPOTrainingConfig, OnlineGRPOError
from .contracts import BEST_CHECKPOINT_METRIC, FROZEN_PRETRAIN_VEHICLE_REWARD_TAG, VALIDATION_REWARD_FAMILY, _application_contract_version, _checkpoint_file_sha256, _checkpoint_payload, _frozen_pretrain_reward_logging_metadata, _grpo_config_artifact_payload, _implementation_commit, _performance_summary, _reward_contract_version, _sampler_state, _validate_online_checkpoint_metadata, rollout_collection_contract
from .environment import _condition_online_model_inputs, _cuda_peak_memory_bytes, _device, _new_env, _new_online_rule_maker, _reset_cuda_peak_memory, _sample_rollout_start_offset, _scenario_ready_from_summary, _scenario_sampling_window_closed_from_summary, _scenario_summary, episode_has_ended, execute_cached_frozen_baseline, execution_mode_valid_mask, model_inputs_to_batch, route_following_warmup_actions
from .logging import _advantage_scalar_metrics, _append_validation_selection_event, _split_loss_metrics_by_step_axis, _write_advantage_vector_summary, _write_baseline_execution_event, _write_dynamic_sampling_attempt_event, _write_rollout_start_event
from .rollout import _attempt_budget_is_exhausted, _balanced_bucket_targets, _bucket_visit_is_complete, _diffusion_attempt_generators, _fixed_scale_reward_signals, _load_trainer, _next_unfinished_bucket_index, _round_robin_training_buckets
from .validation import _FixedValidationFrozenEntry, _fixed_vehicle_mode_validation, _load_grpo_validation_state_bank, _resume_best_checkpoint_anchor, _validation_reward_comparison_metrics
def run_joint_grpo_training(training_config: JointGRPOTrainingConfig, *, run_dir: Path) -> dict[str, object]:
    if not isinstance(training_config, JointGRPOTrainingConfig):
        raise OnlineGRPOError('training_config must be a JointGRPOTrainingConfig')
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise OnlineGRPOError(f'run_dir must already exist: {run_dir}')
    if not run_dir.is_dir():
        raise OnlineGRPOError(f'run_dir must be a directory: {run_dir}')
    unexpected_entries = sorted((child.name for child in run_dir.iterdir() if child.name != 'training.log'))
    if unexpected_entries:
        raise OnlineGRPOError(f"run_dir must be reserved and may contain only training.log; found: {', '.join(unexpected_entries)}")
    training_log = run_dir / 'training.log'
    if training_log.exists() and (not training_log.is_file()):
        raise OnlineGRPOError(f'run_dir training.log must be a file: {training_log}')
    (run_dir / 'checkpoints').mkdir()
    started_at = time.monotonic()
    performance_totals: defaultdict[str, float] = defaultdict(float)
    validation_calls = 0
    fixed_validation_cache: dict[tuple[tuple[str, str], int], _FixedValidationFrozenEntry] = {}
    config = training_config.online
    variant = training_config.variant
    run_mode = training_config.run_mode
    source_checkpoint = training_config.source_checkpoint
    target_accepted_update_states = config.total_rollout_groups
    max_sampling_attempts = config.max_sampling_attempts_multiplier * target_accepted_update_states
    reward_config = VehicleModeRewardConfig(trajectories_per_mode=config.trajectories_per_mode)
    try:
        scenario_contract_sha = validate_primary_scenario_contract(primary_scenario_contract(config.scenarios))
    except BEVScenarioContractError as exc:
        raise OnlineGRPOError(str(exc)) from exc
    torch_device = _device(config.device)
    grpo_config = JointGRPOConfig(trajectories_per_mode=config.trajectories_per_mode)
    trainer, source_payload, source_sha = _load_trainer(variant, Path(source_checkpoint), torch_device, grpo_config=grpo_config, allow_diagnostic_source=run_mode == 'smoke')
    if run_mode == 'formal' and source_payload.get('eligible_for_formal_training') is not True:
        raise OnlineGRPOError('formal GRPO requires an eligible Stage 1 source')
    implementation_commit = _implementation_commit()
    validation_state_bank, validation_state_bank_sha256 = _load_grpo_validation_state_bank(config.validation_state_bank, scenarios=config.scenarios, seeds=HOLDOUT_SEEDS)
    pretrain_validation = _fixed_vehicle_mode_validation(trainer, device=torch_device, reward_config=reward_config, scenarios=config.scenarios, seeds=HOLDOUT_SEEDS, validation_state_bank=validation_state_bank, frozen_cache=fixed_validation_cache)
    validation_calls += 1
    for name, value in pretrain_validation.items():
        if name.startswith('perf/') and name.endswith('_seconds'):
            performance_totals[name] += float(value)
    checkpoint_loader = load_grpo_checkpoint if variant == 'A' else load_grpo_b_checkpoint
    training_buckets = _round_robin_training_buckets(config.scenarios, config.scenario_seeds)
    bucket_target_counts = _balanced_bucket_targets(target_accepted_update_states, len(training_buckets))
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
    rule_diagnostics = {'conditioned_rollouts': 0, 'proposal_match_attempts': 0, 'proposal_matches': 0, 'condition_failures': 0, 'forced_safe_stops': 0, 's7_feedback_exception_hits': 0, 'commitment_conditioned_rollouts': 0, 'commitment_feedback_incompatible': 0}
    last_metrics: dict[str, float] = {str(name): float(value) for name, value in pretrain_validation.items()}
    resume_best_path: Path | None = None
    best_reward: float | None = None
    best_selected_reward_gain: float | None = None
    best_unconstrained_score: tuple[float, float] | None = None
    validation_selection_history: list[dict[str, float]] = []
    if config.resume_checkpoint is not None:
        resume_payload = checkpoint_loader(config.resume_checkpoint, trainer, expected_source_stage1_sha256=source_sha)
        restored_sampler_state = _validate_online_checkpoint_metadata(resume_payload, run_mode=run_mode, reward_config=reward_config, scenario_contract_sha=scenario_contract_sha, scenario_seeds=config.scenario_seeds, collection_contract=collection_contract, bucket_count=len(training_buckets), bucket_target_counts=bucket_target_counts, rollout_groups_per_bucket_visit=config.rollout_groups_per_bucket_visit, optimizer_step=trainer.optimizer_step, max_sampling_attempts=max_sampling_attempts)
        environment_steps = int(resume_payload['environment_steps'])
        warmup_environment_steps = int(restored_sampler_state['warmup_environment_steps'])
        accepted_update_states = int(restored_sampler_state['accepted_update_states'])
        sampling_attempts = int(restored_sampler_state['sampling_attempts'])
        rejected_sampling_attempts = int(restored_sampler_state['rejected_sampling_attempts'])
        stability_guard_rejections = int(restored_sampler_state['stability_guard_rejections'])
        exhausted_states = int(restored_sampler_state['exhausted_states'])
        baseline_execution_steps = int(restored_sampler_state['baseline_execution_steps'])
        zero_signal_epochs = int(restored_sampler_state['zero_signal_epochs'])
        bucket_accepted_update_counts = list(restored_sampler_state['bucket_accepted_update_counts'])
        bucket_sampling_attempt_counts = list(restored_sampler_state['bucket_sampling_attempt_counts'])
        bucket_rejected_sampling_attempt_counts = list(restored_sampler_state['bucket_rejected_sampling_attempt_counts'])
        bucket_stability_guard_rejection_counts = list(restored_sampler_state['bucket_stability_guard_rejection_counts'])
        bucket_exhausted_state_counts = list(restored_sampler_state['bucket_exhausted_state_counts'])
        bucket_baseline_execution_step_counts = list(restored_sampler_state['bucket_baseline_execution_step_counts'])
        bucket_zero_signal_epoch_counts = list(restored_sampler_state['bucket_zero_signal_epoch_counts'])
        bucket_optimizer_step_counts = list(restored_sampler_state['bucket_optimizer_step_counts'])
        bucket_episode_counts = list(restored_sampler_state['bucket_episode_counts'])
        next_bucket_index = int(restored_sampler_state['next_bucket_index'])
        current_visit_progress = int(restored_sampler_state['current_visit_progress'])
        last_validated_update_state = int(restored_sampler_state['last_validated_update_state'])
        generator_states = restored_sampler_state['generator_states']
        rollout_start_generator.set_state(generator_states['rollout_start'])
        last_metrics = {str(name): float(value) for name, value in resume_payload['metrics'].items()}
        resume_best_path, resumed_best_score, validation_selection_history = _resume_best_checkpoint_anchor(Path(config.resume_checkpoint), resume_payload)
        if resumed_best_score is not None:
            best_reward, best_selected_reward_gain = resumed_best_score
        if accepted_update_states >= target_accepted_update_states:
            raise OnlineGRPOError('resume checkpoint already reached requested accepted rollout groups')
    frozen_pretrain_reward_logging = _frozen_pretrain_reward_logging_metadata(trainer.planner)
    run_start_optimizer_step = trainer.optimizer_step
    run_start_accepted_update_state = accepted_update_states
    run_start_sampling_attempt = sampling_attempts
    online_config = dataclasses.asdict(config)
    online_config['resume_checkpoint'] = str(config.resume_checkpoint) if config.resume_checkpoint is not None else None
    online_config['validation_state_bank'] = str(config.validation_state_bank)
    frozen = {'format': 'bev_joint_grpo_online_config_v14', 'implementation_commit': implementation_commit, 'variant': variant, 'run_mode': run_mode, 'diagnostic_only': run_mode != 'formal', 'eligible_for_formal_training': run_mode == 'formal', 'online_config': online_config, 'grpo_config': _grpo_config_artifact_payload(grpo_config), 'source_stage1_sha256': source_sha, 'policy_update_contract': joint_grpo_optimizer_contract(), 'policy_update_contract_sha256': joint_grpo_optimizer_contract_sha256(), 'rollout_collection_contract': collection_contract, 'reward_contract_version': _reward_contract_version(), 'reward_contract': VEHICLE_MODE_REWARD_CONTRACT, 'reward_contract_sha256': VEHICLE_MODE_REWARD_CONTRACT_SHA256, 'reward_config': dataclasses.asdict(reward_config), 'reward_config_sha256': vehicle_mode_reward_config_sha256(reward_config), 'reward_application_contract': GRPO_OPEN_REWARD_APPLICATION_CONTRACT, 'reward_application_contract_sha256': GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256, 'reward_input_domain': 'tau_d', 'training_candidate_domain': 'tau_d_all_vehicle_modes', 'environment_action_source': 'cached_frozen_stage1_argmax', 'execution_input_domain': 'tau_cmd', 'frozen_pretrain_reward_logging': frozen_pretrain_reward_logging, 'best_checkpoint_metric': BEST_CHECKPOINT_METRIC, 'validation_reward_family': VALIDATION_REWARD_FAMILY, 'tracking_expansion_enabled': False, 'calibration_required': False, 'trajectory_optimizer_config': dataclasses.asdict(KinematicTrajectoryOptimizerConfig()), 'trajectory_optimizer_sha256': KinematicTrajectoryOptimizerConfig().sha256(), 'scenario_contract': primary_scenario_contract(config.scenarios), 'scenario_contract_sha256': scenario_contract_sha, 'validation_state_bank': {'path': str(config.validation_state_bank.resolve()), 'sha256': validation_state_bank_sha256, 'scenarios': [list(value) for value in config.scenarios], 'seeds': [int(value) for value in HOLDOUT_SEEDS], 'state': 'earliest_history_ready_and_primary_scenario_ready', 'noise_seed_formula': '10000019 + 1009 * seed + scenario_index', 'common_random_numbers': True, 'ddim_path': DEFAULT_DDIM_PATH.as_dict()}}
    (run_dir / 'config.json').write_text(json.dumps(frozen, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    metrics_path = run_dir / 'metrics.jsonl'
    writer = SummaryWriter(log_dir=str(run_dir / 'tb'))
    initial_performance = {name: float(value) for name, value in pretrain_validation.items() if name.startswith('perf/')}
    for metric_name, metric_value in initial_performance.items():
        writer.add_scalar(metric_name, metric_value, 0)
    with metrics_path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'event': 'initial_validation_performance', 'accepted_update_state': 0, **initial_performance}, sort_keys=True) + '\n')
    vehicle_reward_backend = VehicleModeCounterfactualReward(reward_config)
    trajectory_optimizer = KinematicTrajectoryOptimizer()
    consecutive_empty_episodes = 0
    best_path = run_dir / 'checkpoints' / 'best.pt'
    best_safe_path = run_dir / 'checkpoints' / 'best_safe.pt'
    best_unconstrained_path = run_dir / 'checkpoints' / 'best_unconstrained.pt'
    milestone_200_path = run_dir / 'checkpoints' / 'milestone_200.pt'
    milestone_500_path = run_dir / 'checkpoints' / 'milestone_500.pt'
    last_path = run_dir / 'checkpoints' / 'last.pt'
    best_checkpoint_sha256: str | None = None
    if resume_best_path is not None:
        shutil.copyfile(resume_best_path, best_path)
        shutil.copyfile(resume_best_path, best_safe_path)
        best_checkpoint_sha256 = _checkpoint_file_sha256(best_path)
    _reset_cuda_peak_memory(torch_device)
    attempt_budget_exhausted = False
    diagnostic_early_stop = False
    stability_guard_rejections_this_run = 0
    stability_guard_diagnostic: dict[str, object] | None = None
    try:
        while accepted_update_states < target_accepted_update_states and (not diagnostic_early_stop) and (stability_guard_diagnostic is None):
            bucket_index = next_bucket_index
            if bucket_accepted_update_counts[bucket_index] >= bucket_target_counts[bucket_index]:
                raise OnlineGRPOError('training bucket cursor points to a completed target')
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
                while accepted_update_states < target_accepted_update_states and (not diagnostic_early_stop) and (stability_guard_diagnostic is None) and (bucket_accepted_update_counts[bucket_index] < bucket_target_counts[bucket_index]) and (current_visit_progress < config.rollout_groups_per_bucket_visit) and (episode_step < config.environment_steps_per_episode):
                    builder.capture_state(env, episode_step * dt_s)
                    history_ready = builder.history_ready()
                    if history_ready:
                        scenario_summary = _scenario_summary(env)
                        ready_now = _scenario_ready_from_summary(scenario_summary)
                        window_closed_now = _scenario_sampling_window_closed_from_summary(scenario_summary)
                    else:
                        ready_now = False
                        window_closed_now = False
                    scenario_window_closed = bool(scenario_window_closed or window_closed_now)
                    if scenario_window_closed:
                        if start_target_step is not None and (not start_accepted):
                            assert start_ready_step is not None
                            assert start_offset is not None
                            assert start_upper_bound is not None
                            last_open_offset = episode_step - start_ready_step - 1
                            if last_open_offset >= 0:
                                temporary_start_offset_upper_bound = min(last_open_offset, temporary_start_offset_upper_bound if temporary_start_offset_upper_bound is not None else last_open_offset)
                                retryable_start_rejection = True
                            rollout_start_rejected_count += 1
                            _write_rollout_start_event(metrics_path, optimizer_step=trainer.optimizer_step, rollout_group=accepted_update_states, bucket_index=bucket_index, scenario=scenario, seed=seed, bucket_episode=bucket_episode_counts[bucket_index], ready_step=start_ready_step, upper_bound=start_upper_bound, sampled_offset=start_offset, target_step=start_target_step, status='rejected_window_closed_before_target')
                        break
                    if start_ready_step is None and ready_now:
                        start_ready_step = episode_step
                        start_offset, start_upper_bound = _sample_rollout_start_offset(config, start_ready_step, generator=rollout_start_generator, temporary_max_offset_steps=temporary_start_offset_upper_bound)
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
                            _write_rollout_start_event(metrics_path, optimizer_step=trainer.optimizer_step, rollout_group=accepted_update_states + 1, bucket_index=bucket_index, scenario=scenario, seed=seed, bucket_episode=bucket_episode_counts[bucket_index], ready_step=start_ready_step, upper_bound=start_upper_bound, sampled_offset=start_offset, target_step=start_target_step, status='accepted')
                        values = builder.build_model_inputs(env)
                        condition = _condition_online_model_inputs(rule_maker, env, builder, values, committed_execution_id=committed_execution_id, committed_plan_actions=committed_plan_actions)
                        values = condition.model_inputs
                        execution_mask = execution_mode_valid_mask(values, optimizer=trajectory_optimizer)
                        batch = model_inputs_to_batch(values, torch_device, mode_valid_mask=execution_mask)
                        accepted_attempt: tuple[object, object, object, np.ndarray, dict[str, float]] | None = None
                        state_performance: defaultdict[str, float] = defaultdict(float)
                        frozen_raw_candidate: np.ndarray | None = None
                        frozen_all_mode_candidates: np.ndarray | None = None
                        frozen_modes: np.ndarray | None = None
                        pretrain_score: object | None = None
                        attempts_for_state = 0
                        for retry_index in range(1, config.max_sampling_attempts_per_state + 1):
                            if retry_index == 1 and _attempt_budget_is_exhausted(accepted_update_states=accepted_update_states, target_accepted_update_states=target_accepted_update_states, sampling_attempts=sampling_attempts, max_sampling_attempts=max_sampling_attempts):
                                attempt_budget_exhausted = True
                                break
                            initial_noise_generator, transition_noise_generator = _diffusion_attempt_generators(device=torch_device, training_seed=config.seed, live_state_index=baseline_execution_steps, retry_index=retry_index - 1)
                            sampling_started = time.perf_counter()
                            rollout = trainer.sample_groups(batch, generator=initial_noise_generator, transition_generator=transition_noise_generator, noise_bundle_identity=(int(config.seed), int(baseline_execution_steps), int(retry_index - 1)))
                            raw_candidates = rollout.candidate_trajectories[0].detach().cpu().numpy().astype(np.float32, copy=False)
                            frozen_paired_candidates = rollout.frozen_candidate_trajectories[0].detach().cpu().numpy().astype(np.float32, copy=False)
                            state_performance['perf/train/n48_sampling_seconds'] += time.perf_counter() - sampling_started
                            attempts_for_state += 1
                            if frozen_raw_candidate is None:
                                frozen_inference_started = time.perf_counter()
                                frozen_pretrain = trainer.infer_frozen_pretrain(rollout)
                                frozen_raw_candidate = frozen_pretrain['selected_trajectory'].detach().cpu().numpy().astype(np.float32, copy=False)
                                frozen_modes = frozen_pretrain['selected_mode'].detach().cpu().numpy().astype(np.int64, copy=False)
                                frozen_all_mode_candidates = frozen_pretrain['all_mode_trajectories'].detach().cpu().numpy().astype(np.float32, copy=False)
                                state_performance['perf/train/frozen_inference_seconds'] += time.perf_counter() - frozen_inference_started
                                pretrain_reward_started = time.perf_counter()
                                pretrain_score = vehicle_reward_backend.score_pretrain(env, values, frozen_all_mode_candidates[0], frozen_raw_candidate[0], execution_mask)
                                state_performance['perf/train/pretrain_reward_seconds'] += time.perf_counter() - pretrain_reward_started
                            assert frozen_raw_candidate is not None
                            assert frozen_all_mode_candidates is not None
                            assert pretrain_score is not None
                            current_reward_started = time.perf_counter()
                            proxy = vehicle_reward_backend.score_candidates(env, values, raw_candidates, frozen_raw_candidate[0], execution_mask, pretrain_score)
                            state_performance['perf/train/current_reward_seconds'] += time.perf_counter() - current_reward_started
                            frozen_reward_started = time.perf_counter()
                            frozen_proxy = vehicle_reward_backend.score_candidates(env, values, frozen_paired_candidates, frozen_raw_candidate[0], execution_mask, pretrain_score)
                            state_performance['perf/train/frozen_reward_seconds'] += time.perf_counter() - frozen_reward_started
                            centered_rewards, advantages, signal_mode_mask = _fixed_scale_reward_signals(proxy.rewards, frozen_proxy.rewards, proxy.collision, proxy.out_of_drivable, proxy.valid_mode_mask)
                            sampling_attempts += 1
                            bucket_sampling_attempt_counts[bucket_index] += 1
                            attempt_metrics = _write_dynamic_sampling_attempt_event(metrics_path, optimizer_step=trainer.optimizer_step, accepted_update_state=accepted_update_states + 1, sampling_attempt=sampling_attempts, retry_index=retry_index, bucket_index=bucket_index, scenario=scenario, seed=seed, reward_result=proxy, paired_frozen_rewards=frozen_proxy.rewards, centered_rewards=centered_rewards, advantages=advantages, hard_valid_mode_mask=np.asarray(values.mode_valid_mask, dtype=np.bool_), signal_mode_mask=signal_mode_mask, reward_config=reward_config)
                            writer.add_scalar('dynamic_sampling/accepted', float(signal_mode_mask.any()), sampling_attempts)
                            writer.add_scalar('dynamic_sampling/signal_mode_count', attempt_metrics['signal_mode_count'], sampling_attempts)
                            if signal_mode_mask.any():
                                accepted_attempt = (rollout, proxy, frozen_proxy, signal_mode_mask, attempt_metrics)
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
                            rollout, proxy, frozen_proxy, signal_mode_mask, accepted_attempt_metrics = accepted_attempt
                            rollout = rollout.with_reward_signals(current_rewards=torch.from_numpy(np.asarray(proxy.rewards, dtype=np.float32)).unsqueeze(0).to(torch_device), frozen_rewards=torch.from_numpy(np.asarray(frozen_proxy.rewards, dtype=np.float32)).unsqueeze(0).to(torch_device), collision_mask=torch.from_numpy(np.asarray(proxy.collision, dtype=np.bool_)).unsqueeze(0).to(torch_device), out_of_drivable_mask=torch.from_numpy(np.asarray(proxy.out_of_drivable, dtype=np.bool_)).unsqueeze(0).to(torch_device), valid_executable_mode_mask=torch.from_numpy(np.asarray(proxy.valid_mode_mask, dtype=np.bool_)).unsqueeze(0).to(torch_device))
                            optimizer_step_before = trainer.optimizer_step
                            update_started = time.perf_counter()
                            update = trainer.update(rollout)
                            state_performance['perf/train/update_seconds'] += time.perf_counter() - update_started
                            if update.stability_guard_rejected:
                                stability_guard_rejections += 1
                                stability_guard_rejections_this_run += 1
                                bucket_stability_guard_rejection_counts[bucket_index] += 1
                                stability_guard_diagnostic = {'last_accepted_update_state': int(accepted_update_states), 'guard_rejected_sampling_attempt': int(sampling_attempts), 'frozen_baseline_executed': True, 'optimizer_rollback_applied': True, 'noise_bundle_identity': list(rollout.noise_bundle_identity or ()), 'post_update_reference_kl': float(update.post_update_reference_kl), 'adapter_relative_drifts': list(update.adapter_relative_drifts), 'trigger_modes': list(update.stability_guard_trigger_modes)}
                                accepted_this_step = False
                            optimizer_steps_for_state = trainer.optimizer_step - optimizer_step_before
                            bucket_optimizer_step_counts[bucket_index] += optimizer_steps_for_state
                            zero_signal_epochs += int(update.zero_signal)
                            bucket_zero_signal_epoch_counts[bucket_index] += int(update.zero_signal)
                            next_update_state = accepted_update_states + int(not update.stability_guard_rejected)
                            rollout_advantages = update.loss.advantages
                            _, optimizer_metrics = _split_loss_metrics_by_step_axis(update.loss.scalar_metrics())
                            optimizer_metrics.update({'gradient_total': float(update.total_gradient_norm), 'zero_signal_epoch': float(update.zero_signal), 'policy/post_update_reference_kl': float(update.post_update_reference_kl), 'policy/adapter_drift_max': float(max(update.adapter_relative_drifts)), 'stability_guard/rejected': float(update.stability_guard_rejected), 'optimizer_step': float(update.optimizer_step), 'accepted_update_state': float(next_update_state)})
                            for name, value in update.gradient_norms.items():
                                optimizer_metrics[f'gradient_pre_clip/{name}'] = float(value)
                            for name, value in update.clipped_gradient_norms.items():
                                optimizer_metrics[f'gradient_post_clip/{name}'] = float(value)
                            for mode, drift in enumerate(update.adapter_relative_drifts):
                                optimizer_metrics[f'policy/mode_{mode}/adapter_drift'] = float(drift)
                            for metric_name, metric_value in optimizer_metrics.items():
                                writer.add_scalar(metric_name, metric_value, next_update_state)
                            if not update.stability_guard_rejected and _write_advantage_vector_summary(writer, rollout_advantages, next_update_state, target_accepted_update_states, config.advantage_vector_log_interval_rollouts, trajectories_per_mode=trainer.config.trajectories_per_mode):
                                advantage_vector_record_count += 1
                            valid = np.asarray(proxy.valid_mode_mask, dtype=np.bool_)
                            rollout_metrics = {'environment_steps': float(environment_steps + 1), 'accepted_update_states': float(next_update_state), 'sampling_attempts': float(sampling_attempts), 'rejected_sampling_attempts': float(rejected_sampling_attempts), 'stability_guard_rejections': float(stability_guard_rejections), 'exhausted_states': float(exhausted_states), 'baseline_execution_steps': float(baseline_execution_steps + 1), 'zero_signal_epochs': float(zero_signal_epochs), 'dynamic_sampling_attempts_for_state': float(attempts_for_state), 'training_bucket_index': float(bucket_index), 'vehicle_reward_mean': accepted_attempt_metrics['train/valid_all/vehicle_reward_mean'], 'same_mode_pretrain_reward_mean': accepted_attempt_metrics['train/valid_all/same_mode_pretrain_reward_mean'], FROZEN_PRETRAIN_VEHICLE_REWARD_TAG: float(accepted_attempt_metrics['train/valid_all/same_mode_pretrain_reward_mean']), 'same_mode_reward_gain_mean': accepted_attempt_metrics['train/valid_all/reward_gain_mean'], 'signal_mode_count': float(signal_mode_mask.sum()), 'signal_vehicle_count': float(np.any(signal_mode_mask, axis=1).sum()), 'vehicle_unsafe_rate': float(np.asarray(proxy.unsafe)[valid].mean()), 'vehicle_collision_rate': float(np.asarray(proxy.collision)[valid].mean()), 'vehicle_out_of_drivable_rate': float(np.asarray(proxy.out_of_drivable)[valid].mean()), **{f'vehicle_{role}/reward_mean': float(np.asarray(proxy.rewards)[role][valid[role]].mean()) for role in range(3)}, **_advantage_scalar_metrics(rollout_advantages)}
                            rollout_metrics.update(accepted_attempt_metrics)
                            for metric_name, metric_value in rollout_metrics.items():
                                writer.add_scalar(metric_name, metric_value, next_update_state)
                            with metrics_path.open('a', encoding='utf-8') as stream:
                                stream.write(json.dumps({'event': 'stability_guard_rejection' if update.stability_guard_rejected else 'update_state', **optimizer_metrics, **rollout_metrics}, sort_keys=True) + '\n')
                            last_metrics.update(rollout_metrics)
                            last_metrics.update(optimizer_metrics)
                        baseline_started = time.perf_counter()
                        try:
                            baseline_step_result, optimization, rule_event, committed_execution_id, committed_plan_actions = execute_cached_frozen_baseline(env=env, rule_maker=rule_maker, condition=condition, scenario=scenario, model_inputs=values, frozen_raw_trajectories=frozen_raw_candidate, frozen_selected_modes=frozen_modes, optimizer=trajectory_optimizer)
                            state_performance['perf/train/baseline_step_seconds'] += time.perf_counter() - baseline_started
                        except TrajectoryOptimizationError:
                            np.savez_compressed(run_dir / 'trajectory_optimizer_failure.npz', raw_trajectories=frozen_raw_candidate, frozen_selected_modes=frozen_modes, coarse_trajectories=values.coarse_trajectories, current_speeds_mps=values.ego_state[:, 0], mode_valid_mask=values.mode_valid_mask, candidate_source=np.asarray(['frozen_stage1_argmax']), environment_steps=np.asarray([environment_steps], dtype=np.int64), episode_steps=np.asarray([episode_step], dtype=np.int64))
                            raise
                        for name, value in rule_event.items():
                            rule_diagnostics[name] += int(value)
                        baseline_execution_steps += 1
                        bucket_baseline_execution_step_counts[bucket_index] += 1
                        for metric_name, metric_value in state_performance.items():
                            performance_totals[metric_name] += float(metric_value)
                            writer.add_scalar(metric_name, float(metric_value), baseline_execution_steps)
                        last_metrics.update(state_performance)
                        pretrain_values = np.asarray(pretrain_score.rewards)[execution_mask]
                        _write_baseline_execution_event(metrics_path, optimizer_step=trainer.optimizer_step, accepted_update_states=accepted_update_states + int(accepted_this_step), rejected_sampling_attempts=rejected_sampling_attempts, stability_guard_rejections=stability_guard_rejections, exhausted_states=exhausted_states, baseline_execution_steps=baseline_execution_steps, environment_steps=environment_steps + 1, bucket_index=bucket_index, scenario=scenario, seed=seed, pretrain_reward_mean=float(pretrain_values.mean()), performance=state_performance)
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
                    if _attempt_budget_is_exhausted(accepted_update_states=accepted_update_states, target_accepted_update_states=target_accepted_update_states, sampling_attempts=sampling_attempts, max_sampling_attempts=max_sampling_attempts):
                        attempt_budget_exhausted = True
                        break
                    if episode_has_ended(terminated, truncated, info):
                        if start_target_step is not None and (not start_accepted):
                            assert start_ready_step is not None
                            assert start_offset is not None
                            assert start_upper_bound is not None
                            last_open_offset = stepped_from - start_ready_step
                            if last_open_offset >= 0:
                                temporary_start_offset_upper_bound = min(last_open_offset, temporary_start_offset_upper_bound if temporary_start_offset_upper_bound is not None else last_open_offset)
                                retryable_start_rejection = True
                            rollout_start_rejected_count += 1
                            _write_rollout_start_event(metrics_path, optimizer_step=trainer.optimizer_step, rollout_group=accepted_update_states, bucket_index=bucket_index, scenario=scenario, seed=seed, bucket_episode=bucket_episode_counts[bucket_index], ready_step=start_ready_step, upper_bound=start_upper_bound, sampled_offset=start_offset, target_step=start_target_step, status='rejected_episode_ended_before_target')
                        break
                    if accepted_update_states > accepted_at_episode_start and (_bucket_visit_is_complete(bucket_index=bucket_index, bucket_sample_counts=bucket_accepted_update_counts, bucket_target_counts=bucket_target_counts, current_visit_progress=current_visit_progress, rollout_groups_per_bucket_visit=config.rollout_groups_per_bucket_visit) or accepted_update_states == target_accepted_update_states or accepted_update_states % config.validation_interval_rollouts == 0):
                        break
            finally:
                env.close()
            if accepted_update_states == accepted_at_episode_start and baseline_execution_steps == baseline_steps_at_episode_start:
                if retryable_start_rejection:
                    consecutive_empty_episodes = 0
                else:
                    consecutive_empty_episodes += 1
                    if consecutive_empty_episodes >= 3:
                        raise OnlineGRPOError(f'three consecutive episodes produced no feasible online state rollout for training bucket {bucket_index}: {scenario[0]}/{scenario[1]} seed={seed}')
            else:
                consecutive_empty_episodes = 0
            if _bucket_visit_is_complete(bucket_index=bucket_index, bucket_sample_counts=bucket_accepted_update_counts, bucket_target_counts=bucket_target_counts, current_visit_progress=current_visit_progress, rollout_groups_per_bucket_visit=config.rollout_groups_per_bucket_visit):
                current_visit_progress = 0
                unfinished_bucket = _next_unfinished_bucket_index(bucket_accepted_update_counts, bucket_target_counts, start_index=(bucket_index + 1) % len(training_buckets))
                next_bucket_index = 0 if unfinished_bucket is None else unfinished_bucket
                temporary_start_offset_upper_bound = None
            else:
                next_bucket_index = bucket_index
            if attempt_budget_exhausted:
                break
            if stability_guard_diagnostic is not None:
                break
            if accepted_update_states != last_validated_update_state and (accepted_update_states == target_accepted_update_states or (accepted_update_states > 0 and accepted_update_states % config.validation_interval_rollouts == 0)):
                validation = _fixed_vehicle_mode_validation(trainer, device=torch_device, reward_config=reward_config, scenarios=config.scenarios, seeds=HOLDOUT_SEEDS, validation_state_bank=validation_state_bank, frozen_cache=fixed_validation_cache)
                validation_calls += 1
                for name, value in validation.items():
                    if name.startswith('perf/') and name.endswith('_seconds'):
                        performance_totals[name] += float(value)
                validation.update(_validation_reward_comparison_metrics(validation, pretrain_validation))
                last_metrics.update(validation)
                for metric_name, metric_value in validation.items():
                    writer.add_scalar(metric_name, metric_value, accepted_update_states)
                last_validated_update_state = accepted_update_states
                current_sampler_state = _sampler_state(accepted_update_states=accepted_update_states, sampling_attempts=sampling_attempts, rejected_sampling_attempts=rejected_sampling_attempts, stability_guard_rejections=stability_guard_rejections, exhausted_states=exhausted_states, baseline_execution_steps=baseline_execution_steps, zero_signal_epochs=zero_signal_epochs, warmup_environment_steps=warmup_environment_steps, bucket_target_counts=bucket_target_counts, bucket_accepted_update_counts=bucket_accepted_update_counts, bucket_sampling_attempt_counts=bucket_sampling_attempt_counts, bucket_rejected_sampling_attempt_counts=bucket_rejected_sampling_attempt_counts, bucket_stability_guard_rejection_counts=bucket_stability_guard_rejection_counts, bucket_exhausted_state_counts=bucket_exhausted_state_counts, bucket_baseline_execution_step_counts=bucket_baseline_execution_step_counts, bucket_zero_signal_epoch_counts=bucket_zero_signal_epoch_counts, bucket_optimizer_step_counts=bucket_optimizer_step_counts, bucket_episode_counts=bucket_episode_counts, next_bucket_index=next_bucket_index, current_visit_progress=current_visit_progress, rollout_start_generator_state=rollout_start_generator.get_state(), last_validated_update_state=last_validated_update_state, rollout_groups_per_bucket_visit=config.rollout_groups_per_bucket_visit, optimizer_step=trainer.optimizer_step, environment_steps=environment_steps, max_sampling_attempts=max_sampling_attempts)
                safety_eligible, selection_score = _append_validation_selection_event(validation_selection_history, accepted_update_state=accepted_update_states, validation=validation, pretrain_validation=pretrain_validation)
                validation['validation/checkpoint_safety_eligible'] = float(safety_eligible)
                if selection_score is not None:
                    validation['validation/vehicle_reward_gain_trailing3'] = selection_score[0]
                    validation['validation/selected_reward_gain_trailing3'] = selection_score[1]
                last_metrics.update(validation)
                for metric_name in ('validation/checkpoint_safety_eligible', 'validation/vehicle_reward_gain_trailing3', 'validation/selected_reward_gain_trailing3'):
                    if metric_name in validation:
                        writer.add_scalar(metric_name, validation[metric_name], accepted_update_states)
                checkpoint_started = time.perf_counter()
                if selection_score is not None:
                    if best_unconstrained_score is None or selection_score > best_unconstrained_score:
                        best_unconstrained_score = selection_score
                        unconstrained_checkpoint = _checkpoint_payload(variant=variant, trainer=trainer, source_sha=source_sha, source_payload=source_payload, metrics=last_metrics, diagnostic_only=run_mode != 'formal', run_mode=run_mode, reward_config=reward_config, scenario_contract_sha=scenario_contract_sha, scenario_seeds=config.scenario_seeds, environment_steps=environment_steps, best_validation_reward=best_reward, best_selected_reward_gain=best_selected_reward_gain, best_checkpoint_sha256=best_checkpoint_sha256, validation_selection_history=validation_selection_history, collection_contract=collection_contract, sampler_state=current_sampler_state)
                        save_grpo_checkpoint(best_unconstrained_path, unconstrained_checkpoint)
                if safety_eligible and selection_score is not None and (selection_score[0] >= 0.0) and (best_reward is None or selection_score > (best_reward, float(best_selected_reward_gain))):
                    best_reward, best_selected_reward_gain = selection_score
                    best_checkpoint = _checkpoint_payload(variant=variant, trainer=trainer, source_sha=source_sha, source_payload=source_payload, metrics=last_metrics, diagnostic_only=run_mode != 'formal', run_mode=run_mode, reward_config=reward_config, scenario_contract_sha=scenario_contract_sha, scenario_seeds=config.scenario_seeds, environment_steps=environment_steps, best_validation_reward=best_reward, best_selected_reward_gain=best_selected_reward_gain, best_checkpoint_sha256=None, validation_selection_history=validation_selection_history, collection_contract=collection_contract, sampler_state=current_sampler_state)
                    save_grpo_checkpoint(best_path, best_checkpoint)
                    save_grpo_checkpoint(best_safe_path, best_checkpoint)
                    best_checkpoint_sha256 = _checkpoint_file_sha256(best_path)
                checkpoint = _checkpoint_payload(variant=variant, trainer=trainer, source_sha=source_sha, source_payload=source_payload, metrics=last_metrics, diagnostic_only=run_mode != 'formal', run_mode=run_mode, reward_config=reward_config, scenario_contract_sha=scenario_contract_sha, scenario_seeds=config.scenario_seeds, environment_steps=environment_steps, best_validation_reward=best_reward, best_selected_reward_gain=best_selected_reward_gain, best_checkpoint_sha256=best_checkpoint_sha256, validation_selection_history=validation_selection_history, collection_contract=collection_contract, sampler_state=current_sampler_state)
                save_grpo_checkpoint(last_path, checkpoint)
                if accepted_update_states in (200, 500):
                    save_grpo_checkpoint(milestone_200_path if accepted_update_states == 200 else milestone_500_path, checkpoint)
                checkpoint_seconds = time.perf_counter() - checkpoint_started
                checkpoint_metric = 'perf/validation/checkpoint_seconds'
                validation[checkpoint_metric] = float(checkpoint_seconds)
                performance_totals[checkpoint_metric] += float(checkpoint_seconds)
                last_metrics[checkpoint_metric] = float(checkpoint_seconds)
                writer.add_scalar(checkpoint_metric, float(checkpoint_seconds), accepted_update_states)
                with metrics_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'event': 'validation_performance', 'accepted_update_state': int(accepted_update_states), **{name: float(value) for name, value in validation.items() if name.startswith('perf/')}}, sort_keys=True) + '\n')
                if accepted_update_states >= 200 and len(validation_selection_history) >= 3:
                    trailing = validation_selection_history[-3:]
                    diagnostic_early_stop = all((value['vehicle_reward_gain'] < 0.0 and value['selected_reward_gain'] < 0.0 and (value['s7_out_delta'] > 0.0) for value in trailing))
    finally:
        writer.close()
    if stability_guard_diagnostic is not None:
        guard_sampler_state = _sampler_state(accepted_update_states=accepted_update_states, sampling_attempts=sampling_attempts, rejected_sampling_attempts=rejected_sampling_attempts, stability_guard_rejections=stability_guard_rejections, exhausted_states=exhausted_states, baseline_execution_steps=baseline_execution_steps, zero_signal_epochs=zero_signal_epochs, warmup_environment_steps=warmup_environment_steps, bucket_target_counts=bucket_target_counts, bucket_accepted_update_counts=bucket_accepted_update_counts, bucket_sampling_attempt_counts=bucket_sampling_attempt_counts, bucket_rejected_sampling_attempt_counts=bucket_rejected_sampling_attempt_counts, bucket_stability_guard_rejection_counts=bucket_stability_guard_rejection_counts, bucket_exhausted_state_counts=bucket_exhausted_state_counts, bucket_baseline_execution_step_counts=bucket_baseline_execution_step_counts, bucket_zero_signal_epoch_counts=bucket_zero_signal_epoch_counts, bucket_optimizer_step_counts=bucket_optimizer_step_counts, bucket_episode_counts=bucket_episode_counts, next_bucket_index=next_bucket_index, current_visit_progress=current_visit_progress, rollout_start_generator_state=rollout_start_generator.get_state(), last_validated_update_state=last_validated_update_state, rollout_groups_per_bucket_visit=config.rollout_groups_per_bucket_visit, optimizer_step=trainer.optimizer_step, environment_steps=environment_steps, max_sampling_attempts=max_sampling_attempts)
        guard_payload = _checkpoint_payload(variant=variant, trainer=trainer, source_sha=source_sha, source_payload=source_payload, metrics=last_metrics, diagnostic_only=True, run_mode=run_mode, reward_config=reward_config, scenario_contract_sha=scenario_contract_sha, scenario_seeds=config.scenario_seeds, environment_steps=environment_steps, best_validation_reward=best_reward, best_selected_reward_gain=best_selected_reward_gain, best_checkpoint_sha256=best_checkpoint_sha256, validation_selection_history=validation_selection_history, collection_contract=collection_contract, sampler_state=guard_sampler_state)
        guard_payload['training_status'] = 'stability_guard_rejected'
        guard_path = save_grpo_checkpoint(run_dir / 'checkpoints' / 'stability_guard.pt', guard_payload)
        guard_record = {'format': 'stage2_grpo_stability_guard_diagnostic_v2', 'training_status': 'stability_guard_rejected', 'stability_guard_rejections': stability_guard_rejections, 'stability_guard_rejections_this_run': stability_guard_rejections_this_run, 'optimizer_steps': trainer.optimizer_step, 'accepted_update_states': accepted_update_states, 'baseline_execution_steps': baseline_execution_steps, 'environment_steps': environment_steps, 'checkpoint': str(guard_path.resolve()), **stability_guard_diagnostic}
        (run_dir / 'stability_guard.json').write_text(json.dumps(guard_record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        report = {'format': 'bev_joint_grpo_online_report_v15', 'training_status': 'stability_guard_rejected', 'diagnostic_only': True, 'eligible_for_formal_training': False, 'accepted_update_states': accepted_update_states, 'target_accepted_update_states': target_accepted_update_states, 'optimizer_steps': trainer.optimizer_step, 'sampling_attempts': sampling_attempts, 'rejected_sampling_attempts': rejected_sampling_attempts, 'exhausted_states': exhausted_states, 'baseline_execution_steps': baseline_execution_steps, 'environment_steps': environment_steps, 'stability_guard_rejections': stability_guard_rejections, 'stability_guard_rejections_this_run': stability_guard_rejections_this_run, 'stability_guard': guard_record, 'last_checkpoint': str(guard_path.resolve()), 'wall_time_seconds': time.monotonic() - started_at, 'performance': _performance_summary(performance_totals, validation_calls=validation_calls)}
        (run_dir / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        return report
    if attempt_budget_exhausted:
        incomplete_sampler_state = _sampler_state(accepted_update_states=accepted_update_states, sampling_attempts=sampling_attempts, rejected_sampling_attempts=rejected_sampling_attempts, stability_guard_rejections=stability_guard_rejections, exhausted_states=exhausted_states, baseline_execution_steps=baseline_execution_steps, zero_signal_epochs=zero_signal_epochs, warmup_environment_steps=warmup_environment_steps, bucket_target_counts=bucket_target_counts, bucket_accepted_update_counts=bucket_accepted_update_counts, bucket_sampling_attempt_counts=bucket_sampling_attempt_counts, bucket_rejected_sampling_attempt_counts=bucket_rejected_sampling_attempt_counts, bucket_stability_guard_rejection_counts=bucket_stability_guard_rejection_counts, bucket_exhausted_state_counts=bucket_exhausted_state_counts, bucket_baseline_execution_step_counts=bucket_baseline_execution_step_counts, bucket_zero_signal_epoch_counts=bucket_zero_signal_epoch_counts, bucket_optimizer_step_counts=bucket_optimizer_step_counts, bucket_episode_counts=bucket_episode_counts, next_bucket_index=next_bucket_index, current_visit_progress=current_visit_progress, rollout_start_generator_state=rollout_start_generator.get_state(), last_validated_update_state=last_validated_update_state, rollout_groups_per_bucket_visit=config.rollout_groups_per_bucket_visit, optimizer_step=trainer.optimizer_step, environment_steps=environment_steps, max_sampling_attempts=max_sampling_attempts)
        incomplete_checkpoint = _checkpoint_payload(variant=variant, trainer=trainer, source_sha=source_sha, source_payload=source_payload, metrics=last_metrics, diagnostic_only=run_mode != 'formal', run_mode=run_mode, reward_config=reward_config, scenario_contract_sha=scenario_contract_sha, scenario_seeds=config.scenario_seeds, environment_steps=environment_steps, best_validation_reward=best_reward, best_selected_reward_gain=best_selected_reward_gain, best_checkpoint_sha256=best_checkpoint_sha256, validation_selection_history=validation_selection_history, collection_contract=collection_contract, sampler_state=incomplete_sampler_state)
        incomplete_checkpoint['training_status'] = 'incomplete_attempt_budget_exhausted'
        last_path = save_grpo_checkpoint(last_path, incomplete_checkpoint)
        incomplete_report = {'format': 'bev_joint_grpo_online_report_v15', 'training_status': 'incomplete_attempt_budget_exhausted', 'variant': variant, 'run_mode': run_mode, 'accepted_update_states': accepted_update_states, 'accepted_update_states_this_run': accepted_update_states - run_start_accepted_update_state, 'target_accepted_update_states': target_accepted_update_states, 'sampling_attempts': sampling_attempts, 'sampling_attempts_this_run': sampling_attempts - run_start_sampling_attempt, 'max_sampling_attempts': max_sampling_attempts, 'rejected_sampling_attempts': rejected_sampling_attempts, 'exhausted_states': exhausted_states, 'baseline_execution_steps': baseline_execution_steps, 'zero_signal_epochs': zero_signal_epochs, 'stability_guard_rejections': stability_guard_rejections, 'stability_guard_rejections_this_run': stability_guard_rejections_this_run, 'warmup_environment_steps': warmup_environment_steps, 'environment_steps': environment_steps, 'optimizer_steps': trainer.optimizer_step, 'wall_time_seconds': time.monotonic() - started_at, 'performance': _performance_summary(performance_totals, validation_calls=validation_calls), 'cuda_peak_memory_bytes': _cuda_peak_memory_bytes(torch_device), 'rollout_collection_contract': collection_contract, 'training_bucket_counters': [{'scenario': scenario[0], 'route': scenario[1], 'seed': seed, 'target_accepted_updates': bucket_target_counts[index], 'accepted_update_states': bucket_accepted_update_counts[index], 'sampling_attempts': bucket_sampling_attempt_counts[index], 'rejected_sampling_attempts': bucket_rejected_sampling_attempt_counts[index], 'stability_guard_rejections': bucket_stability_guard_rejection_counts[index], 'exhausted_states': bucket_exhausted_state_counts[index], 'baseline_execution_steps': bucket_baseline_execution_step_counts[index], 'zero_signal_epochs': bucket_zero_signal_epoch_counts[index], 'environment_episodes': bucket_episode_counts[index], 'optimizer_steps': bucket_optimizer_step_counts[index]} for index, (scenario, seed) in enumerate(training_buckets)], 'last_checkpoint': str(last_path.resolve())}
        incomplete_report_path = run_dir / 'report.json'
        incomplete_report_path.write_text(json.dumps(incomplete_report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        raise OnlineGRPOError(f'dynamic sampling exhausted max_sampling_attempts before reaching target_accepted_update_states; incomplete checkpoint and report saved at {last_path} and {incomplete_report_path}')
    if not validation_selection_history:
        raise OnlineGRPOError('online GRPO never completed fixed validation')
    final_sampler_state = _sampler_state(accepted_update_states=accepted_update_states, sampling_attempts=sampling_attempts, rejected_sampling_attempts=rejected_sampling_attempts, stability_guard_rejections=stability_guard_rejections, exhausted_states=exhausted_states, baseline_execution_steps=baseline_execution_steps, zero_signal_epochs=zero_signal_epochs, warmup_environment_steps=warmup_environment_steps, bucket_target_counts=bucket_target_counts, bucket_accepted_update_counts=bucket_accepted_update_counts, bucket_sampling_attempt_counts=bucket_sampling_attempt_counts, bucket_rejected_sampling_attempt_counts=bucket_rejected_sampling_attempt_counts, bucket_stability_guard_rejection_counts=bucket_stability_guard_rejection_counts, bucket_exhausted_state_counts=bucket_exhausted_state_counts, bucket_baseline_execution_step_counts=bucket_baseline_execution_step_counts, bucket_zero_signal_epoch_counts=bucket_zero_signal_epoch_counts, bucket_optimizer_step_counts=bucket_optimizer_step_counts, bucket_episode_counts=bucket_episode_counts, next_bucket_index=next_bucket_index, current_visit_progress=current_visit_progress, rollout_start_generator_state=rollout_start_generator.get_state(), last_validated_update_state=last_validated_update_state, rollout_groups_per_bucket_visit=config.rollout_groups_per_bucket_visit, optimizer_step=trainer.optimizer_step, environment_steps=environment_steps, max_sampling_attempts=max_sampling_attempts)
    payload = _checkpoint_payload(variant=variant, trainer=trainer, source_sha=source_sha, source_payload=source_payload, metrics=last_metrics, diagnostic_only=run_mode != 'formal', run_mode=run_mode, reward_config=reward_config, scenario_contract_sha=scenario_contract_sha, scenario_seeds=config.scenario_seeds, environment_steps=environment_steps, best_validation_reward=best_reward, best_selected_reward_gain=best_selected_reward_gain, best_checkpoint_sha256=best_checkpoint_sha256, validation_selection_history=validation_selection_history, collection_contract=collection_contract, sampler_state=final_sampler_state)
    last_path = save_grpo_checkpoint(last_path, payload)
    cuda_peak_memory_bytes = _cuda_peak_memory_bytes(torch_device)
    restored, _, _ = _load_trainer(variant, Path(source_checkpoint), torch_device, grpo_config=grpo_config, allow_diagnostic_source=run_mode == 'smoke')
    loaded = checkpoint_loader(last_path, restored, expected_source_stage1_sha256=source_sha)
    _validate_online_checkpoint_metadata(loaded, run_mode=run_mode, reward_config=reward_config, scenario_contract_sha=scenario_contract_sha, scenario_seeds=config.scenario_seeds, collection_contract=collection_contract, bucket_count=len(training_buckets), bucket_target_counts=bucket_target_counts, rollout_groups_per_bucket_visit=config.rollout_groups_per_bucket_visit, optimizer_step=restored.optimizer_step, max_sampling_attempts=max_sampling_attempts)
    plot_paths = generate_grpo_plots(run_dir / 'tb', run_dir / 'plots', require_frozen_pretrain_reward=False, rollout_axis_label=ACCEPTED_ROLLOUT_AXIS_LABEL)
    if rollout_start_accepted_count + rollout_start_rejected_count != len(rollout_start_offsets_this_run):
        raise OnlineGRPOError('rollout start diagnostics contain an unresolved sampling attempt')
    report = {'format': 'bev_joint_grpo_online_report_v15', 'implementation_commit': implementation_commit, 'training_status': 'diagnostic_early_stop' if diagnostic_early_stop else 'complete', 'variant': variant, 'run_mode': run_mode, 'diagnostic_only': run_mode != 'formal', 'eligible_for_formal_training': run_mode == 'formal', 'grpo_config': _grpo_config_artifact_payload(grpo_config), 'optimizer_steps': trainer.optimizer_step, 'optimizer_steps_this_run': trainer.optimizer_step - run_start_optimizer_step, 'environment_steps': environment_steps, 'environment_episode_count': sum(bucket_episode_counts), 'warmup_environment_steps': warmup_environment_steps, 'accepted_update_states': accepted_update_states, 'accepted_update_states_this_run': accepted_update_states - run_start_accepted_update_state, 'target_accepted_update_states': target_accepted_update_states, 'sampling_attempts': sampling_attempts, 'sampling_attempts_this_run': sampling_attempts - run_start_sampling_attempt, 'max_sampling_attempts': max_sampling_attempts, 'rejected_sampling_attempts': rejected_sampling_attempts, 'exhausted_states': exhausted_states, 'baseline_execution_steps': baseline_execution_steps, 'zero_signal_epochs': zero_signal_epochs, 'stability_guard_rejections': stability_guard_rejections, 'stability_guard_rejections_this_run': stability_guard_rejections_this_run, 'rollout_start_diagnostics_this_run': {'attempt_count': len(rollout_start_offsets_this_run), 'accepted_count': rollout_start_accepted_count, 'rejected_count': rollout_start_rejected_count, 'offset_min_steps': min(rollout_start_offsets_this_run) if rollout_start_offsets_this_run else None, 'offset_mean_steps': float(np.mean(rollout_start_offsets_this_run)) if rollout_start_offsets_this_run else None, 'offset_max_steps': max(rollout_start_offsets_this_run) if rollout_start_offsets_this_run else None}, 'wall_time_seconds': time.monotonic() - started_at, 'performance': _performance_summary(performance_totals, validation_calls=validation_calls), 'cuda_peak_memory_bytes': cuda_peak_memory_bytes, 'v2_rule_conditioning': {'enabled': True, **rule_diagnostics}, 'reward_contract_version': _reward_contract_version(), 'reward_contract_sha256': VEHICLE_MODE_REWARD_CONTRACT_SHA256, 'reward_config_sha256': vehicle_mode_reward_config_sha256(reward_config), 'reward_application_contract': GRPO_OPEN_REWARD_APPLICATION_CONTRACT, 'reward_application_contract_sha256': GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256, 'reward_input_domain': 'tau_d', 'training_candidate_domain': 'tau_d_all_vehicle_modes', 'environment_action_source': 'cached_frozen_stage1_argmax', 'execution_input_domain': 'tau_cmd', 'frozen_pretrain_reward_logging': frozen_pretrain_reward_logging, 'best_checkpoint_metric': BEST_CHECKPOINT_METRIC, 'validation_reward_family': VALIDATION_REWARD_FAMILY, 'tracking_expansion_enabled': False, 'calibration_required': False, 'policy_update_contract': joint_grpo_optimizer_contract(), 'policy_update_contract_sha256': joint_grpo_optimizer_contract_sha256(), 'rollout_collection_contract': collection_contract, 'scenario_contract_sha256': scenario_contract_sha, 'training_scenarios': [list(value) for value in config.scenarios], 'training_seeds': [int(value) for value in config.scenario_seeds], 'training_bucket_counters': [{'scenario': scenario[0], 'route': scenario[1], 'seed': seed, 'target_accepted_updates': bucket_target_counts[index], 'accepted_update_states': bucket_accepted_update_counts[index], 'sampling_attempts': bucket_sampling_attempt_counts[index], 'rejected_sampling_attempts': bucket_rejected_sampling_attempt_counts[index], 'stability_guard_rejections': bucket_stability_guard_rejection_counts[index], 'exhausted_states': bucket_exhausted_state_counts[index], 'baseline_execution_steps': bucket_baseline_execution_step_counts[index], 'zero_signal_epochs': bucket_zero_signal_epoch_counts[index], 'environment_episodes': bucket_episode_counts[index], 'optimizer_steps': bucket_optimizer_step_counts[index]} for index, (scenario, seed) in enumerate(training_buckets)], 'validation_seeds': list(HOLDOUT_SEEDS), 'last_checkpoint': str(last_path.resolve()), 'best_checkpoint': str(best_path.resolve()) if best_checkpoint_sha256 is not None else None, 'best_safe_checkpoint': str(best_safe_path.resolve()) if best_checkpoint_sha256 is not None else None, 'best_unconstrained_checkpoint': str(best_unconstrained_path.resolve()) if best_unconstrained_path.exists() else None, 'checkpoint_selection_status': 'eligible' if best_checkpoint_sha256 is not None else 'no_eligible_checkpoint', 'best_validation_reward': best_reward, 'best_selected_reward_gain': best_selected_reward_gain, 'validation_selection_history': validation_selection_history, 'metrics': last_metrics, 'checkpoint_round_trip': True, 'advantage_vector_logging': {'storage': 'tensorboard_tensor', 'tensorboard_tag': ADVANTAGE_VECTOR_TAG, 'tensorboard_dir': str((run_dir / 'tb').resolve()), 'shape': [1, 3, 10, int(trainer.config.trajectories_per_mode)], 'interval_rollout_groups': config.advantage_vector_log_interval_rollouts, 'record_count': advantage_vector_record_count, 'scope': 'current_run_only', 'run_start_optimizer_step': run_start_optimizer_step, 'run_start_accepted_update_state': run_start_accepted_update_state, 'group_axis_semantics': 'vehicle_mode_trajectory_sample', 'heatmap': str(plot_paths['advantage_heatmap'].resolve()), 'x_axis': 'absolute_accepted_update_state'}, 'training_plots': {'reward_curve': str(plot_paths['reward_curve'].resolve()), 'reward_tags': ['vehicle_reward_mean', 'same_mode_pretrain_reward_mean', 'same_mode_reward_gain_mean'], 'validation_reward_curve': str(plot_paths['validation_reward_curve'].resolve()), 'validation_reward_tags': ['validation/vehicle_reward_mean', 'validation/same_mode_pretrain_reward_mean', 'validation/vehicle_reward_gain'], 'reward_domain': 'raw_tau_d', 'grpo_loss_curve': str(plot_paths['grpo_loss_curve'].resolve()), 'grpo_loss_tags': ['loss/total', 'loss/trajectory_pg'], 'kl_loss_curve': str(plot_paths['kl_loss_curve'].resolve()), 'kl_loss_tags': ['loss/reference_kl', 'loss/trajectory_reference_kl'], 'reference_kl_weighted': False, 'policy_stability_curve': str(plot_paths['policy_stability_curve'].resolve()), 'reward_x_axis': 'absolute_accepted_update_state', 'validation_reward_x_axis': 'absolute_accepted_update_state', 'optimizer_x_axis': 'absolute_accepted_update_state'}}
    (run_dir / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return report
