"""Fixed-state validation and frozen-reference comparison."""
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
from .config import AGENT_IDS, OnlineGRPOError
from .contracts import BEST_CHECKPOINT_METRIC, VALIDATION_STATE_BANK_FORMAT
from .contracts import _checkpoint_file_sha256, _validate_raw_reward_config, _validated_selection_history
from .rollout import _diffusion_attempt_generators
from .environment import _condition_online_model_inputs, _finalize_online_rule_action, _new_env, _new_online_rule_maker, _scenario_ready_for_primary_sampling, _scenario_summary, episode_has_ended, execution_mode_valid_mask, model_inputs_to_batch, optimize_selected_model_trajectories, route_following_warmup_actions

@dataclass(frozen=True)
class _FixedValidationFrozenEntry:
    """Immutable frozen-policy work reused within one training process."""
    all_mode_trajectories: np.ndarray
    selected_trajectory: np.ndarray
    selected_mode: np.ndarray
    pretrain_reward: VehicleModePretrainRewardResult
    paired_candidates: np.ndarray
    paired_reward: VehicleModeRewardResult

def build_grpo_validation_state_bank(path: Path, *, scenarios: Sequence[tuple[str, str]]=PRIMARY_S5_S9_SCENARIOS, seeds: Sequence[int]=HOLDOUT_SEEDS) -> dict[str, object]:
    """Capture the fixed S5--S9 validation states and common DDIM noises."""
    if tuple((tuple(value) for value in scenarios)) != tuple(PRIMARY_S5_S9_SCENARIOS):
        raise OnlineGRPOError('validation state bank requires the fixed S5--S9 set')
    if tuple((int(value) for value in seeds)) != tuple(HOLDOUT_SEEDS):
        raise OnlineGRPOError('validation state bank requires seeds 31/47')
    records: list[dict[str, object]] = []
    planner_identity = SimpleNamespace(config=SimpleNamespace(model_version='v2'))
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
                    if builder.history_ready() and _scenario_ready_for_primary_sampling(env):
                        break
                    action = route_following_warmup_actions(env, builder)
                    prefix.append({name: np.array(value, copy=True) for name, value in action.items()})
                    _, _, terminated, truncated, info = env.step(action)
                    if episode_has_ended(terminated, truncated, info):
                        raise OnlineGRPOError('validation state-bank warm-up ended before readiness')
                else:
                    raise OnlineGRPOError('validation state bank never reached a realized state')
                values = builder.build_model_inputs(env)
                condition = _condition_online_model_inputs(rule_maker, env, builder, values, committed_execution_id=None, committed_plan_actions=None)
                values = condition.model_inputs
                scenario_summary = _scenario_summary(env)
                execution_mask = execution_mode_valid_mask(values, optimizer=trajectory_optimizer)
                generator = torch.Generator(device='cpu')
                generator.manual_seed(10000019 + 1009 * int(seed) + scenario_index)
                noise = DDIMNoiseBundle.sample((1, 3, 10, 8, 2), device=torch.device('cpu'), generator=generator)
                records.append({'scenario': tuple(scenario), 'seed': int(seed), 'prefix': prefix, 'model_inputs': {name: np.array(value, copy=True) for name, value in values.as_dict().items()}, 'execution_mode_valid_mask': np.array(execution_mask, copy=True), 'rule_actions': dict(condition.rule_actions), 'scenario_state_contract': {'scenario_random_seed': scenario_summary.get('scenario_random_seed'), 'severity_bucket': scenario_summary.get('severity_bucket'), 'resolved_scenario_parameters': dict(scenario_summary.get('resolved_scenario_parameters', {}))}, 'reference_pose_global': np.array(capture_joint_pose_global(env), copy=True), 'initial_noise': noise.initial_noise.cpu(), 'transition_noises': tuple((value.cpu() for value in noise.transition_noises))})
            finally:
                env.close()
    payload: dict[str, object] = {'format': VALIDATION_STATE_BANK_FORMAT, 'scenarios': [tuple(value) for value in scenarios], 'seeds': [int(value) for value in seeds], 'ddim_path': DEFAULT_DDIM_PATH.as_dict(), 'records': records}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload

def _load_grpo_validation_state_bank(path: Path, *, scenarios: Sequence[tuple[str, str]], seeds: Sequence[int]) -> tuple[dict[tuple[tuple[str, str], int], Mapping[str, object]], str]:
    try:
        payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise OnlineGRPOError(f'unable to load validation state bank: {path}') from exc
    if not isinstance(payload, Mapping):
        raise OnlineGRPOError('validation state bank must be an object')
    if payload.get('format') != VALIDATION_STATE_BANK_FORMAT or payload.get('ddim_path') != DEFAULT_DDIM_PATH.as_dict() or payload.get('scenarios') != [tuple(value) for value in scenarios] or (payload.get('seeds') != [int(value) for value in seeds]):
        raise OnlineGRPOError('validation state bank contract mismatch')
    raw_records = payload.get('records')
    expected_count = len(scenarios) * len(seeds)
    if not isinstance(raw_records, list) or len(raw_records) != expected_count:
        raise OnlineGRPOError('validation state bank record count mismatch')
    records: dict[tuple[tuple[str, str], int], Mapping[str, object]] = {}
    for record in raw_records:
        if not isinstance(record, Mapping):
            raise OnlineGRPOError('validation state bank record is invalid')
        raw_scenario = record.get('scenario')
        if not isinstance(raw_scenario, (list, tuple)) or len(raw_scenario) != 2:
            raise OnlineGRPOError('validation state bank scenario is invalid')
        key = ((str(raw_scenario[0]), str(raw_scenario[1])), int(record['seed']))
        if key in records:
            raise OnlineGRPOError('validation state bank has duplicate records')
        records[key] = record
    expected = {(tuple(scenario), int(seed)) for scenario in scenarios for seed in seeds}
    if set(records) != expected:
        raise OnlineGRPOError('validation state bank scenario/seed matrix mismatch')
    return (records, _checkpoint_file_sha256(Path(path)))

def _replay_fixed_validation_state(record: Mapping[str, object], *, scenario: tuple[str, str], seed: int, device: torch.device, trajectory_optimizer: KinematicTrajectoryOptimizer) -> tuple[object, object, _OnlineRuleCondition, object, np.ndarray, dict[str, Tensor], DDIMNoiseBundle, list[dict[str, np.ndarray]], np.ndarray]:
    env = _new_env(scenario, seed)
    builder = JointBEVSampleBuilder(AGENT_IDS)
    builder.reset()
    rule_maker = _new_online_rule_maker(SimpleNamespace(config=SimpleNamespace(model_version='v2')), env)
    raw_prefix = record.get('prefix')
    if not isinstance(raw_prefix, list):
        env.close()
        raise OnlineGRPOError('validation state bank prefix is invalid')
    prefix: list[dict[str, np.ndarray]] = []
    dt_s = simulator_decision_dt_s(env)
    for step_index, raw_action in enumerate(raw_prefix):
        if not isinstance(raw_action, Mapping):
            env.close()
            raise OnlineGRPOError('validation state bank action is invalid')
        builder.capture_state(env, step_index * dt_s)
        action = {str(name): np.asarray(value).copy() for name, value in raw_action.items()}
        prefix.append(action)
        _, _, terminated, truncated, info = env.step(action)
        if episode_has_ended(terminated, truncated, info):
            env.close()
            raise OnlineGRPOError('fixed validation prefix ended unexpectedly')
    builder.capture_state(env, len(prefix) * dt_s)
    values = builder.build_model_inputs(env)
    condition = _condition_online_model_inputs(rule_maker, env, builder, values, committed_execution_id=None, committed_plan_actions=None)
    values = condition.model_inputs
    stored_inputs = record.get('model_inputs')
    if not isinstance(stored_inputs, Mapping):
        env.close()
        raise OnlineGRPOError('validation state bank inputs are invalid')
    current_inputs = values.as_dict()
    if set(stored_inputs) != set(current_inputs) or any((not np.array_equal(np.asarray(stored_inputs[name]), current_inputs[name]) for name in current_inputs)):
        env.close()
        raise OnlineGRPOError('fixed validation replay changed model inputs')
    stored_rule_actions = record.get('rule_actions')
    if dict(condition.rule_actions) != dict(stored_rule_actions):
        env.close()
        raise OnlineGRPOError('fixed validation replay changed RuleMaker condition')
    summary = _scenario_summary(env)
    replayed_scenario_contract = {'scenario_random_seed': summary.get('scenario_random_seed'), 'severity_bucket': summary.get('severity_bucket'), 'resolved_scenario_parameters': dict(summary.get('resolved_scenario_parameters', {}))}
    if replayed_scenario_contract != record.get('scenario_state_contract'):
        env.close()
        raise OnlineGRPOError('fixed validation replay changed resolved scenario parameters')
    execution_mask = execution_mode_valid_mask(values, optimizer=trajectory_optimizer)
    if not np.array_equal(execution_mask, np.asarray(record.get('execution_mode_valid_mask'))):
        env.close()
        raise OnlineGRPOError('fixed validation replay changed executable modes')
    batch = model_inputs_to_batch(values, device, mode_valid_mask=execution_mask)
    try:
        bundle = DDIMNoiseBundle(initial_noise=record['initial_noise'].to(device=device), transition_noises=tuple((value.to(device=device) for value in record['transition_noises'])))
        bundle.validate(batch['coarse_trajectories'][..., :2].shape, device=device)
        reference_pose = np.asarray(record['reference_pose_global'])
    except (KeyError, AttributeError, DDIMTransitionError) as exc:
        env.close()
        raise OnlineGRPOError('validation state bank noise/pose is invalid') from exc
    return (env, rule_maker, condition, values, execution_mask, batch, bundle, prefix, reference_pose)

@torch.no_grad()
def _fixed_raw_proxy_and_simulator_validation(trainer: object, *, device: torch.device, reward_config: JointRewardConfig, scenarios: Sequence[tuple[str, str]], seeds: Sequence[int], validation_state_bank: Mapping[tuple[tuple[str, str], int], Mapping[str, object]], frozen_cache: dict[tuple[tuple[str, str], int], _FixedValidationFrozenEntry] | None=None) -> tuple[dict[str, float], tuple[dict[str, object], ...]]:
    """Evaluate per-vehicle reward; retain joint/simulator diagnostics only."""
    _validate_raw_reward_config(reward_config)
    planner = trainer.planner
    proxy_backend = JointTrajectoryProxyReward(reward_config)
    vehicle_backend = VehicleModeCounterfactualReward(VehicleModeRewardConfig(trajectories_per_mode=trainer.config.trajectories_per_mode))
    single_vehicle_backend = VehicleModeCounterfactualReward(VehicleModeRewardConfig(trajectories_per_mode=1))
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
                raise OnlineGRPOError('validation state bank is missing a scenario/seed record')
            replay_started = time.perf_counter()
            env, rule_maker, condition, values, execution_mask, batch, noise_bundle, _, reference_pose = _replay_fixed_validation_state(record, scenario=tuple(scenario), seed=int(seed), device=device, trajectory_optimizer=trajectory_optimizer)
            performance['perf/validation/state_replay_seconds'] += time.perf_counter() - replay_started
            try:
                current_inference_started = time.perf_counter()
                output = planner_forward_from_batch(planner, batch, ddim_noise_bundle=noise_bundle)
                raw_candidate = output['selected_trajectory'][0].detach().cpu().numpy()[None].astype(np.float32, copy=False)
                selected_modes = output['selected_mode'][0].detach().cpu().numpy()[None].astype(np.int64, copy=False)
                current_all_modes = output['trajectory_candidates'][0].detach().cpu().numpy().astype(np.float32, copy=False)
                performance['perf/validation/current_inference_seconds'] += time.perf_counter() - current_inference_started
                cached = cache.get(cache_key)
                if cached is None:
                    cache_misses += 1
                    frozen_inference_started = time.perf_counter()
                    frozen = trainer.infer_frozen_pretrain_from_inputs(batch, noise_bundle=noise_bundle)
                    frozen_all_modes = frozen['all_mode_trajectories'][0].detach().cpu().numpy().astype(np.float32, copy=False)
                    frozen_argmax = frozen['selected_trajectory'][0].detach().cpu().numpy().astype(np.float32, copy=False)
                    frozen_selected_modes = frozen['selected_mode'][0].detach().cpu().numpy().astype(np.int64, copy=False)
                    performance['perf/validation/frozen_inference_seconds'] += time.perf_counter() - frozen_inference_started
                    pretrain_reward_started = time.perf_counter()
                    pretrain_local = vehicle_backend.score_pretrain(env, values, frozen_all_modes, frozen_argmax, execution_mask)
                    performance['perf/validation/pretrain_reward_seconds'] += time.perf_counter() - pretrain_reward_started
                else:
                    cache_hits += 1
                    frozen_all_modes = cached.all_mode_trajectories
                    frozen_argmax = cached.selected_trajectory
                    frozen_selected_modes = cached.selected_mode
                    pretrain_local = cached.pretrain_reward
                paired_initial, paired_transition = _diffusion_attempt_generators(device=device, training_seed=10000019 + 1009 * int(seed), live_state_index=scenario_index, retry_index=0)
                paired_sampling_started = time.perf_counter()
                if cached is None:
                    paired_rollout = trainer.sample_groups(batch, generator=paired_initial, transition_generator=paired_transition, noise_bundle_identity=(10000019 + 1009 * int(seed), int(scenario_index), 0))
                    paired_current_candidates = paired_rollout.candidate_trajectories[0].detach().cpu().numpy().astype(np.float32, copy=False)
                    paired_frozen_candidates = paired_rollout.frozen_candidate_trajectories[0].detach().cpu().numpy().astype(np.float32, copy=False)
                else:
                    paired_current_candidates = trainer.sample_current_groups(batch, generator=paired_initial, transition_generator=paired_transition)[0].detach().cpu().numpy().astype(np.float32, copy=False)
                    paired_frozen_candidates = cached.paired_candidates
                performance['perf/validation/paired_sampling_seconds'] += time.perf_counter() - paired_sampling_started
                current_reward_started = time.perf_counter()
                paired_current = vehicle_backend.score_candidates(env, values, paired_current_candidates, frozen_argmax, execution_mask, pretrain_local)
                performance['perf/validation/current_n48_reward_seconds'] += time.perf_counter() - current_reward_started
                if cached is None:
                    frozen_reward_started = time.perf_counter()
                    paired_frozen = vehicle_backend.score_candidates(env, values, paired_frozen_candidates, frozen_argmax, execution_mask, pretrain_local)
                    performance['perf/validation/frozen_n48_reward_seconds'] += time.perf_counter() - frozen_reward_started
                    cached = _FixedValidationFrozenEntry(all_mode_trajectories=np.array(frozen_all_modes, dtype=np.float32, copy=True), selected_trajectory=np.array(frozen_argmax, dtype=np.float32, copy=True), selected_mode=np.array(frozen_selected_modes, dtype=np.int64, copy=True), pretrain_reward=pretrain_local, paired_candidates=np.array(paired_frozen_candidates, dtype=np.float32, copy=True), paired_reward=paired_frozen)
                    cache[cache_key] = cached
                else:
                    paired_frozen = cached.paired_reward
                paired_valid = np.asarray(paired_current.valid_mode_mask, dtype=np.bool_)
                role_current = [float(paired_current.rewards[role][paired_valid[role]].mean()) for role in range(3)]
                role_frozen = [float(paired_frozen.rewards[role][paired_valid[role]].mean()) for role in range(3)]
                paired_n48_reward_means.append(float(np.mean(role_current)))
                paired_n48_frozen_reward_means.append(float(np.mean(role_frozen)))
                paired_n48_gain_means.append(float(np.mean(np.asarray(role_current) - np.asarray(role_frozen))))
                n1_reward_started = time.perf_counter()
                current_local = single_vehicle_backend.score_candidates(env, values, current_all_modes[:, :, None], frozen_argmax, execution_mask, pretrain_local)
                performance['perf/validation/current_n1_reward_seconds'] += time.perf_counter() - n1_reward_started
                valid = np.asarray(current_local.valid_mode_mask, dtype=np.bool_)
                scenario_key = str(scenario[0]).split('_', 1)[0]
                for role in range(3):
                    current_mode = int(selected_modes[0, role])
                    frozen_mode = int(frozen_selected_modes[role])
                    current_selected_reward = float(current_local.rewards[role, current_mode, 0])
                    frozen_selected_reward = float(pretrain_local.rewards[role, frozen_mode])
                    selected_vehicle_rewards.append(current_selected_reward)
                    selected_pretrain_vehicle_rewards.append(frozen_selected_reward)
                    scenario_selected_rewards[scenario_key].append(current_selected_reward)
                    scenario_selected_pretrain_rewards[scenario_key].append(frozen_selected_reward)
                    scenario_road_penalties[scenario_key].append(float(current_local.components['road_penalty'][role, current_mode, 0]))
                    scenario_pretrain_road_penalties[scenario_key].append(float(pretrain_local.components['road_penalty'][role, frozen_mode]))
                    scenario_road_margins[scenario_key].append(float(current_local.components['minimum_road_margin_m'][role, current_mode, 0]))
                    scenario_pretrain_road_margins[scenario_key].append(float(pretrain_local.components['minimum_road_margin_m'][role, frozen_mode]))
                per_mode_reward = np.asarray(current_local.rewards).mean(axis=-1)
                per_mode_unsafe = np.asarray(current_local.unsafe)[..., 0]
                per_mode_collision = np.asarray(current_local.collision)[..., 0]
                per_mode_out = np.asarray(current_local.out_of_drivable)[..., 0]
                context_role_rewards = []
                context_role_pretrain_rewards = []
                for role in range(3):
                    role_valid = valid[role]
                    context_role_rewards.append(float(per_mode_reward[role][role_valid].mean()))
                    context_role_pretrain_rewards.append(float(np.asarray(current_local.pretrain_rewards)[role][role_valid].mean()))
                macro_context_vehicle_rewards.append(float(np.mean(context_role_rewards)))
                macro_context_pretrain_rewards.append(float(np.mean(context_role_pretrain_rewards)))
                vehicle_rewards.extend(per_mode_reward[valid].tolist())
                pretrain_vehicle_rewards.extend(np.asarray(current_local.pretrain_rewards)[valid].tolist())
                vehicle_unsafe_count += int(per_mode_unsafe[valid].sum())
                vehicle_collision_count += int(per_mode_collision[valid].sum())
                vehicle_out_count += int(per_mode_out[valid].sum())
                for role in range(3):
                    role_valid = valid[role]
                    role_rewards[role].extend(per_mode_reward[role][role_valid].tolist())
                    role_unsafe_counts[role] += int(per_mode_unsafe[role][role_valid].sum())
                    role_collision_counts[role] += int(per_mode_collision[role][role_valid].sum())
                    role_out_counts[role] += int(per_mode_out[role][role_valid].sum())
                joint_reward_started = time.perf_counter()
                raw_proxy = proxy_backend.score(env, values, raw_candidate)
                performance['perf/validation/joint_reward_seconds'] += time.perf_counter() - joint_reward_started
                raw_rewards.append(float(raw_proxy.rewards[0]))
                raw_unsafe_count += int(raw_proxy.unsafe[0])
                raw_collision_count += int(raw_proxy.collision[0])
                raw_out_count += int(raw_proxy.out_of_drivable[0])
                validation_optimization = optimize_selected_model_trajectories(values, raw_candidate, selected_modes, optimizer=trajectory_optimizer)
                validation_optimization, _, _, _ = _finalize_online_rule_action(rule_maker, condition, env=env, scenario=tuple(scenario), values=values, selected_modes=selected_modes[0], optimization=validation_optimization, optimizer=trajectory_optimizer)
                command_candidate = validation_optimization.optimized_trajectories
                spec = JointEpisodeSpec(scenario_id=str(scenario[0]), local_route=str(scenario[1]), seed=int(seed), reference_pose_global=reference_pose)
                simulator_started = time.perf_counter()
                try:
                    branch = evaluator.evaluate_from_replayed_env(spec, env, command_candidate)
                except Exception as exc:
                    simulator_errors.append({'scenario': str(scenario[0]), 'route': str(scenario[1]), 'seed': int(seed), 'error_type': type(exc).__name__, 'error': str(exc)})
                    branch = None
                finally:
                    performance['perf/validation/simulator_seconds'] += time.perf_counter() - simulator_started
            finally:
                env.close()
            if branch is None:
                continue
            simulator_rewards.append(float(branch.reward.rewards[0]))
            scenario_key = str(scenario[0]).split('_', 1)[0]
            scenario_simulator_rewards[scenario_key].append(float(branch.reward.rewards[0]))
            simulator_unsafe_count += int(branch.reward.unsafe[0])
            simulator_collision_count += int(branch.reward.collision[0])
            simulator_out_count += int(branch.reward.out_of_drivable[0])
            scenario_simulator_collisions[scenario_key] += int(branch.reward.collision[0])
            scenario_simulator_outs[scenario_key] += int(branch.reward.out_of_drivable[0])
    if not raw_rewards or not vehicle_rewards:
        raise OnlineGRPOError('fixed validation produced no raw proxy rewards')
    metrics = {'validation/vehicle_reward_mean': float(np.mean(macro_context_vehicle_rewards)), 'validation/vehicle_reward_flat_mean': float(np.mean(vehicle_rewards)), 'validation/same_mode_pretrain_reward_mean': float(np.mean(macro_context_pretrain_rewards)), 'validation/vehicle_reward_gain': float(np.mean(macro_context_vehicle_rewards) - np.mean(macro_context_pretrain_rewards)), 'validation/paired_n48_vehicle_reward_mean': float(np.mean(paired_n48_reward_means)), 'validation/paired_n48_frozen_reward_mean': float(np.mean(paired_n48_frozen_reward_means)), 'validation/paired_n48_reward_gain': float(np.mean(paired_n48_gain_means)), 'validation/vehicle_unsafe_count': float(vehicle_unsafe_count), 'validation/vehicle_collision_count': float(vehicle_collision_count), 'validation/vehicle_out_of_drivable_count': float(vehicle_out_count), 'validation/raw_proxy_reward_mean': float(np.mean(raw_rewards)), 'validation/raw_proxy_unsafe_count': float(raw_unsafe_count), 'validation/raw_proxy_collision_count': float(raw_collision_count), 'validation/raw_proxy_out_of_road_count': float(raw_out_count), 'validation/simulator_available': float(not simulator_errors), 'validation/simulator_success_count': float(len(simulator_rewards)), 'validation/simulator_failure_count': float(len(simulator_errors)), 'validation/selected_vehicle_reward_mean': float(np.mean(selected_vehicle_rewards)), 'validation/selected_pretrain_vehicle_reward_mean': float(np.mean(selected_pretrain_vehicle_rewards)), 'validation/selected_vehicle_reward_gain': float(np.mean(selected_vehicle_rewards) - np.mean(selected_pretrain_vehicle_rewards))}
    for scenario_key in sorted(scenario_selected_rewards):
        current_selected = scenario_selected_rewards[scenario_key]
        frozen_selected = scenario_selected_pretrain_rewards[scenario_key]
        metrics.update({f'validation/{scenario_key}/selected_vehicle_reward_mean': float(np.mean(current_selected)), f'validation/{scenario_key}/selected_vehicle_reward_gain': float(np.mean(current_selected) - np.mean(frozen_selected)), f'validation/{scenario_key}/road_penalty_mean': float(np.mean(scenario_road_penalties[scenario_key])), f'validation/{scenario_key}/pretrain_road_penalty_mean': float(np.mean(scenario_pretrain_road_penalties[scenario_key])), f'validation/{scenario_key}/minimum_road_margin_m': float(np.min(scenario_road_margins[scenario_key])), f'validation/{scenario_key}/pretrain_minimum_road_margin_m': float(np.min(scenario_pretrain_road_margins[scenario_key]))})
    for role in range(3):
        if not role_rewards[role]:
            raise OnlineGRPOError(f'fixed validation produced no valid modes for vehicle {role}')
        metrics.update({f'validation/vehicle_{role}_reward_mean': float(np.mean(role_rewards[role])), f'validation/vehicle_{role}_unsafe_count': float(role_unsafe_counts[role]), f'validation/vehicle_{role}_collision_count': float(role_collision_counts[role]), f'validation/vehicle_{role}_out_of_drivable_count': float(role_out_counts[role])})
    if simulator_rewards:
        metrics.update({'validation/simulator_reward_mean': float(np.mean(simulator_rewards)), 'validation/unsafe_count': float(simulator_unsafe_count), 'validation/collision_count': float(simulator_collision_count), 'validation/out_of_road_count': float(simulator_out_count)})
        for scenario_key in sorted(scenario_simulator_rewards):
            metrics.update({f'validation/{scenario_key}/simulator_reward_mean': float(np.mean(scenario_simulator_rewards[scenario_key])), f'validation/{scenario_key}/simulator_collision_count': float(scenario_simulator_collisions[scenario_key]), f'validation/{scenario_key}/simulator_out_count': float(scenario_simulator_outs[scenario_key])})
    performance['perf/validation/total_seconds'] = time.perf_counter() - validation_started
    metrics.update(performance)
    metrics['perf/validation/frozen_cache_hits'] = float(cache_hits)
    metrics['perf/validation/frozen_cache_misses'] = float(cache_misses)
    if not all((math.isfinite(value) for value in metrics.values())):
        raise OnlineGRPOError('fixed validation metrics must be finite')
    return (metrics, tuple(simulator_errors))

def _validation_reward_comparison_metrics(current_validation: Mapping[str, object], pretrain_validation: Mapping[str, object]) -> dict[str, float]:
    reward_tag = 'validation/vehicle_reward_mean'
    if reward_tag not in current_validation:
        raise OnlineGRPOError(f'fixed validation is missing required metric {reward_tag}')
    try:
        current_reward = float(current_validation[reward_tag])
        baseline_reward = float(pretrain_validation[reward_tag])
    except (TypeError, ValueError) as exc:
        raise OnlineGRPOError('validation current and pretrain rewards must be finite scalars') from exc
    if not math.isfinite(current_reward) or not math.isfinite(baseline_reward):
        raise OnlineGRPOError('validation current and pretrain rewards must be finite scalars')
    metrics = {'validation/fixed_pretrain_vehicle_reward': baseline_reward, 'validation/fixed_pretrain_vehicle_reward_gain': current_reward - baseline_reward}
    comparison_tags = ('selected_vehicle_reward_mean', 'simulator_reward_mean', 'collision_count', 'out_of_road_count', 'S7/simulator_collision_count', 'S7/simulator_out_count', 'S7/road_penalty_mean', 'S7/minimum_road_margin_m')
    for suffix in comparison_tags:
        tag = f'validation/{suffix}'
        if tag not in current_validation or tag not in pretrain_validation:
            continue
        current_value = float(current_validation[tag])
        frozen_value = float(pretrain_validation[tag])
        if not math.isfinite(current_value) or not math.isfinite(frozen_value):
            raise OnlineGRPOError(f'validation comparison metric {tag} must be finite')
        metrics[f'validation/fixed_pretrain/{suffix}'] = frozen_value
        metrics[f'validation/gain/{suffix}'] = current_value - frozen_value
    aliases = {'simulator_reward_mean': 'validation/simulator_reward_gain', 'selected_vehicle_reward_mean': 'validation/selected_reward_gain', 'S7/simulator_out_count': 'validation/S7/out_delta', 'S7/simulator_collision_count': 'validation/S7/collision_delta', 'S7/road_penalty_mean': 'validation/S7/road_penalty_delta', 'S7/minimum_road_margin_m': 'validation/S7/road_margin_delta'}
    for suffix, alias in aliases.items():
        source = f'validation/gain/{suffix}'
        if source in metrics:
            metrics[alias] = metrics[source]
    return metrics

def _validation_vehicle_reward(validation: Mapping[str, object]) -> float:
    """Return the sole best-checkpoint objective as a finite scalar."""
    reward_tag = 'validation/vehicle_reward_mean'
    if reward_tag not in validation:
        raise OnlineGRPOError(f'fixed validation is missing required metric {reward_tag}')
    try:
        reward = float(validation[reward_tag])
    except (TypeError, ValueError) as exc:
        raise OnlineGRPOError('validation vehicle reward must be a finite scalar') from exc
    if not math.isfinite(reward):
        raise OnlineGRPOError('validation vehicle reward must be a finite scalar')
    return reward

def _validation_is_safety_eligible(validation: Mapping[str, object], pretrain_validation: Mapping[str, object]) -> bool:
    """Apply the frozen-baseline safety constraints for checkpoint selection."""
    if float(validation.get('validation/simulator_available', 0.0)) != 1.0:
        return False
    constrained_tags = ('validation/collision_count', 'validation/out_of_road_count', 'validation/S7/simulator_out_count')
    for tag in constrained_tags:
        if tag not in validation or tag not in pretrain_validation:
            return False
        current = float(validation[tag])
        frozen = float(pretrain_validation[tag])
        if not math.isfinite(current) or not math.isfinite(frozen) or current > frozen:
            return False
    return True

def _resume_best_checkpoint_anchor(resume_checkpoint: Path, resume_payload: Mapping[str, object]) -> tuple[Path | None, tuple[float, float] | None, list[dict[str, float]]]:
    """Resolve the safety-constrained best checkpoint and selection history."""
    history = _validated_selection_history(resume_payload.get('validation_selection_history'))
    raw_best_reward = resume_payload.get('best_validation_reward')
    raw_best_selected = resume_payload.get('best_selected_reward_gain')
    best_sha = resume_payload.get('best_checkpoint_sha256')
    if raw_best_reward is None:
        if raw_best_selected is not None or best_sha is not None:
            raise OnlineGRPOError('resume checkpoint absent best score conflicts')
        return (None, None, history)
    best_score = (float(raw_best_reward), float(raw_best_selected))
    if best_sha is None:
        return (Path(resume_checkpoint), best_score, history)
    best_path = Path(resume_checkpoint).with_name('best.pt')
    if _checkpoint_file_sha256(best_path) != best_sha:
        raise OnlineGRPOError('resume best checkpoint SHA256 mismatch')
    try:
        best_payload = torch.load(best_path, map_location='cpu', weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise OnlineGRPOError(f'unable to load resume best checkpoint: {best_path}') from exc
    if not isinstance(best_payload, Mapping):
        raise OnlineGRPOError('resume best checkpoint must be a mapping')
    binding_fields = ('schema_version', 'format', 'variant', 'predecessor_condition', 'grpo_config', 'source_stage1_sha256', 'run_mode', 'reward_contract_version', 'reward_contract_sha256', 'reward_config_sha256', 'reward_application_contract_sha256', 'reward_input_domain', 'training_candidate_domain', 'environment_action_source', 'execution_input_domain', 'best_checkpoint_metric', 'joint_reward_role', 'tracking_expansion_enabled', 'calibration_required', 'policy_update_contract', 'policy_update_contract_sha256', 'rollout_collection_contract', 'scenario_contract_sha256', 'trajectory_optimizer_sha256')
    if any((best_payload.get(field) != resume_payload.get(field) for field in binding_fields)):
        raise OnlineGRPOError('resume best checkpoint contract binding mismatch')
    raw_best_reward = best_payload.get('best_validation_reward')
    raw_best_selected = best_payload.get('best_selected_reward_gain')
    if isinstance(raw_best_reward, bool) or not isinstance(raw_best_reward, (int, float)) or (not math.isfinite(float(raw_best_reward))) or isinstance(raw_best_selected, bool) or (not isinstance(raw_best_selected, (int, float))) or (not math.isfinite(float(raw_best_selected))):
        raise OnlineGRPOError('resume best checkpoint score binding mismatch')
    if best_payload.get('best_checkpoint_sha256') is not None or (float(raw_best_reward), float(raw_best_selected)) != best_score:
        raise OnlineGRPOError('resume best checkpoint score binding mismatch')
    return (best_path, best_score, history)
