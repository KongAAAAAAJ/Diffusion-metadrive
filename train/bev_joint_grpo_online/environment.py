"""Simulator interaction, rule conditioning, and trajectory execution."""
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
from .config import AGENT_IDS, JointGRPOOnlineConfig, OnlineGRPOError

def _device(name: str) -> torch.device:
    if name == 'cuda' and (not torch.cuda.is_available()):
        raise OnlineGRPOError('CUDA was requested but is unavailable')
    return torch.device(name)

def _reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

def _cuda_peak_memory_bytes(device: torch.device) -> int | None:
    if device.type != 'cuda':
        return None
    return int(torch.cuda.max_memory_allocated(device))

def model_inputs_to_batch(values: object, device: torch.device, *, mode_valid_mask: np.ndarray | None=None) -> dict[str, torch.Tensor]:
    if not hasattr(values, 'as_dict'):
        raise OnlineGRPOError('online model inputs must expose as_dict()')
    result = {}
    for name, value in values.as_dict().items():
        array = np.asarray(mode_valid_mask if name == 'mode_valid_mask' and mode_valid_mask is not None else value)
        if name == 'mode_valid_mask' and (array.shape != (3, 10) or array.dtype != np.bool_):
            raise OnlineGRPOError('execution mode_valid_mask must be bool [3,10]')
        result[name] = torch.from_numpy(np.array(array, copy=True)).unsqueeze(0).to(device)
    return result

def execution_mode_valid_mask(model_inputs: object, *, optimizer: KinematicTrajectoryOptimizer | None=None) -> np.ndarray:
    """Build the calibrated execution subset of the physical hard mask."""
    for name in ('coarse_trajectories', 'ego_state', 'mode_valid_mask'):
        if not hasattr(model_inputs, name):
            raise OnlineGRPOError(f'online model inputs are missing {name}')
    transform = optimizer or KinematicTrajectoryOptimizer()
    return transform.execution_mode_valid_mask(np.asarray(model_inputs.coarse_trajectories), np.asarray(model_inputs.ego_state)[:, 0], np.asarray(model_inputs.mode_valid_mask))

def constant_velocity_actions(env: object) -> dict[str, np.ndarray]:
    actions = {}
    for agent_id in AGENT_IDS:
        vehicle = env.agents[agent_id]
        speed = max(0.0, float(getattr(vehicle, 'speed_km_h', 0.0)) / 3.6)
        trajectory = np.zeros((8, 3), dtype=np.float32)
        trajectory[:, 0] = speed * np.arange(1, 9, dtype=np.float32) * 0.5
        actions[agent_id] = trajectory
    return actions

def route_following_warmup_actions(env: object, builder: JointBEVSampleBuilder) -> dict[str, np.ndarray]:
    """Label-free route-following warm-up for curved S5--S9 approaches."""
    if not builder.history_ready():
        return constant_velocity_actions(env)
    values = builder.build_model_inputs(env)
    actions = {}
    for role, agent_id in enumerate(AGENT_IDS):
        valid = np.asarray(values.mode_valid_mask[role], dtype=np.bool_)
        preferred = [int(ModeIndex.KEEP_HIGH), int(ModeIndex.KEEP_MEDIUM), int(ModeIndex.KEEP_LOW)]
        mode = next((index for index in preferred if valid[index]), None)
        if mode is None:
            raise OnlineGRPOError(f'{agent_id} has no valid KEEP anchor during route warm-up')
        actions[agent_id] = np.array(values.coarse_trajectories[role, mode], dtype=np.float32, copy=True)
    return actions

def joint_trajectory_action(trajectories: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(trajectories)
    if value.shape != (3, 8, 3) or not np.isfinite(value).all():
        raise OnlineGRPOError('selected online action must be finite [3,8,3]')
    return {agent_id: np.array(value[role], dtype=np.float32, copy=True, order='C') for role, agent_id in enumerate(AGENT_IDS)}

def optimize_selected_model_trajectories(model_inputs: object, raw_trajectories: np.ndarray, selected_modes: np.ndarray, *, optimizer: KinematicTrajectoryOptimizer | None=None) -> TrajectoryOptimizationResult:
    """Apply the execution transform after policy sampling.

    The caller retains ``raw_trajectories`` in the GRPO rollout, so DDIM
    replay/log-prob remains defined on the unmodified diffusion action.
    """
    for name in ('coarse_trajectories', 'ego_state'):
        if not hasattr(model_inputs, name):
            raise OnlineGRPOError(f'online model inputs are missing {name}')
    transform = optimizer or KinematicTrajectoryOptimizer()
    return transform.optimize(raw_trajectories, np.asarray(model_inputs.coarse_trajectories), np.asarray(model_inputs.ego_state)[:, 0], selected_modes)

def optimize_safe_stop_trajectories(model_inputs: object, *, optimizer: KinematicTrajectoryOptimizer | None=None) -> TrajectoryOptimizationResult:
    """Project the always-valid fixed STOP anchors for fail-closed execution."""
    if not hasattr(model_inputs, 'coarse_trajectories'):
        raise OnlineGRPOError('online model inputs are missing coarse_trajectories')
    coarse = np.asarray(model_inputs.coarse_trajectories)
    raw_stop = np.asarray(coarse[:, int(ModeIndex.STOP)], dtype=np.float32)
    return optimize_selected_model_trajectories(model_inputs, raw_stop, np.full((3,), int(ModeIndex.STOP), dtype=np.int64), optimizer=optimizer)

def episode_has_ended(terminated: Mapping[str, object], truncated: Mapping[str, object], info: Mapping[str, object]) -> bool:
    if bool(terminated.get('__all__', False)) or bool(truncated.get('__all__', False)):
        return True
    for agent_id in AGENT_IDS:
        value = info.get(agent_id)
        if not isinstance(value, Mapping):
            continue
        if any((bool(value.get(name, False)) for name in ('crash', 'crash_vehicle', 'crash_object', 'crash_building', 'crash_human', 'out_of_road', 'out_of_route'))):
            return True
    return False

def _new_env(scenario: tuple[str, str], seed: int) -> object:
    env = SensorlessJointBEVPlatoonEnv({'num_agents': 3, 'traffic_density': 0.0, 'initial_speed_km_h': deterministic_initial_speed_km_h(scenario[0], seed), 'start_seed': int(seed), 'num_scenarios': 1, 'scenario_id': str(scenario[0]), 'local_route': str(scenario[1])})
    spawn_manager = getattr(getattr(env, 'engine', None), 'spawn_manager', None)
    set_spawn_seed = getattr(spawn_manager, 'set_episode_spawn_seed', None)
    if callable(set_spawn_seed):
        set_spawn_seed(int(seed))
    env.reset(seed=int(seed))
    return env

def _new_online_rule_maker(planner: object, env: object) -> object:
    planner_config = getattr(planner, 'config', None)
    if getattr(planner_config, 'model_version', None) != 'v2':
        raise OnlineGRPOError('online GRPO requires a v2 planner')
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

def _condition_online_model_inputs(rule_maker: object, env: object, builder: JointBEVSampleBuilder, values: object, *, committed_execution_id: int | None, committed_plan_actions: Mapping[str, int] | None) -> _OnlineRuleCondition:
    proposal_batch = None
    rule_condition: dict[str, int] | None = None
    rule_condition_is_commitment = False
    committed_actions = None if committed_plan_actions is None else {agent_id: int(committed_plan_actions[agent_id]) for agent_id in AGENT_IDS}
    try:
        hard_modes = hard_valid_modes_by_rule_action(AGENT_IDS, np.asarray(values.mode_valid_mask))
        if rule_maker.has_active_lane_change_commitments:
            if committed_execution_id is None or committed_actions is None:
                raise OnlineGRPOError('online RuleMaker commitment has no execution state')
            rule_maker.advance_committed_execution(env, list(AGENT_IDS), committed_execution_id)
            if rule_maker.has_active_lane_change_commitments:
                rule_condition = {agent_id: int(action) for agent_id, action in rule_maker.committed_execution_rule_actions(env, committed_actions).items()}
                rule_condition_is_commitment = True
            else:
                committed_execution_id = None
                committed_actions = None
        if rule_condition is None:
            proposal_batch = rule_maker.propose_joint_actions(env, list(AGENT_IDS), getattr(env, '_last_planner_batch', None) or {}, hard_valid_modes_by_action=hard_modes)
            rule_condition = joint_proposal_actions(proposal_batch.proposals[0], AGENT_IDS) if proposal_batch.proposals else {agent_id: 0 for agent_id in AGENT_IDS}
    except LaneChangeCommitmentError as exc:
        raise OnlineGRPOError(f'online RuleMaker proposal failed: {exc}') from exc
    conditioned = builder.augment_v2_model_inputs(env, values, rule_action_condition=rule_condition, rule_formation_state=rule_maker.is_formation_locked)
    return _OnlineRuleCondition(model_inputs=conditioned, proposal_batch=proposal_batch, rule_actions={agent_id: int(rule_condition[agent_id]) for agent_id in AGENT_IDS}, is_commitment=rule_condition_is_commitment, committed_execution_id=committed_execution_id, committed_plan_actions=committed_actions)

def _validate_online_trajectory_controls(env: object, trajectories: np.ndarray) -> None:
    action = joint_trajectory_action(trajectories)
    for agent_id in AGENT_IDS:
        try:
            control = np.asarray(env.trajectory_to_control(agent_id, action[agent_id]))
        except Exception as exc:
            raise OnlineGRPOError(f'trajectory control failed for {agent_id}: {exc}') from exc
        if not np.isfinite(control).all():
            raise OnlineGRPOError(f'trajectory control is non-finite for {agent_id}')

def _finalize_online_rule_action(rule_maker: object, condition: _OnlineRuleCondition, *, env: object, scenario: tuple[str, str], values: object, selected_modes: np.ndarray, optimization: TrajectoryOptimizationResult, optimizer: KinematicTrajectoryOptimizer) -> tuple[TrajectoryOptimizationResult, dict[str, int], int | None, dict[str, int] | None]:
    diagnostics = {'conditioned_rollouts': 0, 'proposal_match_attempts': 0, 'proposal_matches': 0, 'condition_failures': 0, 'forced_safe_stops': 0, 's7_feedback_exception_hits': 0, 'commitment_conditioned_rollouts': 0, 'commitment_feedback_incompatible': 0}
    diagnostics['conditioned_rollouts'] = 1
    diagnostics['proposal_match_attempts'] = int(not condition.is_commitment)
    try:
        _, feedback_actions, s7_exceptions = diffusion_mode_feedback_actions(np.asarray(selected_modes, dtype=np.int64).tolist(), AGENT_IDS, scenario_id=scenario[0], local_route=scenario[1])
        if condition.is_commitment:
            diagnostics['commitment_conditioned_rollouts'] = 1
            compatible = all((feedback_actions[agent_id] == int(condition.rule_actions[agent_id]) for agent_id in AGENT_IDS))
            matched = None
        else:
            compatible = True
            matched = None if condition.proposal_batch is None else match_joint_action_proposal(condition.proposal_batch, feedback_actions, AGENT_IDS)
    except LaneChangeCommitmentError as exc:
        raise OnlineGRPOError(f'online RuleMaker action matching failed: {exc}') from exc
    diagnostics['s7_feedback_exception_hits'] = sum((int(value) for value in s7_exceptions.values()))
    if not compatible or (not condition.is_commitment and matched is None):
        diagnostics['condition_failures'] = 1
        diagnostics['forced_safe_stops'] = 1
        diagnostics['commitment_feedback_incompatible'] = int(condition.is_commitment and (not compatible))
        coarse = np.asarray(values.coarse_trajectories)
        raw_stop = np.asarray(coarse[:, int(ModeIndex.STOP)], dtype=np.float32)[None]
        resolved = optimize_selected_model_trajectories(values, raw_stop, np.full((1, 3), int(ModeIndex.STOP), dtype=np.int64), optimizer=optimizer)
    else:
        diagnostics['proposal_matches'] = int(not condition.is_commitment)
        resolved = optimization
    _validate_online_trajectory_controls(env, resolved.optimized_trajectories[0])
    committed_execution_id = condition.committed_execution_id
    committed_plan_actions = condition.committed_plan_actions
    if matched is not None:
        if condition.proposal_batch is None:
            raise OnlineGRPOError('matched RuleMaker action has no proposal batch')
        try:
            rule_maker.accept_joint_action(condition.proposal_batch.batch_id, matched.proposal_id)
        except LaneChangeCommitmentError as exc:
            raise OnlineGRPOError(f'online RuleMaker proposal acceptance failed: {exc}') from exc
        if rule_maker.has_active_lane_change_commitments:
            committed_execution_id = int(condition.proposal_batch.batch_id)
            committed_plan_actions = joint_proposal_actions(matched, AGENT_IDS)
    return (resolved, diagnostics, committed_execution_id, committed_plan_actions)

def execute_cached_frozen_baseline(*, env: object, rule_maker: object, condition: _OnlineRuleCondition, scenario: tuple[str, str], model_inputs: object, frozen_raw_trajectories: np.ndarray, frozen_selected_modes: np.ndarray, optimizer: KinematicTrajectoryOptimizer) -> tuple[tuple[object, object, object, object, object], TrajectoryOptimizationResult, dict[str, int], int | None, dict[str, int] | None]:
    """Execute one cached frozen Stage-1 action and advance the env once.

    The sampled GRPO candidates are deliberately absent from this interface,
    making the online execution boundary auditable and hard to misuse.
    """
    raw = np.asarray(frozen_raw_trajectories)
    modes = np.asarray(frozen_selected_modes)
    if raw.shape != (1, 3, 8, 3) or modes.shape != (1, 3):
        raise OnlineGRPOError('cached frozen baseline must be [1,3,8,3] with modes [1,3]')
    optimization = optimize_selected_model_trajectories(model_inputs, raw, modes, optimizer=optimizer)
    optimization, diagnostics, committed_execution_id, committed_plan_actions = _finalize_online_rule_action(rule_maker, condition, env=env, scenario=scenario, values=model_inputs, selected_modes=modes[0], optimization=optimization, optimizer=optimizer)
    action = joint_trajectory_action(optimization.optimized_trajectories[0])
    step_result = env.step(action)
    if not isinstance(step_result, tuple) or len(step_result) != 5:
        raise OnlineGRPOError('online env.step must return a five-item tuple')
    return (step_result, optimization, diagnostics, committed_execution_id, committed_plan_actions)

def _scenario_summary(env: object) -> dict[str, object]:
    orchestrator = getattr(env, '_scenario_orchestrator', None)
    getter = getattr(orchestrator, 'get_episode_summary', None)
    if not callable(getter):
        raise OnlineGRPOError('environment has no scenario realization summary')
    value = getter()
    if not isinstance(value, Mapping):
        raise OnlineGRPOError('scenario realization summary is invalid')
    return dict(value)

def _scenario_ready_from_summary(summary: Mapping[str, object]) -> bool:
    """Return whether one already-captured scenario summary is trainable."""
    if not bool(summary.get('scenario_realized', False)):
        return False
    scenario_id = str(summary.get('scenario_id', ''))
    if scenario_id == 'S5_hard_brake_lead':
        notes = summary.get('scenario_notes', ())
        return bool(summary.get('scenario_triggered', False)) and (isinstance(notes, (list, tuple)) and 'lead_brake_profile' in notes)
    if scenario_id in {'S6_background_merge_in', 'S8_ego_exit_to_ramp'}:
        return bool(summary.get('scenario_recipes_complete', False))
    return True

def _scenario_ready_for_primary_sampling(env: object) -> bool:
    return _scenario_ready_from_summary(_scenario_summary(env))

def _scenario_sampling_window_closed_from_summary(summary: Mapping[str, object]) -> bool:
    """Return the scenario-specific, sticky end of the training window."""
    scenario_id = str(summary.get('scenario_id', ''))
    conflict = summary.get('conflict_evidence', {})
    route = summary.get('route_completion', {})
    if not isinstance(conflict, Mapping) or not isinstance(route, Mapping):
        raise OnlineGRPOError('scenario realization summary evidence is invalid')
    if scenario_id == 'S5_hard_brake_lead':
        return bool(conflict.get('formation_recovered_after_hazard', False))
    if scenario_id == 'S6_background_merge_in':
        return bool(conflict.get('formation_recovered_after_merge', False))
    if scenario_id == 'S7_ego_merge_from_ramp':
        return bool(route.get('all_agents_entered_mainline', False)) and bool(conflict.get('formation_recovered_after_merge', False))
    if scenario_id == 'S8_ego_exit_to_ramp':
        return bool(route.get('all_agents_continued_on_exit_ramp', False)) and bool(conflict.get('formation_recovered_on_ramp', False))
    if scenario_id == 'S9_narrow_channel_negotiation':
        return bool(route.get('all_agents_returned_to_original_lane', False)) and bool(conflict.get('formation_recovered_after_return', False))
    raise OnlineGRPOError(f'unsupported primary scenario sampling window: {scenario_id}')

def _scenario_sampling_window_closed(env: object) -> bool:
    return _scenario_sampling_window_closed_from_summary(_scenario_summary(env))

def _rollout_start_offset_upper_bound(config: JointGRPOOnlineConfig, ready_step: int, temporary_max_offset_steps: int | None=None) -> int:
    """Return the inclusive feasible offset upper bound for one episode."""
    if isinstance(ready_step, bool) or not isinstance(ready_step, int) or ready_step < 0:
        raise OnlineGRPOError('rollout ready step must be a non-negative integer')
    if temporary_max_offset_steps is not None and (isinstance(temporary_max_offset_steps, bool) or not isinstance(temporary_max_offset_steps, int) or temporary_max_offset_steps < 0):
        raise OnlineGRPOError('temporary rollout start offset bound must be a non-negative integer')
    remaining_upper = config.environment_steps_per_episode - ready_step - config.rollout_start_min_remaining_steps
    if remaining_upper < 0:
        raise OnlineGRPOError('no feasible rollout start offset leaves the configured minimum remaining steps')
    upper = min(config.rollout_start_offset_max_steps, remaining_upper)
    if temporary_max_offset_steps is not None:
        upper = min(upper, temporary_max_offset_steps)
    return int(upper)

def _sample_rollout_start_offset(config: JointGRPOOnlineConfig, ready_step: int, generator: torch.Generator, temporary_max_offset_steps: int | None=None) -> tuple[int, int]:
    """Draw one inclusive uniform offset from the shared training generator."""
    if not isinstance(generator, torch.Generator):
        raise OnlineGRPOError('rollout start sampling requires a torch.Generator')
    upper = _rollout_start_offset_upper_bound(config, ready_step, temporary_max_offset_steps=temporary_max_offset_steps)
    offset = int(torch.randint(0, upper + 1, (1,), generator=generator, device=generator.device, dtype=torch.int64).item())
    return (offset, upper)
