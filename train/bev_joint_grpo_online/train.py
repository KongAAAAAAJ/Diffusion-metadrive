"""Online vehicle-mode reward joint GRPO training for Variants A/B."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Mapping
import yaml
if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
from scenarios.bev_round13_contract import DEVELOPMENT_SEEDS, HOLDOUT_SEEDS, PRIMARY_S5_S9_SCENARIOS
from train.bev_joint_grpo_online.config import (
    JointGRPOConstraintConfig,
    JointGRPOOnlineConfig,
    JointGRPOTrainingConfig,
    OnlineGRPOError,
    RiskPACTCurriculumConfig,
    RiskPACTPostTrainingConfig,
    SafetyPostTrainingConfig,
)
from train.bev_risk_pact.config import RiskPACTConfig, RiskPACTVisualizationConfig
from train.bev_joint_grpo_online.environment import constant_velocity_actions, episode_has_ended, execute_cached_frozen_baseline, joint_trajectory_action, model_inputs_to_batch, optimize_selected_model_trajectories
from train.bev_joint_grpo_online.validation import build_grpo_validation_state_bank
from train.bev_joint_grpo_online.runner import run_joint_grpo_training

def _mapping(value: object, *, name: str, default: Mapping[str, object] | None = None) -> Mapping[str, object]:
    if value is None and default is not None:
        return default
    if not isinstance(value, Mapping):
        raise OnlineGRPOError(f'online GRPO YAML {name} must be a mapping')
    return value

def _feasibility_from_mapping(value: Mapping[str, object], *, enabled: bool) -> JointGRPOConstraintConfig:
    return JointGRPOConstraintConfig(
        mode='tv_feasibility' if enabled else 'none',
        loss_weight=float(value.get('loss_weight', 0.01)),
        steering_weight=float(value.get('steering_weight', 1.0)),
        steering_rate_weight=float(value.get('steering_rate_weight', 1.0)),
        wheelbase_m=float(value.get('wheelbase_m', 5.6)),
        trajectory_dt_s=float(value.get('trajectory_dt_s', 0.5)),
        min_segment_length_m=float(value.get('min_segment_length_m', 0.2)),
        steering_initial_limit_deg=float(value.get('steering_initial_limit_deg', 48.24)),
        steering_final_limit_deg=float(value.get('steering_final_limit_deg', 24.13)),
        steering_rate_initial_limit_deg_s=float(value.get('steering_rate_initial_limit_deg_s', 120.0)),
        steering_rate_final_limit_deg_s=float(value.get('steering_rate_final_limit_deg_s', 60.0)),
        schedule_power=float(value.get('schedule_power', 1.0)),
    )

def _risk_pact_from_mapping(value: Mapping[str, object]) -> RiskPACTPostTrainingConfig:
    curriculum = _mapping(value.get('curriculum'), name='safety_post_training.risk_pact.curriculum', default={})
    visualization = _mapping(value.get('visualization'), name='safety_post_training.risk_pact.visualization', default={})
    components = _mapping(value.get('components'), name='safety_post_training.risk_pact.components', default={})
    actor = _mapping(value.get('actor'), name='safety_post_training.risk_pact.actor', default={})
    platoon = _mapping(value.get('platoon'), name='safety_post_training.risk_pact.platoon', default={})
    road = _mapping(value.get('road'), name='safety_post_training.risk_pact.road', default={})

    # Step-5.6 schema: signed-clearance actor geometry + smooth-max aggregation.
    # Legacy Step-5.5 keys are intentionally not silently reinterpreted because
    # Gaussian sigma margins and physical clearance margins have different units/semantics.
    risk_config = RiskPACTConfig(
        use_background_actor=bool(components.get('background_actor', True)),
        use_platoon_actor=bool(components.get('platoon_actor', True)),
        use_road_boundary=bool(components.get('road_boundary', True)),
        horizon_dt_s=float(value.get('horizon_dt_s', 0.5)),
        ego_length_m=float(actor.get('ego_length_m', 5.74)),
        ego_width_m=float(actor.get('ego_width_m', 2.30)),
        background_longitudinal_clearance_m=float(actor.get('background_longitudinal_clearance_m', 5.0)),
        background_lateral_clearance_m=float(actor.get('background_lateral_clearance_m', 0.4)),
        actor_temperature_m=float(actor.get('temperature_m', 0.50)),
        actor_softmax_beta=float(actor.get('softmax_beta', 12.0)),
        platoon_vehicle_length_m=float(platoon.get('vehicle_length_m', 5.74)),
        platoon_vehicle_width_m=float(platoon.get('vehicle_width_m', 2.30)),
        platoon_longitudinal_clearance_m=float(platoon.get('longitudinal_clearance_m', 7.0)),
        platoon_lateral_clearance_m=float(platoon.get('lateral_clearance_m', 0.4)),
        component_softmax_beta=float(value.get('component_softmax_beta', 12.0)),
        road_safety_margin_m=float(road.get('safety_margin_m', 0.4)),
        road_temperature_m=float(road.get('temperature_m', 0.30)),
        temporal_softmax_beta=float(value.get('temporal_softmax_beta', 12.0)),
        risk_threshold=float(value.get('risk_threshold', 0.35)),
        violation_temperature=float(value.get('violation_temperature', 0.04)),
        safe_margin=float(value.get('safe_margin', 0.05)),
        teacher_step_m=float(value.get('teacher_step_m', 0.20)),
        gradient_eps=float(value.get('gradient_eps', 1.0e-6)),
        gradient_clip_norm=float(value.get('gradient_clip_norm', 10.0)),
    )
    curriculum_config = RiskPACTCurriculumConfig(
        start_scale=float(curriculum.get('start_scale', 0.2)),
        end_scale=float(curriculum.get('end_scale', 1.0)),
        warmup_updates=int(curriculum.get('warmup_updates', 0)),
        ramp_updates=int(curriculum.get('ramp_updates', 100)),
        schedule=str(curriculum.get('schedule', 'linear')),
    )
    max_events_raw = visualization.get('max_events', 4)
    visualization_config = RiskPACTVisualizationConfig(
        enabled=visualization.get('enabled', False),
        interval_steps=int(visualization.get('interval_steps', 5)),
        start_step=int(visualization.get('start_step', 0)),
        max_events=None if max_events_raw is None else int(max_events_raw),
        batch_index=int(visualization.get('batch_index', 0)),
        role_index=int(visualization.get('role_index', 0)),
        mode_index=int(visualization.get('mode_index', 0)),
        output_subdir=str(visualization.get('output_subdir', 'risk_field_visualizations')),
    )
    return RiskPACTPostTrainingConfig(
        distill_weight=float(value.get('distill_weight', 1.0)),
        risk=risk_config,
        curriculum=curriculum_config,
        visualization=visualization_config,
    )

def _legacy_safety_config(payload: Mapping[str, object]) -> tuple[SafetyPostTrainingConfig, JointGRPOConstraintConfig]:
    constraint = _mapping(payload.get('constraint'), name='constraint', default={'mode': 'none'})
    legacy_mode = str(constraint.get('mode', 'none'))
    if legacy_mode not in ('none', 'tv_feasibility'):
        raise OnlineGRPOError('constraint mode must be none or tv_feasibility')
    enabled = legacy_mode == 'tv_feasibility'
    feasibility = _feasibility_from_mapping(constraint, enabled=enabled)
    mode = 'feasibility_loss' if enabled else 'none'
    return SafetyPostTrainingConfig(mode=mode, feasibility=feasibility), feasibility

def _safety_config_from_payload(payload: Mapping[str, object]) -> tuple[SafetyPostTrainingConfig, JointGRPOConstraintConfig]:
    if 'safety_post_training' not in payload:
        return _legacy_safety_config(payload)
    if 'constraint' in payload:
        raise OnlineGRPOError('use either safety_post_training or legacy constraint, not both')
    safety = _mapping(payload.get('safety_post_training'), name='safety_post_training')
    mode = str(safety.get('mode', 'none'))
    allowed = ('none', 'feasibility_loss', 'risk_pact_lite', 'feasibility_plus_risk_pact')
    if mode not in allowed:
        raise OnlineGRPOError('safety_post_training mode must be one of: ' + ', '.join(allowed))
    feasibility_mapping = _mapping(safety.get('feasibility'), name='safety_post_training.feasibility', default={})
    feasibility_enabled = mode in ('feasibility_loss', 'feasibility_plus_risk_pact')
    feasibility = _feasibility_from_mapping(feasibility_mapping, enabled=feasibility_enabled)
    risk_mapping = _mapping(safety.get('risk_pact'), name='safety_post_training.risk_pact', default={})
    risk_pact = _risk_pact_from_mapping(risk_mapping)
    return SafetyPostTrainingConfig(mode=mode, feasibility=feasibility, risk_pact=risk_pact), feasibility

def _config_from_yaml(path: Path) -> JointGRPOTrainingConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise OnlineGRPOError(f'unable to read online GRPO config: {path}') from exc
    if not isinstance(payload, Mapping):
        raise OnlineGRPOError('online GRPO YAML root must be a mapping')
    run = payload.get('run')
    if not isinstance(run, Mapping):
        raise OnlineGRPOError('online GRPO YAML requires a run mapping')
    required_run_fields = {'variant', 'run_mode', 'source_checkpoint'}
    if any((not isinstance(name, str) for name in run)) or set(run) != required_run_fields:
        raise OnlineGRPOError('online GRPO YAML run mapping must contain exactly variant, run_mode, and source_checkpoint')
    variant = run['variant']
    if not isinstance(variant, str) or variant not in ('A', 'B'):
        raise OnlineGRPOError('online GRPO variant must be A or B')
    run_mode = run['run_mode']
    if not isinstance(run_mode, str) or run_mode not in ('formal', 'smoke'):
        raise OnlineGRPOError('run_mode must be formal or smoke')
    source_checkpoint = run['source_checkpoint']
    if not isinstance(source_checkpoint, str) or not source_checkpoint.strip():
        raise OnlineGRPOError('source_checkpoint must be a non-empty path string')
    online = payload.get('online')
    if not isinstance(online, Mapping):
        raise OnlineGRPOError('online GRPO YAML requires an online mapping')
    legacy_fields = {'total_optimizer_steps', 'validation_interval_steps', 'advantage_vector_log_interval_steps', 'clip_epsilon', 'clip_epsilon_low', 'clip_epsilon_high', 'update_epochs', 'group_size', 'pretrain_improvement_margin', 'max_candidate_groups_per_state', 'max_attempted_groups_multiplier'}
    present_legacy = sorted(legacy_fields.intersection(online))
    if present_legacy:
        raise OnlineGRPOError('online GRPO YAML contains legacy fields: ' + ', '.join(present_legacy))
    scenarios = tuple(((str(value['scenario']), str(value['route'])) for value in online.get('scenarios', ()) if isinstance(value, Mapping)))
    online_config = JointGRPOOnlineConfig(device=str(online.get('device', 'cuda')), seed=online.get('seed', 17), trajectories_per_mode=online.get('trajectories_per_mode', 48), total_rollout_groups=online.get('total_rollout_groups', 100), resume_checkpoint=Path(str(online['resume_checkpoint'])) if online.get('resume_checkpoint') else None, validation_state_bank=Path(str(online.get('validation_state_bank', 'evaluation/artifacts/grpo_validation_state_bank_v1.pt'))), scenarios=scenarios or PRIMARY_S5_S9_SCENARIOS, scenario_seeds=tuple((int(value) for value in online.get('scenario_seeds', DEVELOPMENT_SEEDS))), environment_steps_per_episode=int(online.get('environment_steps_per_episode', 100)), rollout_groups_per_bucket_visit=online.get('rollout_groups_per_bucket_visit', 10), rollout_start_offset_max_steps=online.get('rollout_start_offset_max_steps', 200), rollout_start_min_remaining_steps=online.get('rollout_start_min_remaining_steps', 10), validation_interval_rollouts=online.get('validation_interval_rollouts', 20), advantage_vector_log_interval_rollouts=online.get('advantage_vector_log_interval_rollouts', 10), max_sampling_attempts_per_state=online.get('max_sampling_attempts_per_state', 3), max_sampling_attempts_multiplier=online.get('max_sampling_attempts_multiplier', 10), max_consecutive_empty_episodes=online.get('max_consecutive_empty_episodes', 5))
    safety_config, compatibility_constraint = _safety_config_from_payload(payload)
    return JointGRPOTrainingConfig(
        variant=variant,
        run_mode=run_mode,
        source_checkpoint=Path(source_checkpoint),
        online=online_config,
        constraint=compatibility_constraint,
        safety_post_training=safety_config,
    )

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    arguments = parser.parse_args()
    training_config = _config_from_yaml(arguments.config)
    report = run_joint_grpo_training(training_config, run_dir=arguments.run_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0

__all__ = ["DEVELOPMENT_SEEDS", "HOLDOUT_SEEDS", "PRIMARY_S5_S9_SCENARIOS", "JointGRPOOnlineConfig", "JointGRPOTrainingConfig", "OnlineGRPOError", "constant_velocity_actions", "episode_has_ended", "execute_cached_frozen_baseline", "build_grpo_validation_state_bank", "joint_trajectory_action", "model_inputs_to_batch", "optimize_selected_model_trajectories", "run_joint_grpo_training"]

if __name__ == "__main__":
    raise SystemExit(main())
