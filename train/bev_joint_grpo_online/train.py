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
from train.bev_joint_grpo_online.config import JointGRPOConstraintConfig, JointGRPOOnlineConfig, JointGRPOTrainingConfig, OnlineGRPOError
from train.bev_joint_grpo_online.environment import constant_velocity_actions, episode_has_ended, execute_cached_frozen_baseline, joint_trajectory_action, model_inputs_to_batch, optimize_selected_model_trajectories
from train.bev_joint_grpo_online.validation import build_grpo_validation_state_bank
from train.bev_joint_grpo_online.runner import run_joint_grpo_training

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
    constraint = payload.get('constraint', {'mode': 'none'})
    if not isinstance(constraint, Mapping):
        raise OnlineGRPOError('online GRPO YAML constraint must be a mapping')
    constraint_config = JointGRPOConstraintConfig(
        mode=str(constraint.get('mode', 'none')),
        loss_weight=float(constraint.get('loss_weight', 0.05)),
        wheelbase_m=float(constraint.get('wheelbase_m', 5.6)),
        min_segment_length_m=float(constraint.get('min_segment_length_m', 0.2)),
        steering_initial_limit_deg=float(constraint.get('steering_initial_limit_deg', 48.24)),
        steering_final_limit_deg=float(constraint.get('steering_final_limit_deg', 24.13)),
        steering_schedule_power=float(constraint.get('steering_schedule_power', 1.0)),
        steering_projection_passes=int(constraint.get('steering_projection_passes', 2)),
        steering_max_target_correction_m=float(constraint.get('steering_max_target_correction_m', 0.75)),
    )
    return JointGRPOTrainingConfig(variant=variant, run_mode=run_mode, source_checkpoint=Path(source_checkpoint), online=online_config, constraint=constraint_config)

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
