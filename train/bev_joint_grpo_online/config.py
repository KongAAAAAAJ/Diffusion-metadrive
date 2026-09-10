"""Configuration for online BEV joint GRPO training."""
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from scenarios.bev_round13_contract import DEVELOPMENT_SEEDS, PRIMARY_S5_S9_SCENARIOS, BEVScenarioContractError, primary_scenario_contract
AGENT_IDS = ("agent0", "agent1", "agent2")
MUTABLE_RUNTIME_CONFIG_PATH = "configs/train/bev_joint_grpo.yaml"

class OnlineGRPOError(RuntimeError):
    """Raised when online raw-domain GRPO violates its contract."""

@dataclass(frozen=True)
class JointGRPOOnlineConfig:
    device: str = 'cuda'
    seed: int = 17
    trajectories_per_mode: int = 48
    total_rollout_groups: int = 100
    resume_checkpoint: Path | None = None
    validation_state_bank: Path = Path('evaluation/artifacts/grpo_validation_state_bank_v1.pt')
    scenarios: tuple[tuple[str, str], ...] = PRIMARY_S5_S9_SCENARIOS
    scenario_seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    environment_steps_per_episode: int = 100
    rollout_groups_per_bucket_visit: int = 10
    rollout_start_offset_max_steps: int = 200
    rollout_start_min_remaining_steps: int = 10
    validation_interval_rollouts: int = 20
    advantage_vector_log_interval_rollouts: int = 10
    max_sampling_attempts_per_state: int = 3
    max_sampling_attempts_multiplier: int = 10
    max_consecutive_empty_episodes: int = 5

    def __post_init__(self) -> None:
        if self.device not in ('cpu', 'cuda'):
            raise OnlineGRPOError('online GRPO device must be cpu or cuda')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise OnlineGRPOError('online GRPO seed must be an integer')
        if isinstance(self.trajectories_per_mode, bool) or not isinstance(self.trajectories_per_mode, int) or self.trajectories_per_mode < 2:
            raise OnlineGRPOError('trajectories_per_mode must be an integer greater than or equal to 2')
        for name in ('total_rollout_groups', 'environment_steps_per_episode', 'rollout_groups_per_bucket_visit', 'validation_interval_rollouts', 'advantage_vector_log_interval_rollouts', 'max_sampling_attempts_per_state', 'max_sampling_attempts_multiplier', 'max_consecutive_empty_episodes'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise OnlineGRPOError(f'{name} must be a positive integer')
        if isinstance(self.rollout_start_offset_max_steps, bool) or not isinstance(self.rollout_start_offset_max_steps, int) or self.rollout_start_offset_max_steps < 0:
            raise OnlineGRPOError('rollout_start_offset_max_steps must be a non-negative integer')
        if isinstance(self.rollout_start_min_remaining_steps, bool) or not isinstance(self.rollout_start_min_remaining_steps, int) or self.rollout_start_min_remaining_steps < 0:
            raise OnlineGRPOError('rollout_start_min_remaining_steps must be a non-negative integer')
        if self.environment_steps_per_episode < 10:
            raise OnlineGRPOError('environment_steps_per_episode must be at least 10')
        if self.rollout_start_min_remaining_steps < self.rollout_groups_per_bucket_visit:
            raise OnlineGRPOError('rollout_start_min_remaining_steps must be greater than or equal to rollout_groups_per_bucket_visit')
        if self.environment_steps_per_episode < self.rollout_start_min_remaining_steps:
            raise OnlineGRPOError('environment_steps_per_episode must be greater than or equal to rollout_start_min_remaining_steps')
        if not self.scenarios or any((len(value) != 2 or not value[0] or (not value[1]) for value in self.scenarios)):
            raise OnlineGRPOError('at least one scenario/route pair is required')
        try:
            primary_scenario_contract(self.scenarios)
        except BEVScenarioContractError as exc:
            raise OnlineGRPOError(str(exc)) from exc
        if not self.scenario_seeds or any((isinstance(value, bool) or not isinstance(value, int) for value in self.scenario_seeds)):
            raise OnlineGRPOError('scenario_seeds must contain integers')
        if self.resume_checkpoint is not None:
            object.__setattr__(self, 'resume_checkpoint', Path(self.resume_checkpoint))
        object.__setattr__(self, 'validation_state_bank', Path(self.validation_state_bank))

@dataclass(frozen=True)
class JointGRPOConstraintConfig:
    mode: Literal['none', 'tv_cbf_curvature'] = 'none'
    loss_weight: float = 0.05
    curvature_initial_limit_inv_m: float = 0.20
    curvature_final_limit_inv_m: float = 0.08
    curvature_schedule_power: float = 1.0
    curvature_projection_passes: int = 2
    curvature_max_target_correction_m: float = 0.75

    def __post_init__(self) -> None:
        if self.mode not in ('none', 'tv_cbf_curvature'):
            raise OnlineGRPOError('constraint mode must be none or tv_cbf_curvature')
        for name in (
            'curvature_initial_limit_inv_m',
            'curvature_final_limit_inv_m',
            'curvature_schedule_power',
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise OnlineGRPOError(f'{name} must be positive and finite')
        for name in ('loss_weight', 'curvature_max_target_correction_m'):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise OnlineGRPOError(f'{name} must be non-negative and finite')
        if self.curvature_initial_limit_inv_m < self.curvature_final_limit_inv_m:
            raise OnlineGRPOError('initial curvature limit must be >= final curvature limit')
        if (
            isinstance(self.curvature_projection_passes, bool)
            or not isinstance(self.curvature_projection_passes, int)
            or self.curvature_projection_passes <= 0
        ):
            raise OnlineGRPOError('curvature_projection_passes must be a positive integer')
        if self.mode == 'tv_cbf_curvature' and self.loss_weight <= 0.0:
            raise OnlineGRPOError('tv_cbf_curvature requires loss_weight > 0')

@dataclass(frozen=True)
class JointGRPOTrainingConfig:
    variant: Literal['A', 'B']
    run_mode: Literal['formal', 'smoke']
    source_checkpoint: Path
    online: JointGRPOOnlineConfig
    constraint: JointGRPOConstraintConfig = JointGRPOConstraintConfig()

    def __post_init__(self) -> None:
        if self.variant not in ('A', 'B'):
            raise OnlineGRPOError('online GRPO variant must be A or B')
        if self.run_mode not in ('formal', 'smoke'):
            raise OnlineGRPOError('run_mode must be formal or smoke')
        if not isinstance(self.source_checkpoint, Path):
            raise OnlineGRPOError('source_checkpoint must be a Path')
        if not isinstance(self.online, JointGRPOOnlineConfig):
            raise OnlineGRPOError('online must be a JointGRPOOnlineConfig')
        if not isinstance(self.constraint, JointGRPOConstraintConfig):
            raise OnlineGRPOError('constraint must be a JointGRPOConstraintConfig')
