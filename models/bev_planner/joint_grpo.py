"""Per-vehicle, same-mode GRPO core for the BEV-only diffusion planner."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import weakref
from dataclasses import dataclass, replace
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from models.bev_planner.bev_only_diffusion_planner import (
    BEVOnlyDiffusionPlanner,
    BEVPlannerContext,
    NUM_PLATOON_ROLES,
    TRAJECTORY_DIM,
)
from models.bev_planner.mode_contract import NUM_MODES, TRAJECTORY_STEPS
from models.bev_planner.ddim_transition import (
    DDIMNoiseBundle,
    DEFAULT_DDIM_PATH,
    StandardGaussianDDIM,
)
from models.bev_planner.time_varying_feasibility import (
    steering_feasibility_loss,
    time_varying_limit,
)
from train.bev_risk_pact.config import RiskPACTConfig
from train.bev_risk_pact.curriculum import risk_pact_curriculum_state
from train.bev_risk_pact.loss import pact_lite_distillation_loss
from train.bev_risk_pact.platoon_actor import build_platoon_actor_state
from train.bev_risk_pact.road_field import build_drivable_signed_distance
from train.bev_risk_pact.teacher import build_x0_pact_teacher


class JointGRPOError(RuntimeError):
    """Raised when the strict joint GRPO contract is violated."""


@dataclass(frozen=True)
class JointGRPOConfig:
    trajectories_per_mode: int = 48
    trajectory_pg_weight: float = 1.0
    bc_weight: float = 0.1
    reference_kl_weight: float = 0.02
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    advantage_scale: float = 1.0
    baseline_tolerance: float = 1e-6
    post_update_reference_kl_max: float = 0.25
    max_adapter_relative_drift: float = 0.02
    xy_beta_m: float = 1.0
    heading_beta_rad: float = 0.1
    heading_bc_weight: float = 0.2

    # Unified safety post-training switch.  Risk-PACT-lite remains a final-x0
    # pilot; feasibility is evaluated at every DDIM replay step.
    safety_post_training_mode: str = "none"

    feasibility_weight: float = 0.01
    steering_feasibility_weight: float = 1.0
    steering_rate_feasibility_weight: float = 1.0
    wheelbase_m: float = 5.6
    trajectory_dt_s: float = 0.5
    min_segment_length_m: float = 0.2
    steering_initial_limit_deg: float = 48.24
    steering_final_limit_deg: float = 24.13
    steering_rate_initial_limit_deg_s: float = 120.0
    steering_rate_final_limit_deg_s: float = 60.0
    feasibility_schedule_power: float = 1.0

    risk_pact_distill_weight: float = 1.0
    risk_pact_use_background_actor: bool = True
    risk_pact_use_platoon_actor: bool = True
    risk_pact_use_road_boundary: bool = True
    risk_pact_horizon_dt_s: float = 0.5
    risk_pact_ego_length_m: float = 5.74
    risk_pact_ego_width_m: float = 2.30
    risk_pact_background_longitudinal_clearance_m: float = 5.0
    risk_pact_background_lateral_clearance_m: float = 0.4
    risk_pact_actor_temperature_m: float = 0.50
    risk_pact_actor_softmax_beta: float = 12.0
    risk_pact_platoon_vehicle_length_m: float = 5.74
    risk_pact_platoon_vehicle_width_m: float = 2.30
    risk_pact_platoon_longitudinal_clearance_m: float = 7.0
    risk_pact_platoon_lateral_clearance_m: float = 0.4
    risk_pact_component_softmax_beta: float = 12.0
    risk_pact_road_safety_margin_m: float = 0.4
    risk_pact_road_temperature_m: float = 0.30
    risk_pact_temporal_softmax_beta: float = 12.0
    risk_pact_risk_threshold: float = 0.35
    risk_pact_violation_temperature: float = 0.04
    risk_pact_safe_margin: float = 0.05
    risk_pact_teacher_step_m: float = 0.20
    risk_pact_gradient_eps: float = 1.0e-6
    risk_pact_gradient_clip_norm: float = 10.0
    risk_pact_curriculum_start_scale: float = 0.2
    risk_pact_curriculum_end_scale: float = 1.0
    risk_pact_curriculum_warmup_updates: int = 0
    risk_pact_curriculum_ramp_updates: int = 100
    risk_pact_curriculum_schedule: str = "linear"

    # Step-6 diagnostics. These do not change the optimized objective.
    risk_pact_diagnostics_enabled: bool = True
    risk_pact_gradient_diagnostics: bool = True
    risk_pact_gradient_diagnostics_interval: int = 1
    risk_pact_teacher_improvement_eps: float = 1.0e-6
    risk_pact_late_horizon_start_step: int = 6

    def __post_init__(self) -> None:
        allowed_safety_modes = (
            "none",
            "feasibility_loss",
            "risk_pact_lite",
            "feasibility_plus_risk_pact",
        )
        if self.safety_post_training_mode not in allowed_safety_modes:
            raise JointGRPOError(
                "safety_post_training_mode must be one of: "
                + ", ".join(allowed_safety_modes)
            )
        if (
            isinstance(self.trajectories_per_mode, bool)
            or not isinstance(self.trajectories_per_mode, int)
            or self.trajectories_per_mode < 2
        ):
            raise JointGRPOError(
                "trajectories_per_mode must be an integer greater than or equal to 2"
            )
        positive = (
            "learning_rate",
            "max_grad_norm",
            "advantage_scale",
            "baseline_tolerance",
            "post_update_reference_kl_max",
            "max_adapter_relative_drift",
            "xy_beta_m",
            "heading_beta_rad",
            "wheelbase_m",
            "trajectory_dt_s",
            "min_segment_length_m",
            "steering_initial_limit_deg",
            "steering_final_limit_deg",
            "steering_rate_initial_limit_deg_s",
            "steering_rate_final_limit_deg_s",
            "feasibility_schedule_power",
            "risk_pact_horizon_dt_s",
            "risk_pact_ego_length_m",
            "risk_pact_ego_width_m",
            "risk_pact_actor_temperature_m",
            "risk_pact_actor_softmax_beta",
            "risk_pact_platoon_vehicle_length_m",
            "risk_pact_platoon_vehicle_width_m",
            "risk_pact_component_softmax_beta",
            "risk_pact_road_temperature_m",
            "risk_pact_temporal_softmax_beta",
            "risk_pact_violation_temperature",
            "risk_pact_teacher_step_m",
            "risk_pact_gradient_eps",
            "risk_pact_gradient_clip_norm",
        )
        non_negative = (
            "trajectory_pg_weight",
            "bc_weight",
            "reference_kl_weight",
            "weight_decay",
            "heading_bc_weight",
            "feasibility_weight",
            "steering_feasibility_weight",
            "steering_rate_feasibility_weight",
            "risk_pact_distill_weight",
            "risk_pact_background_longitudinal_clearance_m",
            "risk_pact_background_lateral_clearance_m",
            "risk_pact_platoon_longitudinal_clearance_m",
            "risk_pact_platoon_lateral_clearance_m",
            "risk_pact_road_safety_margin_m",
            "risk_pact_teacher_improvement_eps",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise JointGRPOError(f"{name} must be positive and finite")
        for name in non_negative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise JointGRPOError(f"{name} must be non-negative and finite")
        if self.steering_initial_limit_deg >= 90.0 or self.steering_final_limit_deg >= 90.0:
            raise JointGRPOError("steering limits must be smaller than 90 degrees")
        if self.steering_initial_limit_deg < self.steering_final_limit_deg:
            raise JointGRPOError("initial steering limit must be >= final steering limit")
        if self.steering_rate_initial_limit_deg_s < self.steering_rate_final_limit_deg_s:
            raise JointGRPOError("initial steering-rate limit must be >= final steering-rate limit")
        if self.uses_feasibility and self.feasibility_weight <= 0.0:
            raise JointGRPOError("feasibility safety mode requires feasibility_weight > 0")
        if self.uses_feasibility and self.steering_feasibility_weight <= 0.0 and self.steering_rate_feasibility_weight <= 0.0:
            raise JointGRPOError("feasibility safety mode requires a positive component weight")
        for name in (
            "risk_pact_use_background_actor",
            "risk_pact_use_platoon_actor",
            "risk_pact_use_road_boundary",
        ):
            if not isinstance(getattr(self, name), bool):
                raise JointGRPOError(f"{name} must be bool")
        for name in ("risk_pact_diagnostics_enabled", "risk_pact_gradient_diagnostics"):
            if not isinstance(getattr(self, name), bool):
                raise JointGRPOError(f"{name} must be bool")
        if (
            isinstance(self.risk_pact_gradient_diagnostics_interval, bool)
            or int(self.risk_pact_gradient_diagnostics_interval) <= 0
        ):
            raise JointGRPOError("risk_pact_gradient_diagnostics_interval must be a positive integer")
        if (
            isinstance(self.risk_pact_late_horizon_start_step, bool)
            or not 0 <= int(self.risk_pact_late_horizon_start_step) < TRAJECTORY_STEPS
        ):
            raise JointGRPOError(
                "risk_pact_late_horizon_start_step must lie on the planning horizon"
            )
        if self.uses_risk_pact and not (
            self.risk_pact_use_background_actor
            or self.risk_pact_use_platoon_actor
            or self.risk_pact_use_road_boundary
        ):
            raise JointGRPOError("Risk-PACT requires at least one enabled risk component")
        if self.uses_risk_pact and self.risk_pact_distill_weight <= 0.0:
            raise JointGRPOError("Risk-PACT safety mode requires risk_pact_distill_weight > 0")
        if not 0.0 < self.risk_pact_risk_threshold < 1.0:
            raise JointGRPOError("risk_pact_risk_threshold must be in (0,1)")
        if not 0.0 <= self.risk_pact_safe_margin < self.risk_pact_risk_threshold:
            raise JointGRPOError("risk_pact_safe_margin must be in [0, risk_threshold)")
        if not (0.0 <= self.risk_pact_curriculum_start_scale <= self.risk_pact_curriculum_end_scale <= 1.0):
            raise JointGRPOError("Risk-PACT curriculum scales must satisfy 0 <= start <= end <= 1")
        if self.risk_pact_curriculum_warmup_updates < 0 or self.risk_pact_curriculum_ramp_updates <= 0:
            raise JointGRPOError("Risk-PACT curriculum update counts are invalid")
        if self.risk_pact_curriculum_schedule != "linear":
            raise JointGRPOError("Risk-PACT Step-5 supports only a linear curriculum")

    @property
    def uses_feasibility(self) -> bool:
        return self.safety_post_training_mode in (
            "feasibility_loss", "feasibility_plus_risk_pact"
        )

    @property
    def uses_risk_pact(self) -> bool:
        return self.safety_post_training_mode in (
            "risk_pact_lite", "feasibility_plus_risk_pact"
        )

    @property
    def roll_timesteps(self) -> tuple[int, ...]:
        return DEFAULT_DDIM_PATH.timesteps

    @property
    def stochastic_timesteps(self) -> tuple[int, ...]:
        return self.roll_timesteps[:-1]


def joint_grpo_optimizer_contract() -> dict[str, object]:
    """Return the machine-readable single-step on-policy optimizer contract."""

    return {
        "version": "stage2_joint_grpo_optimizer_v13_risk_pact_step6_diagnostics",
        "ddim_path": DEFAULT_DDIM_PATH.as_dict(),
        "paired_behavior_policy": (
            "current and frozen N=48 paths use identical initial and DDIM "
            "transition noise"
        ),
        "policy_gradient": "-exp(logp-logp.detach())*advantage",
        "ppo_ratio_or_clipping": False,
        "advantage": (
            "collision_or_out=-1; otherwise max(R_current-mean(R_current),0) "
            "only when R_current>=R_frozen-1e-6; fixed scale 1"
        ),
        "signal_mode": "at least one nonzero sample advantage",
        "rollout_consumption": (
            "one external update call consumes one accepted live rollout and "
            "performs at most one backward and one optimizer step"
        ),
        "loss_reduction": (
            "fixed mean over 48 trajectories*3 DDIM steps, then signal modes, "
            "then vehicles, then batch"
        ),
        "minibatch_semantics": "none; one complete paired rollout",
        "reference_regularization": (
            "trajectory behavior-cloning and frozen-Stage1 trajectory KL cover "
            "every hard-valid optimizer-executable mode; no mode KL"
        ),
        "risk_pact_lite": (
            "final-x0 projected teacher from signed-clearance actor fields and "
            "DRIVABLE signed-distance road risk with smooth-max aggregation; masked "
            "distillation shares the same guarded optimizer step"
        ),
        "trainable_parameters": (
            "ten zero-initialized mode-specific output residual weights and biases"
        ),
        "weight_decay": 0.0,
        "post_update_guard": {
            "reference_kl_max": 0.25,
            "max_mode_adapter_relative_drift": 0.02,
            "failure": "restore residual parameters and Adam state",
        },
        "budget_unit": "accepted_update_state",
        "checkpoint_boundary": "after one accepted guarded optimizer step",
        "trainable_modules": ["diffusion_decoder.trajectory_head.mode_residual"],
    }


def joint_grpo_optimizer_contract_sha256() -> str:
    """Return the canonical digest for a concrete optimizer contract."""

    encoded = json.dumps(
        joint_grpo_optimizer_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RiskPACTRolloutContext:
    """Detached scene tensors needed by multi-source Risk-PACT post-training.

    Background actor state uses the shared 8-D contract
    [x_local, y_local, sin(d_heading), cos(d_heading),
     dvx_local, dvy_local, length, width].

    Platoon neighbors are reconstructed from ``ego_state`` and
    ``formation_relation_state`` during the guarded update.  ``drivable_sdf``
    is a detached metric signed-distance raster built once at rollout capture.
    """

    background_actor_state: Tensor
    background_actor_valid_mask: Tensor
    ego_state: Tensor
    formation_relation_state: Tensor
    relation_valid_mask: Tensor
    drivable_sdf: Tensor | None = None

    def __post_init__(self) -> None:
        state = self.background_actor_state
        valid = self.background_actor_valid_mask
        if not isinstance(state, Tensor) or state.dtype != torch.float32:
            raise JointGRPOError(
                "background_actor_state must be a float32 torch.Tensor"
            )
        if state.ndim != 4 or int(state.shape[1]) != NUM_PLATOON_ROLES or int(state.shape[-1]) != 8:
            raise JointGRPOError(
                "background_actor_state must have shape [B,3,A,8]"
            )
        if int(state.shape[0]) <= 0 or int(state.shape[2]) <= 0:
            raise JointGRPOError(
                "background_actor_state must contain a non-empty batch and actor axis"
            )
        if not isinstance(valid, Tensor) or valid.dtype != torch.bool:
            raise JointGRPOError(
                "background_actor_valid_mask must be a bool torch.Tensor"
            )
        if tuple(valid.shape) != tuple(state.shape[:-1]):
            raise JointGRPOError(
                "background_actor_valid_mask must have shape [B,3,A]"
            )

        ego = self.ego_state
        relation = self.formation_relation_state
        relation_valid = self.relation_valid_mask
        if not isinstance(ego, Tensor) or ego.dtype != torch.float32 or ego.ndim != 3:
            raise JointGRPOError("ego_state must be float32 [B,3,D]")
        if tuple(ego.shape[:2]) != (int(state.shape[0]), NUM_PLATOON_ROLES):
            raise JointGRPOError("ego_state B/R axes must match Risk-PACT actor context")
        if not isinstance(relation, Tensor) or relation.dtype != torch.float32:
            raise JointGRPOError("formation_relation_state must be float32")
        if tuple(relation.shape) != (int(state.shape[0]), NUM_PLATOON_ROLES, 12):
            raise JointGRPOError("formation_relation_state must have shape [B,3,12]")
        if not isinstance(relation_valid, Tensor) or relation_valid.dtype != torch.bool:
            raise JointGRPOError("relation_valid_mask must be bool")
        if tuple(relation_valid.shape) != (int(state.shape[0]), NUM_PLATOON_ROLES, 2):
            raise JointGRPOError("relation_valid_mask must have shape [B,3,2]")

        tensors = (state, valid, ego, relation, relation_valid)
        if any(t.device != state.device for t in tensors):
            raise JointGRPOError("Risk-PACT rollout context tensors must share a device")
        if any(t.requires_grad for t in tensors):
            raise JointGRPOError("Risk-PACT rollout context must be graph-free")
        for name, tensor in (
            ("background_actor_state", state),
            ("ego_state", ego),
            ("formation_relation_state", relation),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise JointGRPOError(f"{name} must contain finite values")

        sdf = self.drivable_sdf
        if sdf is not None:
            if not isinstance(sdf, Tensor) or sdf.dtype != torch.float32:
                raise JointGRPOError("drivable_sdf must be float32")
            if tuple(sdf.shape) != (int(state.shape[0]), NUM_PLATOON_ROLES, 256, 256):
                raise JointGRPOError("drivable_sdf must have shape [B,3,256,256]")
            if sdf.device != state.device or sdf.requires_grad:
                raise JointGRPOError("drivable_sdf must be detached and colocated with context")
            if not bool(torch.isfinite(sdf).all()):
                raise JointGRPOError("drivable_sdf must contain finite values")

    @property
    def batch_size(self) -> int:
        return int(self.background_actor_state.shape[0])


@dataclass(frozen=True)
class JointGRPORollout:
    context: BEVPlannerContext
    coarse_trajectories: Tensor
    mode_valid_mask: Tensor
    chains_normalized: Tensor
    candidate_trajectories: Tensor
    frozen_candidate_trajectories: Tensor
    risk_pact_context: RiskPACTRolloutContext | None = None
    noise_bundle_identity: tuple[int, int, int] | None = None
    current_rewards: Tensor | None = None
    frozen_rewards: Tensor | None = None
    collision_mask: Tensor | None = None
    out_of_drivable_mask: Tensor | None = None
    valid_executable_mode_mask: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.candidate_trajectories.shape[0])

    @property
    def trajectories_per_mode(self) -> int:
        return int(self.candidate_trajectories.shape[3])

    def require_risk_pact_context(self) -> RiskPACTRolloutContext:
        """Return the bound Risk-PACT scene context or fail closed."""

        value = self.risk_pact_context
        if value is None:
            raise JointGRPOError(
                "rollout is missing Risk-PACT actor context"
            )
        if value.batch_size != self.batch_size:
            raise JointGRPOError(
                "Risk-PACT actor context batch does not match rollout batch"
            )
        return value

    def with_reward_signals(
        self,
        *,
        current_rewards: Tensor,
        frozen_rewards: Tensor,
        collision_mask: Tensor,
        out_of_drivable_mask: Tensor,
        valid_executable_mode_mask: Tensor,
    ) -> "JointGRPORollout":
        """Bind paired reward and safety tensors without mutating the rollout."""

        return replace(
            self,
            current_rewards=current_rewards.detach(),
            frozen_rewards=frozen_rewards.detach(),
            collision_mask=collision_mask.detach(),
            out_of_drivable_mask=out_of_drivable_mask.detach(),
            valid_executable_mode_mask=valid_executable_mode_mask.detach(),
        )


@dataclass(frozen=True)
class JointGRPORolloutB(JointGRPORollout):
    predecessor_action_history_normalized: Tensor | None = None


@dataclass(frozen=True)
class JointGRPOLossResult:
    total: Tensor
    trajectory_pg: Tensor
    behavior_cloning: Tensor
    trajectory_reference_kl: Tensor
    reference_kl: Tensor
    centered_rewards: Tensor
    advantages: Tensor
    signal_mode_mask: Tensor
    valid_executable_mode_mask: Tensor
    new_trajectory_log_prob: Tensor
    trajectory_pg_by_mode: Tensor
    behavior_cloning_by_mode: Tensor
    reference_kl_by_mode: Tensor
    feasibility: Tensor
    weighted_feasibility: Tensor
    risk_pact_distillation: Tensor
    weighted_risk_pact_distillation: Tensor
    risk_pact_active_ratio: Tensor
    risk_pact_active_count: Tensor
    risk_pact_teacher_displacement_mean_m: Tensor
    risk_pact_teacher_displacement_max_m: Tensor
    risk_pact_trajectory_risk_mean: Tensor
    risk_pact_trajectory_risk_max: Tensor
    risk_pact_violation_mean: Tensor
    risk_pact_curriculum_scale: Tensor
    risk_pact_background_risk_mean: Tensor
    risk_pact_platoon_risk_mean: Tensor
    risk_pact_road_risk_mean: Tensor
    risk_pact_road_signed_distance_min_m: Tensor
    risk_pact_valid_active_ratio: Tensor
    risk_pact_valid_active_count: Tensor
    risk_pact_valid_candidate_count: Tensor
    risk_pact_teacher_trajectory_risk_mean: Tensor
    risk_pact_teacher_trajectory_risk_max: Tensor
    risk_pact_teacher_risk_reduction_mean: Tensor
    risk_pact_teacher_improvement_fraction: Tensor
    risk_pact_critical_time_mean_s: Tensor
    risk_pact_critical_time_max_s: Tensor
    risk_pact_late_horizon_critical_fraction: Tensor
    risk_pact_background_signed_clearance_min_mean_m: Tensor
    risk_pact_background_signed_clearance_min_m: Tensor
    risk_pact_platoon_signed_clearance_min_mean_m: Tensor
    risk_pact_platoon_signed_clearance_min_m: Tensor
    risk_pact_road_safety_clearance_min_mean_m: Tensor
    risk_pact_road_safety_clearance_min_m: Tensor

    def scalar_metrics(self) -> dict[str, float]:
        values = {
            "loss/total": self.total,
            "loss/trajectory_pg": self.trajectory_pg,
            "loss/behavior_cloning": self.behavior_cloning,
            "loss/trajectory_reference_kl": self.trajectory_reference_kl,
            "loss/reference_kl": self.reference_kl,
            "loss/feasibility": self.feasibility,
            "loss/weighted_feasibility": self.weighted_feasibility,
            "loss/risk_pact_distillation": self.risk_pact_distillation,
            "loss/weighted_risk_pact_distillation": self.weighted_risk_pact_distillation,
            "risk_pact/active_ratio": self.risk_pact_active_ratio,
            "risk_pact/active_count": self.risk_pact_active_count,
            "risk_pact/teacher_displacement_mean_m": self.risk_pact_teacher_displacement_mean_m,
            "risk_pact/teacher_displacement_max_m": self.risk_pact_teacher_displacement_max_m,
            "risk_pact/trajectory_risk_mean": self.risk_pact_trajectory_risk_mean,
            "risk_pact/trajectory_risk_max": self.risk_pact_trajectory_risk_max,
            "risk_pact/violation_mean": self.risk_pact_violation_mean,
            "risk_pact/curriculum_scale": self.risk_pact_curriculum_scale,
            "risk_pact/background_risk_mean": self.risk_pact_background_risk_mean,
            "risk_pact/platoon_risk_mean": self.risk_pact_platoon_risk_mean,
            "risk_pact/road_risk_mean": self.risk_pact_road_risk_mean,
            "risk_pact/road_signed_distance_min_m": self.risk_pact_road_signed_distance_min_m,
            "risk_pact/valid_active_ratio": self.risk_pact_valid_active_ratio,
            "risk_pact/valid_active_count": self.risk_pact_valid_active_count,
            "risk_pact/valid_candidate_count": self.risk_pact_valid_candidate_count,
            "risk_pact/teacher_trajectory_risk_mean": self.risk_pact_teacher_trajectory_risk_mean,
            "risk_pact/teacher_trajectory_risk_max": self.risk_pact_teacher_trajectory_risk_max,
            "risk_pact/teacher_risk_reduction_mean": self.risk_pact_teacher_risk_reduction_mean,
            "risk_pact/teacher_improvement_fraction": self.risk_pact_teacher_improvement_fraction,
            "risk_pact/critical_time_mean_s": self.risk_pact_critical_time_mean_s,
            "risk_pact/critical_time_max_s": self.risk_pact_critical_time_max_s,
            "risk_pact/late_horizon_critical_fraction": self.risk_pact_late_horizon_critical_fraction,
            "risk_pact/background_signed_clearance_min_mean_m": self.risk_pact_background_signed_clearance_min_mean_m,
            "risk_pact/background_signed_clearance_min_m": self.risk_pact_background_signed_clearance_min_m,
            "risk_pact/platoon_signed_clearance_min_mean_m": self.risk_pact_platoon_signed_clearance_min_mean_m,
            "risk_pact/platoon_signed_clearance_min_m": self.risk_pact_platoon_signed_clearance_min_m,
            "risk_pact/road_safety_clearance_min_mean_m": self.risk_pact_road_safety_clearance_min_mean_m,
            "risk_pact/road_safety_clearance_min_m": self.risk_pact_road_safety_clearance_min_m,
            "advantage/mean": _active_tensor_mean(
                self.advantages, self.valid_executable_mode_mask
            ),
            "advantage/rms": torch.sqrt(
                _active_tensor_mean(
                    self.advantages.square(), self.valid_executable_mode_mask
                )
            ),
            "advantage/positive_fraction": _active_tensor_mean(
                (self.advantages > 0).float(), self.valid_executable_mode_mask
            ),
            "advantage/collision_or_out_negative_fraction": _active_tensor_mean(
                (self.advantages < 0).float(), self.valid_executable_mode_mask
            ),
            "advantage/centered_rms": torch.sqrt(
                _active_tensor_mean(
                    self.centered_rewards.square(),
                    self.valid_executable_mode_mask,
                )
            ),
            "signal_mode/count": self.signal_mode_mask.float().sum(),
            "no_signal_mode/count": (
                self.valid_executable_mode_mask & ~self.signal_mode_mask
            ).float().sum(),
        }
        for mode in range(NUM_MODES):
            values[f"loss/mode_{mode}/trajectory_pg"] = (
                self.trajectory_pg_by_mode[mode]
            )
            values[f"loss/mode_{mode}/behavior_cloning"] = (
                self.behavior_cloning_by_mode[mode]
            )
            values[f"loss/mode_{mode}/reference_kl"] = (
                self.reference_kl_by_mode[mode]
            )
        return {name: float(value.detach().cpu()) for name, value in values.items()}

    def detached(self) -> JointGRPOLossResult:
        """Return a graph-free result safe to retain after the one update."""

        return JointGRPOLossResult(
            total=self.total.detach(),
            trajectory_pg=self.trajectory_pg.detach(),
            behavior_cloning=self.behavior_cloning.detach(),
            trajectory_reference_kl=self.trajectory_reference_kl.detach(),
            reference_kl=self.reference_kl.detach(),
            centered_rewards=self.centered_rewards.detach(),
            advantages=self.advantages.detach(),
            signal_mode_mask=self.signal_mode_mask.detach(),
            valid_executable_mode_mask=self.valid_executable_mode_mask.detach(),
            new_trajectory_log_prob=self.new_trajectory_log_prob.detach(),
            trajectory_pg_by_mode=self.trajectory_pg_by_mode.detach(),
            behavior_cloning_by_mode=self.behavior_cloning_by_mode.detach(),
            reference_kl_by_mode=self.reference_kl_by_mode.detach(),
            feasibility=self.feasibility.detach(),
            weighted_feasibility=self.weighted_feasibility.detach(),
            risk_pact_distillation=self.risk_pact_distillation.detach(),
            weighted_risk_pact_distillation=self.weighted_risk_pact_distillation.detach(),
            risk_pact_active_ratio=self.risk_pact_active_ratio.detach(),
            risk_pact_active_count=self.risk_pact_active_count.detach(),
            risk_pact_teacher_displacement_mean_m=self.risk_pact_teacher_displacement_mean_m.detach(),
            risk_pact_teacher_displacement_max_m=self.risk_pact_teacher_displacement_max_m.detach(),
            risk_pact_trajectory_risk_mean=self.risk_pact_trajectory_risk_mean.detach(),
            risk_pact_trajectory_risk_max=self.risk_pact_trajectory_risk_max.detach(),
            risk_pact_violation_mean=self.risk_pact_violation_mean.detach(),
            risk_pact_curriculum_scale=self.risk_pact_curriculum_scale.detach(),
            risk_pact_background_risk_mean=self.risk_pact_background_risk_mean.detach(),
            risk_pact_platoon_risk_mean=self.risk_pact_platoon_risk_mean.detach(),
            risk_pact_road_risk_mean=self.risk_pact_road_risk_mean.detach(),
            risk_pact_road_signed_distance_min_m=self.risk_pact_road_signed_distance_min_m.detach(),
            risk_pact_valid_active_ratio=self.risk_pact_valid_active_ratio.detach(),
            risk_pact_valid_active_count=self.risk_pact_valid_active_count.detach(),
            risk_pact_valid_candidate_count=self.risk_pact_valid_candidate_count.detach(),
            risk_pact_teacher_trajectory_risk_mean=self.risk_pact_teacher_trajectory_risk_mean.detach(),
            risk_pact_teacher_trajectory_risk_max=self.risk_pact_teacher_trajectory_risk_max.detach(),
            risk_pact_teacher_risk_reduction_mean=self.risk_pact_teacher_risk_reduction_mean.detach(),
            risk_pact_teacher_improvement_fraction=self.risk_pact_teacher_improvement_fraction.detach(),
            risk_pact_critical_time_mean_s=self.risk_pact_critical_time_mean_s.detach(),
            risk_pact_critical_time_max_s=self.risk_pact_critical_time_max_s.detach(),
            risk_pact_late_horizon_critical_fraction=self.risk_pact_late_horizon_critical_fraction.detach(),
            risk_pact_background_signed_clearance_min_mean_m=self.risk_pact_background_signed_clearance_min_mean_m.detach(),
            risk_pact_background_signed_clearance_min_m=self.risk_pact_background_signed_clearance_min_m.detach(),
            risk_pact_platoon_signed_clearance_min_mean_m=self.risk_pact_platoon_signed_clearance_min_mean_m.detach(),
            risk_pact_platoon_signed_clearance_min_m=self.risk_pact_platoon_signed_clearance_min_m.detach(),
            risk_pact_road_safety_clearance_min_mean_m=self.risk_pact_road_safety_clearance_min_mean_m.detach(),
            risk_pact_road_safety_clearance_min_m=self.risk_pact_road_safety_clearance_min_m.detach(),
        )


@dataclass(frozen=True)
class JointGRPOUpdateResult:
    loss: JointGRPOLossResult
    gradient_norms: Mapping[str, float]
    clipped_gradient_norms: Mapping[str, float]
    total_gradient_norm: float
    adapter_relative_drifts: tuple[float, ...]
    post_update_reference_kl: float
    optimizer_step: int
    zero_signal: bool
    stability_guard_rejected: bool
    stability_guard_trigger_modes: tuple[int, ...]
    objective_gradient_diagnostics: Mapping[str, float]


class FrozenGRPOReference(nn.Module):
    """Frozen decoder and mode head copied from the initial Stage 1 policy."""

    def __init__(self, planner: BEVOnlyDiffusionPlanner) -> None:
        super().__init__()
        self.diffusion_decoder = copy.deepcopy(planner.diffusion_decoder)
        self.mode_head = copy.deepcopy(planner.mode_head)
        self.requires_grad_(False)
        self.eval()

    def predict(
        self,
        planner: BEVOnlyDiffusionPlanner,
        noisy_xy_normalized: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        coarse_trajectories: Tensor,
    ) -> tuple[Tensor, Tensor]:
        noisy_metric = planner._denormalize_xy(
            noisy_xy_normalized.clamp(-1.0, 1.0)
        )
        candidates, features = self.diffusion_decoder(
            noisy_metric,
            coarse_trajectories,
            timesteps,
            context,
        )
        return candidates, self.mode_head(features).squeeze(-1)


# Preserve the Round-11 public name while using a variant-neutral implementation.
FrozenVariantAReference = FrozenGRPOReference


class ModeResidualTrajectoryHead(nn.Module):
    """Frozen Stage-1 head plus disjoint zero-initialized mode residuals."""

    def __init__(self, base: nn.Sequential) -> None:
        super().__init__()
        if (
            not isinstance(base, nn.Sequential)
            or len(base) != 3
            or not isinstance(base[0], nn.Linear)
            or not isinstance(base[1], nn.GELU)
            or not isinstance(base[2], nn.Linear)
            or base[0].out_features != base[2].in_features
            or base[2].out_features != TRAJECTORY_STEPS * TRAJECTORY_DIM
        ):
            raise JointGRPOError("trajectory head does not match the Stage-1 MLP")
        self.base = base
        self.base.requires_grad_(False)
        hidden = int(base[2].in_features)
        output = int(base[2].out_features)
        dtype = base[2].weight.dtype
        self.mode_residual_weight = nn.Parameter(
            torch.zeros(
                NUM_MODES,
                output,
                hidden,
                device=base[2].weight.device,
                dtype=dtype,
            )
        )
        self.mode_residual_bias = nn.Parameter(
            torch.zeros(
                NUM_MODES,
                output,
                device=base[2].weight.device,
                dtype=dtype,
            )
        )

    def forward(self, mode_features: Tensor) -> Tensor:
        if mode_features.ndim < 2 or int(mode_features.shape[-2]) != NUM_MODES:
            raise JointGRPOError("mode residual head requires a size-10 mode axis")
        hidden = self.base[1](self.base[0](mode_features))
        base_output = self.base[2](hidden)
        residual = torch.einsum(
            "...kh,koh->...ko", hidden, self.mode_residual_weight
        ) + self.mode_residual_bias
        return base_output + residual

    def residual_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return self.mode_residual_weight, self.mode_residual_bias

    def adapter_relative_drifts(self) -> tuple[float, ...]:
        base_norm = torch.sqrt(
            self.base[2].weight.detach().float().square().sum()
            + self.base[2].bias.detach().float().square().sum()
        ).clamp_min(1e-12)
        weight = self.mode_residual_weight.detach().float()
        bias = self.mode_residual_bias.detach().float()
        norms = torch.sqrt(weight.square().sum(dim=(1, 2)) + bias.square().sum(dim=1))
        return tuple(float(value.cpu()) for value in norms / base_norm)


def fixed_scale_safe_advantages(
    current_rewards: Tensor,
    frozen_rewards: Tensor,
    collision_mask: Tensor,
    out_of_drivable_mask: Tensor,
    valid_executable_mode_mask: Tensor,
    *,
    trajectories_per_mode: int | None = None,
    advantage_scale: float = 1.0,
    baseline_tolerance: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return centered reward, fixed-scale truncated advantage and signal mask."""

    if not isinstance(current_rewards, Tensor) or current_rewards.dtype != torch.float32:
        raise JointGRPOError("current_rewards must be a float32 torch.Tensor")
    if current_rewards.ndim != 4 or tuple(current_rewards.shape[1:3]) != (
        NUM_PLATOON_ROLES,
        NUM_MODES,
    ):
        raise JointGRPOError("current_rewards must have shape [B,3,10,N]")
    count = int(current_rewards.shape[-1])
    if int(current_rewards.shape[0]) <= 0 or count < 2:
        raise JointGRPOError("rewards must contain a non-empty trajectory group")
    if trajectories_per_mode is not None and (
        isinstance(trajectories_per_mode, bool)
        or not isinstance(trajectories_per_mode, int)
        or trajectories_per_mode != count
    ):
        raise JointGRPOError(
            "trajectories_per_mode must match the rewards trajectory axis"
        )
    expected = tuple(current_rewards.shape)
    tensors = {
        "frozen_rewards": frozen_rewards,
        "collision_mask": collision_mask,
        "out_of_drivable_mask": out_of_drivable_mask,
    }
    for name, value in tensors.items():
        expected_dtype = torch.float32 if name == "frozen_rewards" else torch.bool
        if (
            not isinstance(value, Tensor)
            or value.dtype != expected_dtype
            or tuple(value.shape) != expected
            or value.device != current_rewards.device
        ):
            raise JointGRPOError(
                f"{name} must be colocated {expected_dtype} with shape [B,3,10,N]"
            )
    if not bool(torch.isfinite(current_rewards).all()) or not bool(
        torch.isfinite(frozen_rewards).all()
    ):
        raise JointGRPOError("paired rewards must contain finite values")
    if (
        not isinstance(valid_executable_mode_mask, Tensor)
        or valid_executable_mode_mask.dtype != torch.bool
        or tuple(valid_executable_mode_mask.shape) != tuple(current_rewards.shape[:3])
        or valid_executable_mode_mask.device != current_rewards.device
    ):
        raise JointGRPOError(
            "valid_executable_mode_mask must be colocated bool [B,3,10]"
        )
    scale = float(advantage_scale)
    tolerance = float(baseline_tolerance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise JointGRPOError("advantage_scale must be positive and finite")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise JointGRPOError("baseline_tolerance must be non-negative and finite")

    centered = current_rewards - current_rewards.mean(dim=-1, keepdim=True)
    unsafe = collision_mask | out_of_drivable_mask
    baseline_pass = current_rewards >= frozen_rewards - tolerance
    advantages = torch.where(
        unsafe,
        torch.full_like(centered, -scale),
        torch.where(baseline_pass, centered.clamp_min(0.0) * scale, 0.0),
    )
    advantages = advantages * valid_executable_mode_mask.unsqueeze(-1).to(
        advantages.dtype
    )
    signal_mode_mask = valid_executable_mode_mask & (advantages != 0).any(dim=-1)
    return centered, advantages, signal_mode_mask


def _active_tensor_mean(value: Tensor, active_mode_mask: Tensor) -> Tensor:
    """Mean all trailing entries belonging to active [B,R,K] blocks."""

    expanded = active_mode_mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    weights = expanded.expand_as(value).to(value.dtype)
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def _hierarchical_active_mean(value: Tensor, active_mode_mask: Tensor) -> Tensor:
    """Reduce trailing axes, active modes, roles and batch in that order."""

    if tuple(value.shape[:3]) != tuple(active_mode_mask.shape):
        raise JointGRPOError("loss blocks and active_mode_mask do not align")
    block_mean = value.reshape(*value.shape[:3], -1).mean(dim=-1)
    weights = active_mode_mask.to(block_mean.dtype)
    per_role = (block_mean * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
    return per_role.mean(dim=1).mean(dim=0)


def _per_mode_masked_mean(value: Tensor, mode_mask: Tensor) -> Tensor:
    """Reduce fixed trailing axes and valid batch/role blocks per mode."""

    if tuple(value.shape[:3]) != tuple(mode_mask.shape):
        raise JointGRPOError("loss blocks and mode mask do not align")
    block_mean = value.reshape(*value.shape[:3], -1).mean(dim=-1)
    weights = mode_mask.to(block_mean.dtype)
    numerator = (block_mean * weights).sum(dim=(0, 1))
    denominator = weights.sum(dim=(0, 1)).clamp_min(1.0)
    return numerator / denominator


def _repeat_context(context: BEVPlannerContext, groups: int) -> BEVPlannerContext:
    batch_size = context.batch_size
    bev = (
        context.bev_feature.unsqueeze(1)
        .expand(-1, groups, -1, -1, -1, -1)
        .reshape(batch_size * groups, *context.bev_feature.shape[1:])
    )
    roles = (
        context.role_tokens.unsqueeze(1)
        .expand(-1, groups, -1, -1)
        .reshape(batch_size * groups, *context.role_tokens.shape[1:])
    )
    return BEVPlannerContext(bev_feature=bev, role_tokens=roles)


def _repeat_groups(value: Tensor, groups: int) -> Tensor:
    return (
        value.unsqueeze(1)
        .expand(-1, groups, *([-1] * (value.ndim - 1)))
        .reshape(value.shape[0] * groups, *value.shape[1:])
    )


def _trajectory_bc_blocks(
    current: Tensor, reference: Tensor, config: JointGRPOConfig
) -> Tensor:
    """Return one BC value for every leading trajectory block."""

    xy = F.smooth_l1_loss(
        current[..., :2],
        reference[..., :2],
        beta=float(config.xy_beta_m),
        reduction="none",
    ).mean(dim=(-2, -1))
    heading_error = torch.atan2(
        torch.sin(current[..., 2] - reference[..., 2]),
        torch.cos(current[..., 2] - reference[..., 2]),
    )
    heading = F.smooth_l1_loss(
        heading_error,
        torch.zeros_like(heading_error),
        beta=float(config.heading_beta_rad),
        reduction="none",
    ).mean(dim=-1)
    return xy + float(config.heading_bc_weight) * heading


class _JointGRPOTrainerBase:
    """Variant-neutral single-step, fixed-scale joint GRPO implementation."""

    variant = ""
    predecessor_condition = ""

    required_model_inputs = (
        "bev",
        "ego_state",
        "formation_relation_state",
        "relation_valid_mask",
        "agent_role",
        "coarse_trajectories",
        "mode_valid_mask",
    )

    def __init__(
        self,
        planner: BEVOnlyDiffusionPlanner,
        config: JointGRPOConfig | None = None,
    ) -> None:
        if planner.config.predecessor_condition != self.predecessor_condition:
            raise JointGRPOError(
                f"JointGRPOTrainer{self.variant} requires variant "
                f"{self.variant} / {self.predecessor_condition}"
            )
        self.planner = planner
        self.config = config or JointGRPOConfig()
        self.transition = StandardGaussianDDIM(planner.config.num_train_timesteps)
        self.reference = FrozenGRPOReference(planner).to(
            next(planner.parameters()).device
        )
        planner.diffusion_decoder.trajectory_head = ModeResidualTrajectoryHead(
            planner.diffusion_decoder.trajectory_head
        )
        for parameter in planner.parameters():
            parameter.requires_grad_(False)
        residual_head = planner.diffusion_decoder.trajectory_head
        if not isinstance(residual_head, ModeResidualTrajectoryHead):
            raise JointGRPOError("failed to install mode residual trajectory head")
        for parameter in residual_head.residual_parameters():
            parameter.requires_grad_(True)
        self.planner.eval()
        self.optimizer = AdamW(
            residual_head.residual_parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=0.0,
        )
        self.optimizer_step = 0
        self._consumed_rollouts: weakref.WeakValueDictionary[
            int, JointGRPORollout
        ] = weakref.WeakValueDictionary()

    def _risk_pact_context_from_inputs(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> RiskPACTRolloutContext | None:
        """Capture graph-free multi-source scene geometry for Risk-PACT.

        SDF construction is intentionally skipped for non-Risk-PACT baselines so
        ``none`` and feasibility-only runs retain their previous collection cost.
        """

        if not self.config.uses_risk_pact:
            return None

        required = (
            "background_actor_state",
            "background_actor_valid_mask",
            "ego_state",
            "formation_relation_state",
            "relation_valid_mask",
        )
        missing = [name for name in required if name not in model_inputs]
        if missing:
            raise JointGRPOError(
                f"Risk-PACT multi-source context is missing fields: {missing}"
            )

        drivable_sdf = None
        if self.config.risk_pact_use_road_boundary:
            bev = model_inputs.get("bev")
            if bev is None:
                raise JointGRPOError("road-boundary Risk-PACT requires bev input")
            try:
                drivable_sdf = build_drivable_signed_distance(bev)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise JointGRPOError(f"failed to build Risk-PACT drivable SDF: {exc}") from exc

        context = RiskPACTRolloutContext(
            background_actor_state=model_inputs["background_actor_state"].detach(),
            background_actor_valid_mask=model_inputs["background_actor_valid_mask"].detach(),
            ego_state=model_inputs["ego_state"].detach(),
            formation_relation_state=model_inputs["formation_relation_state"].detach(),
            relation_valid_mask=model_inputs["relation_valid_mask"].detach(),
            drivable_sdf=None if drivable_sdf is None else drivable_sdf.detach(),
        )
        if context.batch_size != int(batch_size):
            raise JointGRPOError(
                "Risk-PACT scene context batch does not match planner context"
            )
        if context.background_actor_state.device != device:
            raise JointGRPOError(
                "Risk-PACT scene context must be colocated with planner inputs"
            )
        return context

    def _context_from_inputs(self, model_inputs: Mapping[str, Tensor]) -> BEVPlannerContext:
        missing = [name for name in self.required_model_inputs if name not in model_inputs]
        if missing:
            raise JointGRPOError(f"model_inputs are missing fields: {missing}")
        v2_inputs = {
            name: model_inputs[name]
            for name in (
                "background_actor_state",
                "background_actor_valid_mask",
                "scenario_code",
                "rule_formation_state",
                "rule_action_condition",
            )
            if name in model_inputs
        }
        with torch.no_grad():
            return self.planner.encode_context(
                model_inputs["bev"],
                model_inputs["ego_state"],
                model_inputs["formation_relation_state"],
                model_inputs["relation_valid_mask"],
                model_inputs["agent_role"],
                **v2_inputs,
            )

    def _rollout_prediction(
        self,
        sample: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        candidates, logits = self.planner.predict_denoised_candidates(
            sample,
            timesteps,
            context,
            coarse,
            valid_mask,
        )
        return candidates, logits, None

    def _frozen_reference_prediction(
        self,
        sample: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del valid_mask
        return self.reference.predict(
            self.planner,
            sample,
            timesteps,
            context,
            coarse,
        )

    @torch.inference_mode()
    def infer_frozen_pretrain(
        self,
        rollout: JointGRPORollout,
        *,
        noise_bundle: DDIMNoiseBundle | None = None,
    ) -> dict[str, Tensor]:
        """Run one deterministic Stage-1 policy inference on a rollout state."""

        if not isinstance(rollout, JointGRPORollout):
            raise JointGRPOError("frozen pretrain inference requires a joint rollout")
        return self._infer_frozen_pretrain_context(
            rollout.context,
            rollout.coarse_trajectories,
            rollout.mode_valid_mask,
            noise_bundle=noise_bundle,
        )

    @torch.inference_mode()
    def infer_frozen_pretrain_from_inputs(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        noise_bundle: DDIMNoiseBundle | None = None,
    ) -> dict[str, Tensor]:
        """Run deterministic Stage-1 inference without allocating an N-rollout."""

        context = self._context_from_inputs(model_inputs)
        return self._infer_frozen_pretrain_context(
            context,
            model_inputs["coarse_trajectories"],
            model_inputs["mode_valid_mask"],
            noise_bundle=noise_bundle,
        )

    def _infer_frozen_pretrain_context(
        self,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
        *,
        noise_bundle: DDIMNoiseBundle | None,
    ) -> dict[str, Tensor]:
        self.planner._validate_trajectory_inputs(
            coarse,
            valid_mask,
            batch_size=context.batch_size,
            device=context.role_tokens.device,
        )

        batch_size = context.batch_size
        device = coarse.device
        anchor_normalized = self.planner._normalize_xy(coarse[..., :2])
        bundle = (
            self.planner._inference_noise_bundle(
                anchor_normalized.shape,
                device=device,
            )
            if noise_bundle is None
            else noise_bundle
        )
        bundle.validate(anchor_normalized.shape, device=device)
        noise_timesteps = torch.full(
            (batch_size * NUM_PLATOON_ROLES,),
            DEFAULT_DDIM_PATH.initial_timestep,
            device=device,
            dtype=torch.int64,
        )
        flat_shape = (
            batch_size * NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            2,
        )
        sample = self.planner.diffusion_scheduler.add_noise(
            anchor_normalized.reshape(flat_shape),
            bundle.initial_noise.reshape(flat_shape),
            noise_timesteps,
        ).reshape_as(anchor_normalized)

        candidates: Tensor | None = None
        raw_logits: Tensor | None = None
        transition_noise_index = 0
        for timestep, previous_timestep in DEFAULT_DDIM_PATH.transitions():
            batch_timesteps = torch.full(
                (batch_size, NUM_PLATOON_ROLES),
                timestep,
                device=device,
                dtype=torch.int64,
            )
            candidates, raw_logits = self._frozen_reference_prediction(
                sample.clamp(-1.0, 1.0),
                batch_timesteps,
                context,
                coarse,
                valid_mask,
            )
            predicted_normalized = self.planner._normalize_xy(
                candidates[..., :2]
            )
            step_noise = None
            if previous_timestep >= 0:
                step_noise = bundle.transition_noises[transition_noise_index]
                transition_noise_index += 1
            transition = self.transition.step(
                model_output=predicted_normalized.reshape(flat_shape),
                timestep=timestep,
                previous_timestep=previous_timestep,
                sample=sample.reshape(flat_shape),
                eta=DEFAULT_DDIM_PATH.eta,
                noise=(
                    step_noise.reshape(flat_shape)
                    if step_noise is not None
                    else None
                ),
            )
            sample = transition.prev_sample.reshape_as(sample)
        if candidates is None or raw_logits is None:
            raise JointGRPOError(
                "frozen pretrain inference produced no denoising steps"
            )
        selected = self.planner._select(candidates, raw_logits, valid_mask)
        return {
            "selected_trajectory": selected["selected_trajectory"],
            "selected_mode": selected["selected_mode"],
            "all_mode_trajectories": candidates,
            "mode_logits": raw_logits,
        }

    def _make_rollout(
        self,
        *,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
        chains: Tensor,
        candidates: Tensor,
        frozen_candidates: Tensor,
        noise_bundle_identity: tuple[int, int, int] | None,
        risk_pact_context: RiskPACTRolloutContext | None,
        step_histories: list[Tensor],
    ) -> JointGRPORollout:
        if step_histories:
            raise JointGRPOError("variant A rollout must not contain action history")
        return JointGRPORollout(
            context=BEVPlannerContext(
                bev_feature=context.bev_feature.detach(),
                role_tokens=context.role_tokens.detach(),
            ),
            coarse_trajectories=coarse.detach(),
            mode_valid_mask=valid_mask.detach(),
            chains_normalized=chains.detach(),
            candidate_trajectories=candidates.detach(),
            frozen_candidate_trajectories=frozen_candidates.detach(),
            risk_pact_context=risk_pact_context,
            noise_bundle_identity=noise_bundle_identity,
        )

    @torch.no_grad()
    def _sample_frozen_candidates(
        self,
        *,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
        bundle: DDIMNoiseBundle,
    ) -> Tensor:
        trajectories = self.config.trajectories_per_mode
        repeated_context = _repeat_context(context, trajectories)
        repeated_coarse = _repeat_groups(coarse, trajectories)
        repeated_mask = _repeat_groups(valid_mask, trajectories)
        anchor = self.planner._normalize_xy(repeated_coarse[..., :2])
        bundle.validate(anchor.shape, device=anchor.device)
        flat_shape = (
            anchor.shape[0] * NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            2,
        )
        noise_timestep = torch.full(
            (anchor.shape[0] * NUM_PLATOON_ROLES,),
            DEFAULT_DDIM_PATH.initial_timestep,
            device=anchor.device,
            dtype=torch.int64,
        )
        sample = self.transition.add_noise(
            anchor.reshape(flat_shape),
            bundle.initial_noise.reshape(flat_shape),
            noise_timestep,
        ).reshape_as(anchor).clamp(-1.0, 1.0)
        final_candidates: Tensor | None = None
        transition_index = 0
        for timestep, previous_timestep in DEFAULT_DDIM_PATH.transitions():
            batch_timestep = torch.full(
                (anchor.shape[0], NUM_PLATOON_ROLES),
                timestep,
                device=anchor.device,
                dtype=torch.int64,
            )
            final_candidates, _ = self._frozen_reference_prediction(
                sample,
                batch_timestep,
                repeated_context,
                repeated_coarse,
                repeated_mask,
            )
            model_output = self.planner._normalize_xy(
                final_candidates[..., :2]
            ).float()
            step_noise = None
            if previous_timestep >= 0:
                step_noise = bundle.transition_noises[transition_index]
                transition_index += 1
            transition = self.transition.step(
                model_output=model_output,
                timestep=timestep,
                previous_timestep=previous_timestep,
                sample=sample.float(),
                eta=DEFAULT_DDIM_PATH.eta,
                noise=step_noise,
            )
            sample = transition.prev_sample.detach()
        if final_candidates is None:
            raise JointGRPOError("frozen paired rollout produced no decoder output")
        batch_size = context.batch_size
        return final_candidates.reshape(
            batch_size,
            trajectories,
            NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            TRAJECTORY_DIM,
        ).permute(0, 2, 3, 1, 4, 5).contiguous()

    def _sample_noise_bundle(
        self,
        anchor: Tensor,
        *,
        generator: torch.Generator,
        transition_generator: torch.Generator,
    ) -> DDIMNoiseBundle:
        """Draw the one paired DDIM noise bundle for a current rollout."""

        bundle = DDIMNoiseBundle(
            initial_noise=torch.randn(
                anchor.shape,
                device=anchor.device,
                dtype=torch.float32,
                generator=generator,
            ),
            transition_noises=tuple(
                torch.randn(
                    anchor.shape,
                    device=anchor.device,
                    dtype=torch.float32,
                    generator=transition_generator,
                )
                for _ in range(DEFAULT_DDIM_PATH.stochastic_transition_count)
            ),
        )
        bundle.validate(anchor.shape, device=anchor.device)
        return bundle

    def _sample_current_candidates(
        self,
        *,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
        bundle: DDIMNoiseBundle,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Run the current-policy DDIM path from an already-drawn noise bundle."""

        trajectories = self.config.trajectories_per_mode
        repeated_context = _repeat_context(context, trajectories)
        repeated_coarse = _repeat_groups(coarse, trajectories)
        repeated_mask = _repeat_groups(valid_mask, trajectories)
        anchor = self.planner._normalize_xy(repeated_coarse[..., :2])
        bundle.validate(anchor.shape, device=anchor.device)
        noise_timestep = torch.full(
            (anchor.shape[0] * NUM_PLATOON_ROLES,),
            DEFAULT_DDIM_PATH.initial_timestep,
            device=anchor.device,
            dtype=torch.int64,
        )
        flat_shape = (
            anchor.shape[0] * NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            2,
        )
        sample = self.transition.add_noise(
            anchor.reshape(flat_shape),
            bundle.initial_noise.reshape(flat_shape),
            noise_timestep,
        ).reshape_as(anchor).clamp(-1.0, 1.0)
        chains = [sample]
        step_histories: list[Tensor] = []
        final_candidates = final_logits = None
        for timestep, previous_timestep in DEFAULT_DDIM_PATH.transitions():
            batch_timestep = torch.full(
                (anchor.shape[0], NUM_PLATOON_ROLES),
                timestep,
                device=anchor.device,
                dtype=torch.int64,
            )
            candidates, logits, step_history = self._rollout_prediction(
                sample,
                batch_timestep,
                repeated_context,
                repeated_coarse,
                repeated_mask,
            )
            if step_history is not None:
                step_histories.append(step_history)
            model_output = self.planner._normalize_xy(candidates[..., :2]).float()
            step_noise = None
            if previous_timestep >= 0:
                step_noise = bundle.transition_noises[len(chains) - 1]
            transition = self.transition.step(
                model_output=model_output,
                timestep=timestep,
                previous_timestep=previous_timestep,
                sample=sample.float(),
                eta=DEFAULT_DDIM_PATH.eta,
                noise=step_noise,
            )
            sample = transition.prev_sample.detach()
            chains.append(sample)
            final_candidates, final_logits = candidates, logits
        if final_candidates is None or final_logits is None:
            raise JointGRPOError("joint rollout produced no decoder output")
        batch_size = context.batch_size
        chain_tensor = torch.stack(chains, dim=1).reshape(
            batch_size,
            trajectories,
            len(chains),
            NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            2,
        )
        candidate_tensor = final_candidates.reshape(
            batch_size,
            trajectories,
            NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            TRAJECTORY_DIM,
        ).permute(0, 2, 3, 1, 4, 5).contiguous()
        return chain_tensor, candidate_tensor, step_histories

    @torch.no_grad()
    def sample_current_groups(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        generator: torch.Generator,
        transition_generator: torch.Generator | None = None,
    ) -> Tensor:
        """Return current-policy N trajectories without frozen paired decoding."""

        if not isinstance(generator, torch.Generator):
            raise JointGRPOError(
                "sample_current_groups requires an explicit torch.Generator"
            )
        if transition_generator is None:
            transition_generator = generator
        if not isinstance(transition_generator, torch.Generator):
            raise JointGRPOError(
                "sample_current_groups transition_generator must be a torch.Generator"
            )
        context = self._context_from_inputs(model_inputs)
        coarse = model_inputs["coarse_trajectories"]
        valid_mask = model_inputs["mode_valid_mask"]
        self.planner._validate_trajectory_inputs(
            coarse,
            valid_mask,
            batch_size=context.batch_size,
            device=context.role_tokens.device,
        )
        repeated_coarse = _repeat_groups(coarse, self.config.trajectories_per_mode)
        anchor = self.planner._normalize_xy(repeated_coarse[..., :2])
        bundle = self._sample_noise_bundle(
            anchor,
            generator=generator,
            transition_generator=transition_generator,
        )
        _, candidates, _ = self._sample_current_candidates(
            context=context,
            coarse=coarse,
            valid_mask=valid_mask,
            bundle=bundle,
        )
        return candidates.detach()

    @torch.no_grad()
    def sample_groups(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        generator: torch.Generator,
        transition_generator: torch.Generator | None = None,
        noise_bundle_identity: tuple[int, int, int] | None = None,
    ) -> JointGRPORollout:
        if not isinstance(generator, torch.Generator):
            raise JointGRPOError("sample_groups requires an explicit torch.Generator")
        if transition_generator is None:
            transition_generator = generator
        if not isinstance(transition_generator, torch.Generator):
            raise JointGRPOError(
                "sample_groups transition_generator must be a torch.Generator"
            )
        context = self._context_from_inputs(model_inputs)
        risk_pact_context = self._risk_pact_context_from_inputs(
            model_inputs,
            batch_size=context.batch_size,
            device=context.role_tokens.device,
        )
        coarse = model_inputs["coarse_trajectories"]
        valid_mask = model_inputs["mode_valid_mask"]
        self.planner._validate_trajectory_inputs(
            coarse,
            valid_mask,
            batch_size=context.batch_size,
            device=context.role_tokens.device,
        )
        trajectories = self.config.trajectories_per_mode
        repeated_coarse = _repeat_groups(coarse, trajectories)
        anchor = self.planner._normalize_xy(repeated_coarse[..., :2])
        bundle = self._sample_noise_bundle(
            anchor,
            generator=generator,
            transition_generator=transition_generator,
        )
        chain_tensor, candidate_tensor, step_histories = self._sample_current_candidates(
            context=context,
            coarse=coarse,
            valid_mask=valid_mask,
            bundle=bundle,
        )
        frozen_candidate_tensor = self._sample_frozen_candidates(
            context=context,
            coarse=coarse,
            valid_mask=valid_mask,
            bundle=bundle,
        )
        return self._make_rollout(
            context=context,
            coarse=coarse,
            valid_mask=valid_mask,
            chains=chain_tensor,
            candidates=candidate_tensor,
            frozen_candidates=frozen_candidate_tensor,
            noise_bundle_identity=noise_bundle_identity,
            risk_pact_context=risk_pact_context,
            step_histories=step_histories,
        )

    def _replay_history(
        self,
        rollout: JointGRPORollout,
        *,
        step_index: int,
        flat_count: int,
    ) -> Tensor | None:
        del rollout, step_index, flat_count
        return None

    def _replay_predictions(
        self,
        *,
        sample: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
        predecessor_history: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if predecessor_history is not None:
            raise JointGRPOError("variant A replay must not receive action history")
        current_candidates, current_logits = (
            self.planner.predict_denoised_candidates(
                sample,
                timesteps,
                context,
                coarse,
                valid_mask,
            )
        )
        with torch.no_grad():
            reference_candidates, reference_logits = self.reference.predict(
                self.planner,
                sample,
                timesteps,
                context,
                coarse,
            )
        return (
            current_candidates,
            current_logits,
            reference_candidates,
            reference_logits,
        )

    def compute_loss(
        self,
        rollout: JointGRPORollout,
    ) -> JointGRPOLossResult:
        signals = (
            rollout.current_rewards,
            rollout.frozen_rewards,
            rollout.collision_mask,
            rollout.out_of_drivable_mask,
            rollout.valid_executable_mode_mask,
        )
        if any(value is None for value in signals):
            raise JointGRPOError("rollout is missing paired reward signals")
        centered, advantages, signal_mode_mask = fixed_scale_safe_advantages(
            rollout.current_rewards,
            rollout.frozen_rewards,
            rollout.collision_mask,
            rollout.out_of_drivable_mask,
            rollout.valid_executable_mode_mask,
            trajectories_per_mode=self.config.trajectories_per_mode,
            advantage_scale=self.config.advantage_scale,
            baseline_tolerance=self.config.baseline_tolerance,
        )
        return self._compute_loss_from_advantages(
            rollout,
            centered.detach(),
            advantages.detach(),
            signal_mode_mask,
            rollout.valid_executable_mode_mask,
        )

    def _compute_loss_from_advantages(
        self,
        rollout: JointGRPORollout,
        centered_rewards: Tensor,
        advantages: Tensor,
        signal_mode_mask: Tensor,
        valid_executable_mode_mask: Tensor,
        *,
        include_risk_pact_teacher: bool = True,
    ) -> JointGRPOLossResult:
        batch_size = rollout.batch_size
        trajectories = rollout.trajectories_per_mode
        expected_advantage = (
            batch_size,
            NUM_PLATOON_ROLES,
            NUM_MODES,
            trajectories,
        )
        if tuple(advantages.shape) != expected_advantage:
            raise JointGRPOError("reward batch does not match rollout")
        expected_mask = expected_advantage[:3]
        if (
            signal_mode_mask.dtype != torch.bool
            or tuple(signal_mode_mask.shape) != expected_mask
            or signal_mode_mask.device != advantages.device
            or valid_executable_mode_mask.dtype != torch.bool
            or tuple(valid_executable_mode_mask.shape) != expected_mask
            or valid_executable_mode_mask.device != advantages.device
        ):
            raise JointGRPOError(
                "signal and valid-executable masks must be colocated bool [B,3,10]"
            )
        if tuple(centered_rewards.shape) != expected_advantage:
            raise JointGRPOError("centered reward batch does not match rollout")
        valid_mode_mask = rollout.mode_valid_mask.to(signal_mode_mask.device)
        if bool((valid_executable_mode_mask & ~valid_mode_mask).any()):
            raise JointGRPOError(
                "valid-executable modes must be a subset of hard-valid modes"
            )
        if bool((signal_mode_mask & ~valid_executable_mode_mask).any()):
            raise JointGRPOError("signal modes must be valid and executable")
        stochastic_steps = len(self.config.stochastic_timesteps)
        if tuple(rollout.candidate_trajectories.shape) != (
            *expected_advantage,
            TRAJECTORY_STEPS,
            TRAJECTORY_DIM,
        ):
            raise JointGRPOError("rollout candidate trajectory shape is invalid")
        if tuple(rollout.frozen_candidate_trajectories.shape) != tuple(
            rollout.candidate_trajectories.shape
        ):
            raise JointGRPOError("paired frozen candidates do not match current shape")
        flat_count = batch_size * trajectories
        context = _repeat_context(rollout.context, trajectories)
        coarse = _repeat_groups(rollout.coarse_trajectories, trajectories)
        valid_mask = _repeat_groups(rollout.mode_valid_mask, trajectories)
        chains = rollout.chains_normalized.reshape(
            flat_count,
            len(self.config.roll_timesteps) + 1,
            NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            2,
        )

        current_log_probs = []
        trajectory_kls = []
        feasibility_step_losses: list[Tensor] = []
        final_current_candidates = final_reference_candidates = None
        for step_index, (timestep, previous_timestep) in enumerate(
            DEFAULT_DDIM_PATH.transitions()
        ):
            sample = chains[:, step_index]
            next_sample = chains[:, step_index + 1]
            batch_timestep = torch.full(
                (flat_count, NUM_PLATOON_ROLES),
                timestep,
                device=sample.device,
                dtype=torch.int64,
            )
            predecessor_history = self._replay_history(
                rollout,
                step_index=step_index,
                flat_count=flat_count,
            )
            (
                current_candidates,
                _,
                reference_candidates,
                _,
            ) = self._replay_predictions(
                sample=sample,
                timesteps=batch_timestep,
                context=context,
                coarse=coarse,
                valid_mask=valid_mask,
                predecessor_history=predecessor_history,
            )
            if self.config.uses_feasibility:
                steering_limit_deg = time_varying_limit(
                    timestep=timestep,
                    initial_timestep=DEFAULT_DDIM_PATH.initial_timestep,
                    initial_limit=self.config.steering_initial_limit_deg,
                    final_limit=self.config.steering_final_limit_deg,
                    schedule_power=self.config.feasibility_schedule_power,
                )
                steering_rate_limit_deg_s = time_varying_limit(
                    timestep=timestep,
                    initial_timestep=DEFAULT_DDIM_PATH.initial_timestep,
                    initial_limit=self.config.steering_rate_initial_limit_deg_s,
                    final_limit=self.config.steering_rate_final_limit_deg_s,
                    schedule_power=self.config.feasibility_schedule_power,
                )
                fea = steering_feasibility_loss(
                    current_candidates[..., :2],
                    steering_limit_deg=steering_limit_deg,
                    steering_rate_limit_deg_s=steering_rate_limit_deg_s,
                    wheelbase_m=self.config.wheelbase_m,
                    trajectory_dt_s=self.config.trajectory_dt_s,
                    min_segment_length_m=self.config.min_segment_length_m,
                )
                def _fea_blocks(value: Tensor) -> Tensor:
                    return value.reshape(
                        batch_size, trajectories, NUM_PLATOON_ROLES, NUM_MODES
                    ).permute(0, 2, 3, 1).contiguous()
                steering_loss = _hierarchical_active_mean(
                    _fea_blocks(fea.steering_loss), valid_executable_mode_mask
                )
                steering_rate_loss = _hierarchical_active_mean(
                    _fea_blocks(fea.steering_rate_loss), valid_executable_mode_mask
                )
                feasibility_step_losses.append(
                    float(self.config.steering_feasibility_weight) * steering_loss
                    + float(self.config.steering_rate_feasibility_weight) * steering_rate_loss
                )
            else:
                feasibility_step_losses.append(current_candidates.sum() * 0.0)

            current_output = self.planner._normalize_xy(
                current_candidates[..., :2]
            ).float()
            reference_output = self.planner._normalize_xy(
                reference_candidates[..., :2]
            ).float()
            current_transition = self.transition.step(
                model_output=current_output,
                timestep=timestep,
                previous_timestep=previous_timestep,
                sample=sample.float(),
                eta=DEFAULT_DDIM_PATH.eta,
                prev_sample=next_sample.float(),
            )
            with torch.no_grad():
                reference_transition = self.transition.step(
                    model_output=reference_output,
                    timestep=timestep,
                    previous_timestep=previous_timestep,
                    sample=sample.float(),
                    eta=DEFAULT_DDIM_PATH.eta,
                    prev_sample=next_sample.float(),
            )
            if current_transition.log_prob is not None:
                current_log_probs.append(current_transition.log_prob)
                sigma_squared = current_transition.std.square()
                trajectory_kls.append(
                    (
                        (
                            current_transition.mean
                            - reference_transition.mean
                        ).square()
                        / (2.0 * sigma_squared)
                    ).mean(dim=(-2, -1))
                )
            final_current_candidates = current_candidates
            final_reference_candidates = reference_candidates

        if (
            final_current_candidates is None
            or final_reference_candidates is None
        ):
            raise JointGRPOError("joint replay produced no decoder output")
        new_trajectory_log_prob = torch.stack(
            current_log_probs, dim=-1
        ).reshape(
            batch_size,
            trajectories,
            NUM_PLATOON_ROLES,
            NUM_MODES,
            stochastic_steps,
        ).permute(0, 2, 3, 1, 4).contiguous()
        score_function_weight = torch.exp(
            new_trajectory_log_prob - new_trajectory_log_prob.detach()
        )
        trajectory_pg_blocks = (
            -score_function_weight
            * advantages.unsqueeze(-1).expand_as(score_function_weight)
        )
        trajectory_pg = _hierarchical_active_mean(
            trajectory_pg_blocks,
            signal_mode_mask,
        )
        trajectory_pg_by_mode = _per_mode_masked_mean(
            trajectory_pg_blocks,
            signal_mode_mask,
        )

        current_all_modes = final_current_candidates.reshape(
            batch_size,
            trajectories,
            NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            TRAJECTORY_DIM,
        ).permute(0, 2, 3, 1, 4, 5)
        reference_all_modes = final_reference_candidates.reshape(
            batch_size,
            trajectories,
            NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            TRAJECTORY_DIM,
        ).permute(0, 2, 3, 1, 4, 5)
        behavior_cloning_blocks = _trajectory_bc_blocks(
            current_all_modes,
            reference_all_modes,
            self.config,
        )
        behavior_cloning = _hierarchical_active_mean(
            behavior_cloning_blocks,
            valid_executable_mode_mask,
        )
        behavior_cloning_by_mode = _per_mode_masked_mean(
            behavior_cloning_blocks,
            valid_executable_mode_mask,
        )
        trajectory_kl_blocks = torch.stack(
            trajectory_kls, dim=-1
        ).reshape(
            batch_size,
            trajectories,
            NUM_PLATOON_ROLES,
            NUM_MODES,
            stochastic_steps,
        ).permute(0, 2, 3, 1, 4)
        trajectory_reference_kl = _hierarchical_active_mean(
            trajectory_kl_blocks,
            valid_executable_mode_mask,
        )
        reference_kl_by_mode = _per_mode_masked_mean(
            trajectory_kl_blocks,
            valid_executable_mode_mask,
        )
        reference_kl = trajectory_reference_kl

        feasibility = torch.stack(feasibility_step_losses).mean()
        weighted_feasibility = (
            float(self.config.feasibility_weight) * feasibility
            if self.config.uses_feasibility
            else feasibility * 0.0
        )

        # PACT-lite deliberately projects the *on-policy rollout* final x0, not
        # the frozen Stage-1 reference.  The student replay is generated from
        # the exact stored DDIM chain, preserving timestep/noise/mode identity.
        zero_aux = current_all_modes.sum() * 0.0
        risk_pact_distillation = zero_aux
        weighted_risk_pact_distillation = zero_aux
        risk_pact_active_ratio = zero_aux.detach()
        risk_pact_active_count = zero_aux.detach()
        risk_pact_teacher_displacement_mean_m = zero_aux.detach()
        risk_pact_teacher_displacement_max_m = zero_aux.detach()
        risk_pact_trajectory_risk_mean = zero_aux.detach()
        risk_pact_trajectory_risk_max = zero_aux.detach()
        risk_pact_violation_mean = zero_aux.detach()
        risk_pact_curriculum_scale = zero_aux.detach()
        risk_pact_background_risk_mean = zero_aux.detach()
        risk_pact_platoon_risk_mean = zero_aux.detach()
        risk_pact_road_risk_mean = zero_aux.detach()
        risk_pact_road_signed_distance_min_m = zero_aux.detach()
        risk_pact_valid_active_ratio = zero_aux.detach()
        risk_pact_valid_active_count = zero_aux.detach()
        risk_pact_valid_candidate_count = zero_aux.detach()
        risk_pact_teacher_trajectory_risk_mean = zero_aux.detach()
        risk_pact_teacher_trajectory_risk_max = zero_aux.detach()
        risk_pact_teacher_risk_reduction_mean = zero_aux.detach()
        risk_pact_teacher_improvement_fraction = zero_aux.detach()
        risk_pact_critical_time_mean_s = zero_aux.detach()
        risk_pact_critical_time_max_s = zero_aux.detach()
        risk_pact_late_horizon_critical_fraction = zero_aux.detach()
        risk_pact_background_signed_clearance_min_mean_m = zero_aux.detach()
        risk_pact_background_signed_clearance_min_m = zero_aux.detach()
        risk_pact_platoon_signed_clearance_min_mean_m = zero_aux.detach()
        risk_pact_platoon_signed_clearance_min_m = zero_aux.detach()
        risk_pact_road_safety_clearance_min_mean_m = zero_aux.detach()
        risk_pact_road_safety_clearance_min_m = zero_aux.detach()

        if self.config.uses_risk_pact and include_risk_pact_teacher:
            actor = rollout.require_risk_pact_context()
            old_all_modes = rollout.candidate_trajectories.to(current_all_modes.device)
            if tuple(old_all_modes.shape) != tuple(current_all_modes.shape):
                raise JointGRPOError(
                    "Risk-PACT old/student candidate shapes must match exactly"
                )
            # [B,R,M,N,H,D] -> [B,R,M*N,H,D], preserving identical flatten order
            old_flat = old_all_modes.reshape(
                batch_size, NUM_PLATOON_ROLES, NUM_MODES * trajectories,
                TRAJECTORY_STEPS, TRAJECTORY_DIM,
            )
            student_flat = current_all_modes.reshape(
                batch_size, NUM_PLATOON_ROLES, NUM_MODES * trajectories,
                TRAJECTORY_STEPS, TRAJECTORY_DIM,
            )
            risk_cfg = RiskPACTConfig(
                use_background_actor=self.config.risk_pact_use_background_actor,
                use_platoon_actor=self.config.risk_pact_use_platoon_actor,
                use_road_boundary=self.config.risk_pact_use_road_boundary,
                horizon_dt_s=self.config.risk_pact_horizon_dt_s,
                ego_length_m=self.config.risk_pact_ego_length_m,
                ego_width_m=self.config.risk_pact_ego_width_m,
                background_longitudinal_clearance_m=self.config.risk_pact_background_longitudinal_clearance_m,
                background_lateral_clearance_m=self.config.risk_pact_background_lateral_clearance_m,
                actor_temperature_m=self.config.risk_pact_actor_temperature_m,
                actor_softmax_beta=self.config.risk_pact_actor_softmax_beta,
                platoon_vehicle_length_m=self.config.risk_pact_platoon_vehicle_length_m,
                platoon_vehicle_width_m=self.config.risk_pact_platoon_vehicle_width_m,
                platoon_longitudinal_clearance_m=self.config.risk_pact_platoon_longitudinal_clearance_m,
                platoon_lateral_clearance_m=self.config.risk_pact_platoon_lateral_clearance_m,
                component_softmax_beta=self.config.risk_pact_component_softmax_beta,
                road_safety_margin_m=self.config.risk_pact_road_safety_margin_m,
                road_temperature_m=self.config.risk_pact_road_temperature_m,
                temporal_softmax_beta=self.config.risk_pact_temporal_softmax_beta,
                risk_threshold=self.config.risk_pact_risk_threshold,
                violation_temperature=self.config.risk_pact_violation_temperature,
                safe_margin=self.config.risk_pact_safe_margin,
                teacher_step_m=self.config.risk_pact_teacher_step_m,
                gradient_eps=self.config.risk_pact_gradient_eps,
                gradient_clip_norm=self.config.risk_pact_gradient_clip_norm,
            )
            class _CurriculumConfig:
                start_scale = self.config.risk_pact_curriculum_start_scale
                end_scale = self.config.risk_pact_curriculum_end_scale
                warmup_updates = self.config.risk_pact_curriculum_warmup_updates
                ramp_updates = self.config.risk_pact_curriculum_ramp_updates
                schedule = self.config.risk_pact_curriculum_schedule
            curriculum = risk_pact_curriculum_state(self.optimizer_step, _CurriculumConfig())
            platoon_actor_state, platoon_actor_valid = build_platoon_actor_state(
                actor.ego_state.to(current_all_modes.device),
                actor.formation_relation_state.to(current_all_modes.device),
                actor.relation_valid_mask.to(current_all_modes.device),
                config=risk_cfg,
            )
            teacher = build_x0_pact_teacher(
                old_flat,
                actor.background_actor_state.to(current_all_modes.device),
                actor.background_actor_valid_mask.to(current_all_modes.device),
                platoon_actor_state=platoon_actor_state,
                platoon_actor_valid_mask=platoon_actor_valid,
                road_sdf=(
                    None
                    if actor.drivable_sdf is None
                    else actor.drivable_sdf.to(current_all_modes.device)
                ),
                curriculum_scale=curriculum.scale,
                config=risk_cfg,
                evaluate_teacher=self.config.risk_pact_diagnostics_enabled,
            )
            valid_flat = valid_executable_mode_mask.unsqueeze(-1).expand(
                batch_size, NUM_PLATOON_ROLES, NUM_MODES, trajectories
            ).reshape(batch_size, NUM_PLATOON_ROLES, NUM_MODES * trajectories)
            active_flat = teacher.constraint.near_or_unsafe_mask & valid_flat
            teacher = replace(
                teacher,
                constraint=replace(
                    teacher.constraint,
                    near_or_unsafe_mask=active_flat,
                    safe_mask=teacher.constraint.safe_mask | ~valid_flat,
                    violation=torch.where(
                        valid_flat,
                        teacher.constraint.violation,
                        torch.zeros_like(teacher.constraint.violation),
                    ),
                ),
            )
            pact = pact_lite_distillation_loss(student_flat, teacher)
            risk_pact_distillation = pact.loss
            weighted_risk_pact_distillation = (
                float(self.config.risk_pact_distill_weight) * pact.loss
            )
            risk_pact_active_ratio = pact.active_ratio.detach()
            risk_pact_active_count = pact.active_count.detach()
            risk_pact_teacher_displacement_mean_m = pact.teacher_displacement_mean_m.detach()
            risk_pact_teacher_displacement_max_m = pact.teacher_displacement_max_m.detach()
            valid_weight = valid_flat.to(teacher.constraint.trajectory_risk.dtype)
            valid_count = valid_weight.sum().clamp_min(1.0)
            risk_pact_trajectory_risk_mean = (
                teacher.constraint.trajectory_risk * valid_weight
            ).sum().div(valid_count).detach()
            masked_risk = torch.where(
                valid_flat,
                teacher.constraint.trajectory_risk,
                torch.full_like(teacher.constraint.trajectory_risk, float("-inf")),
            )
            risk_pact_trajectory_risk_max = torch.where(
                torch.isfinite(masked_risk.max()),
                masked_risk.max(),
                masked_risk.new_zeros(()),
            ).detach()
            risk_pact_violation_mean = (
                teacher.constraint.violation * valid_weight
            ).sum().div(valid_count).detach()
            risk_pact_curriculum_scale = current_all_modes.new_tensor(curriculum.scale).detach()

            valid_time_weight = valid_weight.unsqueeze(-1)
            valid_time_count = (valid_count * float(TRAJECTORY_STEPS)).clamp_min(1.0)

            def _component_mean(component: Tensor | None) -> Tensor:
                if component is None:
                    return current_all_modes.new_zeros(()).detach()
                return (component * valid_time_weight).sum().div(valid_time_count).detach()

            risk_pact_background_risk_mean = _component_mean(teacher.field.background_risk)
            risk_pact_platoon_risk_mean = _component_mean(teacher.field.platoon_risk)
            risk_pact_road_risk_mean = _component_mean(teacher.field.road_risk)
            if teacher.field.road_signed_distance_m is not None:
                road_distance = torch.where(
                    valid_flat.unsqueeze(-1),
                    teacher.field.road_signed_distance_m,
                    torch.full_like(teacher.field.road_signed_distance_m, float("inf")),
                )
                road_min = road_distance.min()
                risk_pact_road_signed_distance_min_m = torch.where(
                    torch.isfinite(road_min),
                    road_min,
                    road_min.new_zeros(()),
                ).detach()

            # Step-6 diagnostics. These are detached scalar summaries and do not
            # change the optimized objective or the teacher target.
            if self.config.risk_pact_diagnostics_enabled:
                raw_valid_count = valid_weight.sum()
                risk_pact_valid_candidate_count = raw_valid_count.detach()
                active_weight = active_flat.to(valid_weight.dtype)
                raw_active_count = active_weight.sum()
                risk_pact_valid_active_count = raw_active_count.detach()
                risk_pact_valid_active_ratio = (
                    raw_active_count / raw_valid_count.clamp_min(1.0)
                ).detach()

                projected_constraint = teacher.teacher_constraint
                if projected_constraint is None:
                    raise JointGRPOError("Risk-PACT Step-6 teacher diagnostics are missing")
                projected_risk = projected_constraint.trajectory_risk
                risk_pact_teacher_trajectory_risk_mean = (
                    projected_risk * valid_weight
                ).sum().div(valid_count).detach()
                projected_masked = torch.where(
                    valid_flat,
                    projected_risk,
                    torch.full_like(projected_risk, float("-inf")),
                )
                projected_max = projected_masked.max()
                risk_pact_teacher_trajectory_risk_max = torch.where(
                    torch.isfinite(projected_max),
                    projected_max,
                    projected_max.new_zeros(()),
                ).detach()

                risk_reduction = teacher.constraint.trajectory_risk - projected_risk
                active_count_safe = raw_active_count.clamp_min(1.0)
                risk_pact_teacher_risk_reduction_mean = (
                    risk_reduction * active_weight
                ).sum().div(active_count_safe).detach()
                improved = (
                    risk_reduction > float(self.config.risk_pact_teacher_improvement_eps)
                ) & active_flat
                risk_pact_teacher_improvement_fraction = (
                    improved.to(valid_weight.dtype).sum().div(active_count_safe)
                ).detach()

                critical_index = teacher.constraint.critical_timestep_index
                if critical_index is None:
                    critical_index = teacher.field.risk.argmax(dim=-1)
                critical_time = (critical_index.to(valid_weight.dtype) + 1.0) * float(
                    self.config.risk_pact_horizon_dt_s
                )
                if bool(active_flat.any()):
                    active_critical = critical_time[active_flat]
                    risk_pact_critical_time_mean_s = active_critical.mean().detach()
                    risk_pact_critical_time_max_s = active_critical.max().detach()
                    late = critical_index >= int(self.config.risk_pact_late_horizon_start_step)
                    risk_pact_late_horizon_critical_fraction = (
                        (late & active_flat).to(valid_weight.dtype).sum().div(active_count_safe)
                    ).detach()

                def _signed_clearance_stats(value: Tensor | None) -> tuple[Tensor, Tensor]:
                    zero = current_all_modes.new_zeros(()).detach()
                    if value is None:
                        return zero, zero
                    per_trajectory = value.amin(dim=(-1, -2))
                    finite_valid = valid_flat & torch.isfinite(per_trajectory)
                    if not bool(finite_valid.any()):
                        return zero, zero
                    selected = per_trajectory[finite_valid]
                    return selected.mean().detach(), selected.min().detach()

                (
                    risk_pact_background_signed_clearance_min_mean_m,
                    risk_pact_background_signed_clearance_min_m,
                ) = _signed_clearance_stats(teacher.field.background_signed_clearance_m)
                (
                    risk_pact_platoon_signed_clearance_min_mean_m,
                    risk_pact_platoon_signed_clearance_min_m,
                ) = _signed_clearance_stats(teacher.field.platoon_signed_clearance_m)
                if teacher.field.road_signed_distance_m is not None:
                    required_road_clearance = (
                        0.5 * float(self.config.risk_pact_ego_width_m)
                        + float(self.config.risk_pact_road_safety_margin_m)
                    )
                    road_safety_clearance = (
                        teacher.field.road_signed_distance_m - required_road_clearance
                    )
                    per_trajectory_road = road_safety_clearance.amin(dim=-1)
                    finite_valid_road = valid_flat & torch.isfinite(per_trajectory_road)
                    if bool(finite_valid_road.any()):
                        selected_road = per_trajectory_road[finite_valid_road]
                        risk_pact_road_safety_clearance_min_mean_m = (
                            selected_road.mean().detach()
                        )
                        risk_pact_road_safety_clearance_min_m = selected_road.min().detach()

        total = (
            float(self.config.trajectory_pg_weight) * trajectory_pg
            + float(self.config.bc_weight) * behavior_cloning
            + float(self.config.reference_kl_weight) * reference_kl
            + weighted_feasibility
            + weighted_risk_pact_distillation
        )
        tensors = (
            total,
            trajectory_pg,
            behavior_cloning,
            trajectory_reference_kl,
            reference_kl,
            new_trajectory_log_prob,
            centered_rewards,
            advantages,
            feasibility,
            weighted_feasibility,
            risk_pact_distillation,
            weighted_risk_pact_distillation,
        )
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise JointGRPOError("joint GRPO loss contains non-finite values")
        if bool((trajectory_reference_kl < -1e-7).item()):
            raise JointGRPOError("reference KL must be non-negative")
        return JointGRPOLossResult(
            total=total,
            trajectory_pg=trajectory_pg,
            behavior_cloning=behavior_cloning,
            trajectory_reference_kl=trajectory_reference_kl,
            reference_kl=reference_kl,
            centered_rewards=centered_rewards,
            advantages=advantages,
            signal_mode_mask=signal_mode_mask,
            valid_executable_mode_mask=valid_executable_mode_mask,
            new_trajectory_log_prob=new_trajectory_log_prob,
            trajectory_pg_by_mode=trajectory_pg_by_mode,
            behavior_cloning_by_mode=behavior_cloning_by_mode,
            reference_kl_by_mode=reference_kl_by_mode,
            feasibility=feasibility,
            weighted_feasibility=weighted_feasibility,
            risk_pact_distillation=risk_pact_distillation,
            weighted_risk_pact_distillation=weighted_risk_pact_distillation,
            risk_pact_active_ratio=risk_pact_active_ratio,
            risk_pact_active_count=risk_pact_active_count,
            risk_pact_teacher_displacement_mean_m=risk_pact_teacher_displacement_mean_m,
            risk_pact_teacher_displacement_max_m=risk_pact_teacher_displacement_max_m,
            risk_pact_trajectory_risk_mean=risk_pact_trajectory_risk_mean,
            risk_pact_trajectory_risk_max=risk_pact_trajectory_risk_max,
            risk_pact_violation_mean=risk_pact_violation_mean,
            risk_pact_curriculum_scale=risk_pact_curriculum_scale,
            risk_pact_background_risk_mean=risk_pact_background_risk_mean,
            risk_pact_platoon_risk_mean=risk_pact_platoon_risk_mean,
            risk_pact_road_risk_mean=risk_pact_road_risk_mean,
            risk_pact_road_signed_distance_min_m=risk_pact_road_signed_distance_min_m,
            risk_pact_valid_active_ratio=risk_pact_valid_active_ratio,
            risk_pact_valid_active_count=risk_pact_valid_active_count,
            risk_pact_valid_candidate_count=risk_pact_valid_candidate_count,
            risk_pact_teacher_trajectory_risk_mean=risk_pact_teacher_trajectory_risk_mean,
            risk_pact_teacher_trajectory_risk_max=risk_pact_teacher_trajectory_risk_max,
            risk_pact_teacher_risk_reduction_mean=risk_pact_teacher_risk_reduction_mean,
            risk_pact_teacher_improvement_fraction=risk_pact_teacher_improvement_fraction,
            risk_pact_critical_time_mean_s=risk_pact_critical_time_mean_s,
            risk_pact_critical_time_max_s=risk_pact_critical_time_max_s,
            risk_pact_late_horizon_critical_fraction=risk_pact_late_horizon_critical_fraction,
            risk_pact_background_signed_clearance_min_mean_m=risk_pact_background_signed_clearance_min_mean_m,
            risk_pact_background_signed_clearance_min_m=risk_pact_background_signed_clearance_min_m,
            risk_pact_platoon_signed_clearance_min_mean_m=risk_pact_platoon_signed_clearance_min_mean_m,
            risk_pact_platoon_signed_clearance_min_m=risk_pact_platoon_signed_clearance_min_m,
            risk_pact_road_safety_clearance_min_mean_m=risk_pact_road_safety_clearance_min_mean_m,
            risk_pact_road_safety_clearance_min_m=risk_pact_road_safety_clearance_min_m,
        )

    @staticmethod
    def _module_gradient_norm(module: nn.Module) -> float:
        values = []
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            if not bool(torch.isfinite(gradient).all()):
                raise JointGRPOError("joint GRPO gradient is non-finite")
            values.append(gradient.square().sum())
        if not values:
            return 0.0
        return float(torch.stack(values).sum().sqrt().cpu())

    def _residual_gradient_norms(
        self, head: ModeResidualTrajectoryHead
    ) -> dict[str, float]:
        weight, bias = head.residual_parameters()
        values = {"mode_residual": self._module_gradient_norm(head)}
        for mode in range(NUM_MODES):
            squared = torch.zeros((), device=weight.device, dtype=torch.float32)
            if weight.grad is not None:
                squared = squared + weight.grad[mode].detach().float().square().sum()
            if bias.grad is not None:
                squared = squared + bias.grad[mode].detach().float().square().sum()
            if not bool(torch.isfinite(squared)):
                raise JointGRPOError("joint GRPO gradient is non-finite")
            values[f"mode_{mode}"] = float(torch.sqrt(squared).cpu())
        return values

    def _residual_head(self) -> ModeResidualTrajectoryHead:
        head = self.planner.diffusion_decoder.trajectory_head
        if not isinstance(head, ModeResidualTrajectoryHead):
            raise JointGRPOError("trainer trajectory head lost its residual contract")
        return head

    @staticmethod
    def _objective_gradient_vector(loss_value: Tensor, parameters: list[nn.Parameter]) -> Tensor:
        """Return a detached flat gradient vector without touching ``parameter.grad``."""
        gradients = torch.autograd.grad(
            loss_value,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        pieces: list[Tensor] = []
        for parameter, gradient in zip(parameters, gradients):
            if gradient is None:
                pieces.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1))
            else:
                detached = gradient.detach().float()
                if not bool(torch.isfinite(detached).all()):
                    raise JointGRPOError("objective-gradient diagnostic is non-finite")
                pieces.append(detached.reshape(-1))
        if not pieces:
            return loss_value.new_zeros((0,), dtype=torch.float32)
        return torch.cat(pieces, dim=0)

    def _risk_pact_objective_gradient_diagnostics(
        self,
        loss: JointGRPOLossResult,
        parameters: list[nn.Parameter],
    ) -> dict[str, float]:
        diagnostics = {"gradient_objective/computed": 0.0}
        if not (
            self.config.uses_risk_pact
            and self.config.risk_pact_diagnostics_enabled
            and self.config.risk_pact_gradient_diagnostics
            and self.optimizer_step % int(self.config.risk_pact_gradient_diagnostics_interval) == 0
            and float(loss.risk_pact_active_count.detach().cpu()) > 0.0
        ):
            return diagnostics

        pact_objective = loss.weighted_risk_pact_distillation
        base_objective = loss.total - pact_objective
        base_gradient = self._objective_gradient_vector(base_objective, parameters)
        pact_gradient = self._objective_gradient_vector(pact_objective, parameters)
        base_norm = torch.linalg.vector_norm(base_gradient)
        pact_norm = torch.linalg.vector_norm(pact_gradient)
        denominator = (base_norm * pact_norm).clamp_min(1.0e-12)
        cosine = torch.dot(base_gradient, pact_gradient) / denominator
        if float(base_norm) <= 1.0e-12 or float(pact_norm) <= 1.0e-12:
            cosine = cosine.new_zeros(())
        diagnostics.update(
            {
                "gradient_objective/computed": 1.0,
                "gradient_objective/base_norm": float(base_norm.cpu()),
                "gradient_objective/risk_pact_norm": float(pact_norm.cpu()),
                "gradient_objective/risk_pact_to_base_ratio": float(
                    (pact_norm / base_norm.clamp_min(1.0e-12)).cpu()
                ),
                "gradient_objective/base_risk_pact_cosine": float(cosine.cpu()),
            }
        )
        return diagnostics

    def update(
        self,
        rollout: JointGRPORollout,
    ) -> JointGRPOUpdateResult:
        rollout_id = id(rollout)
        if self._consumed_rollouts.get(rollout_id) is rollout:
            raise JointGRPOError(
                "each live joint rollout may enter update exactly once"
            )
        signals = (
            rollout.current_rewards,
            rollout.frozen_rewards,
            rollout.collision_mask,
            rollout.out_of_drivable_mask,
            rollout.valid_executable_mode_mask,
        )
        if any(value is None for value in signals):
            raise JointGRPOError("rollout is missing paired reward signals")
        centered, advantages, signal_mode_mask = fixed_scale_safe_advantages(
            rollout.current_rewards,
            rollout.frozen_rewards,
            rollout.collision_mask,
            rollout.out_of_drivable_mask,
            rollout.valid_executable_mode_mask,
            trajectories_per_mode=self.config.trajectories_per_mode,
            advantage_scale=self.config.advantage_scale,
            baseline_tolerance=self.config.baseline_tolerance,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._compute_loss_from_advantages(
            rollout,
            centered.detach(),
            advantages.detach(),
            signal_mode_mask,
            rollout.valid_executable_mode_mask,
        )
        # Register before any mutation. A rejected or zero-signal rollout is
        # still an on-policy sample and must never be consumed twice.
        self._consumed_rollouts[rollout_id] = rollout
        head = self._residual_head()
        trainable = list(head.residual_parameters())
        objective_gradient_diagnostics = self._risk_pact_objective_gradient_diagnostics(
            loss, trainable
        )
        auxiliary_signal = (
            (self.config.uses_feasibility and float(loss.feasibility.detach().cpu()) > 0.0)
            or (
                self.config.uses_risk_pact
                and float(loss.risk_pact_active_count.detach().cpu()) > 0.0
                and float(loss.risk_pact_distillation.detach().cpu()) > 0.0
            )
        )
        if not bool(signal_mode_mask.any()) and not auxiliary_signal:
            zero_gradients = {"mode_residual": 0.0}
            zero_gradients.update(
                {f"mode_{mode}": 0.0 for mode in range(NUM_MODES)}
            )
            return JointGRPOUpdateResult(
                loss=loss.detached(),
                gradient_norms=zero_gradients,
                clipped_gradient_norms=dict(zero_gradients),
                total_gradient_norm=0.0,
                adapter_relative_drifts=head.adapter_relative_drifts(),
                post_update_reference_kl=float(loss.reference_kl.detach().cpu()),
                optimizer_step=self.optimizer_step,
                zero_signal=True,
                stability_guard_rejected=False,
                stability_guard_trigger_modes=(),
                objective_gradient_diagnostics=objective_gradient_diagnostics,
            )

        parameter_snapshot = tuple(
            parameter.detach().clone() for parameter in trainable
        )
        optimizer_snapshot = copy.deepcopy(self.optimizer.state_dict())
        loss.total.backward()
        gradient_norms = self._residual_gradient_norms(head)
        total_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(self.config.max_grad_norm)
        )
        if not bool(torch.isfinite(total_norm)):
            raise JointGRPOError("joint GRPO total gradient norm is non-finite")
        total_gradient_norm = float(total_norm.detach().cpu())
        clipped_gradient_norms = self._residual_gradient_norms(head)
        zero_signal = gradient_norms["mode_residual"] == 0.0
        if zero_signal:
            return JointGRPOUpdateResult(
                loss=loss.detached(),
                gradient_norms=gradient_norms,
                clipped_gradient_norms=clipped_gradient_norms,
                total_gradient_norm=total_gradient_norm,
                adapter_relative_drifts=head.adapter_relative_drifts(),
                post_update_reference_kl=float(loss.reference_kl.detach().cpu()),
                optimizer_step=self.optimizer_step,
                zero_signal=True,
                stability_guard_rejected=False,
                stability_guard_trigger_modes=(),
                objective_gradient_diagnostics=objective_gradient_diagnostics,
            )

        self.optimizer.step()
        with torch.no_grad():
            post_loss = self._compute_loss_from_advantages(
                rollout,
                centered.detach(),
                advantages.detach(),
                signal_mode_mask,
                rollout.valid_executable_mode_mask,
                include_risk_pact_teacher=False,
            )
        post_kl = float(post_loss.reference_kl.detach().cpu())
        drifts = head.adapter_relative_drifts()
        drift_modes = tuple(
            mode
            for mode, drift in enumerate(drifts)
            if drift > float(self.config.max_adapter_relative_drift)
        )
        kl_rejected = post_kl > float(self.config.post_update_reference_kl_max)
        guard_rejected = kl_rejected or bool(drift_modes)
        if guard_rejected:
            with torch.no_grad():
                for parameter, saved in zip(trainable, parameter_snapshot):
                    parameter.copy_(saved)
            self.optimizer.load_state_dict(optimizer_snapshot)
            self.optimizer.zero_grad(set_to_none=True)
            trigger_modes = drift_modes
            if kl_rejected and not trigger_modes:
                trigger_modes = tuple(
                    int(mode)
                    for mode in torch.where(signal_mode_mask.any(dim=(0, 1)))[0]
                    .detach()
                    .cpu()
                    .tolist()
                )
            return JointGRPOUpdateResult(
                loss=loss.detached(),
                gradient_norms=gradient_norms,
                clipped_gradient_norms=clipped_gradient_norms,
                total_gradient_norm=total_gradient_norm,
                adapter_relative_drifts=drifts,
                post_update_reference_kl=post_kl,
                optimizer_step=self.optimizer_step,
                zero_signal=False,
                stability_guard_rejected=True,
                stability_guard_trigger_modes=trigger_modes,
                objective_gradient_diagnostics=objective_gradient_diagnostics,
            )

        self.optimizer_step += 1
        return JointGRPOUpdateResult(
            loss=loss.detached(),
            gradient_norms=gradient_norms,
            clipped_gradient_norms=clipped_gradient_norms,
            total_gradient_norm=total_gradient_norm,
            adapter_relative_drifts=drifts,
            post_update_reference_kl=post_kl,
            optimizer_step=self.optimizer_step,
            zero_signal=False,
            stability_guard_rejected=False,
            stability_guard_trigger_modes=(),
            objective_gradient_diagnostics=objective_gradient_diagnostics,
        )


class JointGRPOTrainerA(_JointGRPOTrainerBase):
    """Variant-A joint GRPO with vectorized independent role decoding."""

    variant = "A"
    predecessor_condition = "none"


class JointGRPOTrainerB(_JointGRPOTrainerBase):
    """Variant-B joint GRPO with fixed predicted-detached action history."""

    variant = "B"
    predecessor_condition = "predicted_detached"

    def _decode_roles(
        self,
        *,
        decoder: nn.Module,
        mode_head: nn.Module,
        sample: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
        fixed_history: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = context.batch_size
        if fixed_history is not None:
            expected = (
                batch_size,
                NUM_PLATOON_ROLES - 1,
                TRAJECTORY_STEPS,
                TRAJECTORY_DIM,
            )
            if (
                fixed_history.dtype != torch.float32
                or tuple(fixed_history.shape) != expected
                or fixed_history.device != context.role_tokens.device
                or not bool(torch.isfinite(fixed_history).all())
                or fixed_history.requires_grad
            ):
                raise JointGRPOError(
                    "fixed predecessor history must be detached float32 "
                    "with shape [N,2,8,3]"
                )
        noisy_metric = self.planner._denormalize_xy(
            sample.clamp(-1.0, 1.0)
        )
        role_candidates = []
        role_logits = []
        generated_history = []
        predecessor_action: Tensor | None = None
        for role_index in range(NUM_PLATOON_ROLES):
            if role_index > 0 and fixed_history is not None:
                predecessor_action = fixed_history[:, role_index - 1]
            candidates, mode_features = decoder.forward_role(
                noisy_metric,
                coarse,
                timesteps,
                context,
                role_index=role_index,
                predecessor_action=predecessor_action,
            )
            logits = mode_head(mode_features).squeeze(-1)
            role_candidates.append(candidates)
            role_logits.append(logits)
            if role_index + 1 < NUM_PLATOON_ROLES and fixed_history is None:
                selected = self.planner._select_predecessor_trajectory(
                    candidates,
                    logits,
                    valid_mask[:, role_index],
                )
                predecessor_action = (
                    self.planner._normalize_predecessor_action(selected)
                ).float()
                generated_history.append(predecessor_action)
        history = (
            fixed_history
            if fixed_history is not None
            else torch.stack(generated_history, dim=1).detach()
        )
        return (
            torch.stack(role_candidates, dim=1),
            torch.stack(role_logits, dim=1),
            history,
        )

    def _rollout_prediction(
        self,
        sample: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self._decode_roles(
            decoder=self.planner.diffusion_decoder,
            mode_head=self.planner.mode_head,
            sample=sample,
            timesteps=timesteps,
            context=context,
            coarse=coarse,
            valid_mask=valid_mask,
            fixed_history=None,
        )

    def _frozen_reference_prediction(
        self,
        sample: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        candidates, logits, _ = self._decode_roles(
            decoder=self.reference.diffusion_decoder,
            mode_head=self.reference.mode_head,
            sample=sample,
            timesteps=timesteps,
            context=context,
            coarse=coarse,
            valid_mask=valid_mask,
            fixed_history=None,
        )
        return candidates, logits

    def _make_rollout(
        self,
        *,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
        chains: Tensor,
        candidates: Tensor,
        frozen_candidates: Tensor,
        noise_bundle_identity: tuple[int, int, int] | None,
        risk_pact_context: RiskPACTRolloutContext | None,
        step_histories: list[Tensor],
    ) -> JointGRPORolloutB:
        if len(step_histories) != len(self.config.roll_timesteps):
            raise JointGRPOError("variant B rollout action history count mismatch")
        batch_size, trajectories = chains.shape[:2]
        history = torch.stack(step_histories, dim=1).reshape(
            batch_size,
            trajectories,
            len(self.config.roll_timesteps),
            NUM_PLATOON_ROLES - 1,
            TRAJECTORY_STEPS,
            TRAJECTORY_DIM,
        )
        if history.dtype != torch.float32 or not bool(
            torch.isfinite(history).all()
        ):
            raise JointGRPOError(
                "variant B rollout action history must be finite float32"
            )
        return JointGRPORolloutB(
            context=BEVPlannerContext(
                bev_feature=context.bev_feature.detach(),
                role_tokens=context.role_tokens.detach(),
            ),
            coarse_trajectories=coarse.detach(),
            mode_valid_mask=valid_mask.detach(),
            chains_normalized=chains.detach(),
            candidate_trajectories=candidates.detach(),
            frozen_candidate_trajectories=frozen_candidates.detach(),
            risk_pact_context=risk_pact_context,
            noise_bundle_identity=noise_bundle_identity,
            predecessor_action_history_normalized=history.detach(),
        )

    def _replay_history(
        self,
        rollout: JointGRPORollout,
        *,
        step_index: int,
        flat_count: int,
    ) -> Tensor:
        if not isinstance(rollout, JointGRPORolloutB):
            raise JointGRPOError("variant B requires JointGRPORolloutB")
        expected = (
            rollout.batch_size,
            rollout.trajectories_per_mode,
            len(self.config.roll_timesteps),
            NUM_PLATOON_ROLES - 1,
            TRAJECTORY_STEPS,
            TRAJECTORY_DIM,
        )
        history = rollout.predecessor_action_history_normalized
        if (
            history.dtype != torch.float32
            or tuple(history.shape) != expected
            or not bool(torch.isfinite(history).all())
            or history.requires_grad
        ):
            raise JointGRPOError("variant B rollout action history is invalid")
        return history.reshape(
            flat_count,
            len(self.config.roll_timesteps),
            NUM_PLATOON_ROLES - 1,
            TRAJECTORY_STEPS,
            TRAJECTORY_DIM,
        )[:, step_index]

    def _replay_predictions(
        self,
        *,
        sample: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
        predecessor_history: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if predecessor_history is None:
            raise JointGRPOError("variant B replay requires action history")
        current_candidates, current_logits, _ = self._decode_roles(
            decoder=self.planner.diffusion_decoder,
            mode_head=self.planner.mode_head,
            sample=sample,
            timesteps=timesteps,
            context=context,
            coarse=coarse,
            valid_mask=valid_mask,
            fixed_history=predecessor_history,
        )
        with torch.no_grad():
            reference_candidates, reference_logits, _ = self._decode_roles(
                decoder=self.reference.diffusion_decoder,
                mode_head=self.reference.mode_head,
                sample=sample,
                timesteps=timesteps,
                context=context,
                coarse=coarse,
                valid_mask=valid_mask,
                fixed_history=predecessor_history,
            )
        return (
            current_candidates,
            current_logits,
            reference_candidates,
            reference_logits,
        )

__all__ = [
    "FrozenGRPOReference",
    "FrozenVariantAReference",
    "ModeResidualTrajectoryHead",
    "JointGRPOConfig",
    "JointGRPOError",
    "JointGRPOLossResult",
    "JointGRPORollout",
    "JointGRPORolloutB",
    "RiskPACTRolloutContext",
    "JointGRPOTrainerA",
    "JointGRPOTrainerB",
    "JointGRPOUpdateResult",
    "joint_grpo_optimizer_contract",
    "joint_grpo_optimizer_contract_sha256",
    "fixed_scale_safe_advantages",
]
