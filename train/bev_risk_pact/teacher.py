from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import RiskPACTConfig
from .constraint import RiskConstraintResult, RiskLevelSetConstraint
from .risk_field import MultiSourceSafetyRiskField, RiskFieldResult


@dataclass(frozen=True)
class PACTTeacherResult:
    teacher_trajectory: Tensor
    raw_gradient: Tensor
    normalized_gradient: Tensor
    displacement_norm: Tensor
    field: RiskFieldResult
    constraint: RiskConstraintResult


def _detach_optional(value: Tensor | None) -> Tensor | None:
    return None if value is None else value.detach()


def build_x0_pact_teacher(
    old_trajectory: Tensor,
    actor_state: Tensor | None = None,
    actor_valid_mask: Tensor | None = None,
    *,
    platoon_actor_state: Tensor | None = None,
    platoon_actor_valid_mask: Tensor | None = None,
    road_sdf: Tensor | None = None,
    curriculum_scale: float,
    config: RiskPACTConfig | None = None,
    risk_field: MultiSourceSafetyRiskField | None = None,
    constraint: RiskLevelSetConstraint | None = None,
) -> PACTTeacherResult:
    """Construct a final-x0 PACT-lite teacher from a multi-source safety field.

    ``actor_state`` / ``actor_valid_mask`` retain the Step-1~5 positional API
    and denote background traffic.  Step 5.5 adds optional platoon-neighbor and
    drivable-road geometry.  Only x/y are corrected; heading is unchanged.
    """

    cfg = config or RiskPACTConfig()
    if not 0.0 <= float(curriculum_scale) <= 1.0:
        raise ValueError("curriculum_scale must be in [0,1]")
    if old_trajectory.ndim != 5 or old_trajectory.shape[-1] not in (2, 3):
        raise ValueError("old_trajectory must have shape [B,R,M,H,2/3]")

    xy = old_trajectory[..., :2].detach().clone().requires_grad_(True)
    field_impl = risk_field or MultiSourceSafetyRiskField(cfg)
    constraint_impl = constraint or RiskLevelSetConstraint(cfg)
    if isinstance(field_impl, MultiSourceSafetyRiskField):
        field_result = field_impl.query(
            xy,
            background_actor_state=actor_state,
            background_actor_valid_mask=actor_valid_mask,
            platoon_actor_state=platoon_actor_state,
            platoon_actor_valid_mask=platoon_actor_valid_mask,
            road_sdf=road_sdf,
        )
    else:
        # Backward-compatible injection path used by the original pilot tests.
        if actor_state is None or actor_valid_mask is None:
            raise ValueError("legacy actor risk field requires background actor state/mask")
        field_result = field_impl.query(xy, actor_state, actor_valid_mask)
    constraint_result = constraint_impl(field_result.risk)

    total_violation = constraint_result.violation.sum()
    gradient = torch.autograd.grad(
        total_violation,
        xy,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0]

    # Per-trajectory normalization, preserving the temporal/coordinate direction.
    flat = gradient.flatten(start_dim=-2)
    norm = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
    clip = float(cfg.gradient_clip_norm)
    scale = torch.clamp(clip / norm.clamp_min(float(cfg.gradient_eps)), max=1.0)
    clipped = gradient * scale.unsqueeze(-1)
    clipped_flat = clipped.flatten(start_dim=-2)
    clipped_norm = torch.linalg.vector_norm(clipped_flat, dim=-1, keepdim=True)
    normalized = clipped / clipped_norm.clamp_min(float(cfg.gradient_eps)).unsqueeze(-1)

    active = constraint_result.near_or_unsafe_mask.unsqueeze(-1).unsqueeze(-1)
    step_m = float(cfg.teacher_step_m) * float(curriculum_scale)
    correction = torch.where(active, -step_m * normalized, torch.zeros_like(normalized))
    teacher_xy = xy.detach() + correction.detach()

    if old_trajectory.shape[-1] == 3:
        teacher = torch.cat((teacher_xy, old_trajectory[..., 2:3].detach()), dim=-1)
    else:
        teacher = teacher_xy

    displacement = torch.linalg.vector_norm(correction.flatten(start_dim=-2), dim=-1)
    return PACTTeacherResult(
        teacher_trajectory=teacher,
        raw_gradient=gradient.detach(),
        normalized_gradient=normalized.detach(),
        displacement_norm=displacement.detach(),
        field=RiskFieldResult(
            risk=field_result.risk.detach(),
            per_actor_risk=field_result.per_actor_risk.detach(),
            actor_future_xy=field_result.actor_future_xy.detach(),
            background_risk=_detach_optional(field_result.background_risk),
            platoon_risk=_detach_optional(field_result.platoon_risk),
            road_risk=_detach_optional(field_result.road_risk),
            road_signed_distance_m=_detach_optional(field_result.road_signed_distance_m),
        ),
        constraint=RiskConstraintResult(
            trajectory_risk=constraint_result.trajectory_risk.detach(),
            violation=constraint_result.violation.detach(),
            safe_mask=constraint_result.safe_mask.detach(),
            near_or_unsafe_mask=constraint_result.near_or_unsafe_mask.detach(),
            temporal_weights=constraint_result.temporal_weights.detach(),
        ),
    )
