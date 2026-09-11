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
    teacher_field: RiskFieldResult | None = None
    teacher_constraint: RiskConstraintResult | None = None


def _detach_optional(value: Tensor | None) -> Tensor | None:
    return None if value is None else value.detach()


def _detach_field(value: RiskFieldResult) -> RiskFieldResult:
    return RiskFieldResult(
        risk=value.risk.detach(),
        per_actor_risk=value.per_actor_risk.detach(),
        actor_future_xy=value.actor_future_xy.detach(),
        background_risk=_detach_optional(value.background_risk),
        platoon_risk=_detach_optional(value.platoon_risk),
        road_risk=_detach_optional(value.road_risk),
        background_risk_raw=_detach_optional(value.background_risk_raw),
        platoon_risk_raw=_detach_optional(value.platoon_risk_raw),
        actor_confidence=_detach_optional(value.actor_confidence),
        road_signed_distance_m=_detach_optional(value.road_signed_distance_m),
        per_actor_signed_clearance_m=_detach_optional(value.per_actor_signed_clearance_m),
        background_signed_clearance_m=_detach_optional(value.background_signed_clearance_m),
        platoon_signed_clearance_m=_detach_optional(value.platoon_signed_clearance_m),
    )


def _detach_constraint(value: RiskConstraintResult) -> RiskConstraintResult:
    return RiskConstraintResult(
        trajectory_risk=value.trajectory_risk.detach(),
        violation=value.violation.detach(),
        safe_mask=value.safe_mask.detach(),
        near_or_unsafe_mask=value.near_or_unsafe_mask.detach(),
        temporal_weights=value.temporal_weights.detach(),
        critical_timestep_index=(
            None if value.critical_timestep_index is None
            else value.critical_timestep_index.detach()
        ),
    )


def _query_field(
    field_impl: MultiSourceSafetyRiskField,
    xy: Tensor,
    *,
    actor_state: Tensor | None,
    actor_valid_mask: Tensor | None,
    platoon_actor_state: Tensor | None,
    platoon_actor_valid_mask: Tensor | None,
    road_sdf: Tensor | None,
) -> RiskFieldResult:
    return field_impl.query(
        xy,
        background_actor_state=actor_state,
        background_actor_valid_mask=actor_valid_mask,
        platoon_actor_state=platoon_actor_state,
        platoon_actor_valid_mask=platoon_actor_valid_mask,
        road_sdf=road_sdf,
    )


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
    evaluate_teacher: bool = True,
) -> PACTTeacherResult:
    """Construct a final-x0 PACT-lite teacher and Step-6 safety diagnostics.

    The optimization path is unchanged from Step 5.6: the old on-policy x0 is
    projected one normalized risk-gradient step and then detached. Step 6 only
    adds an evaluation of the projected teacher under the *same* safety field so
    that risk-before/risk-after and critical-timestep diagnostics are available.
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
        field_result = _query_field(
            field_impl,
            xy,
            actor_state=actor_state,
            actor_valid_mask=actor_valid_mask,
            platoon_actor_state=platoon_actor_state,
            platoon_actor_valid_mask=platoon_actor_valid_mask,
            road_sdf=road_sdf,
        )
    else:
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

    # Step-6 diagnostic only: optionally evaluate the projected target under the
    # identical detached scene geometry. This can be disabled for formal runs to
    # avoid the extra safety-field query; it never changes the student objective.
    projected_field = None
    projected_constraint = None
    if evaluate_teacher:
        with torch.no_grad():
            if isinstance(field_impl, MultiSourceSafetyRiskField):
                projected_field = _query_field(
                    field_impl,
                    teacher_xy,
                    actor_state=actor_state,
                    actor_valid_mask=actor_valid_mask,
                    platoon_actor_state=platoon_actor_state,
                    platoon_actor_valid_mask=platoon_actor_valid_mask,
                    road_sdf=road_sdf,
                )
            else:
                projected_field = field_impl.query(teacher_xy, actor_state, actor_valid_mask)
            projected_constraint = constraint_impl(projected_field.risk)

    displacement = torch.linalg.vector_norm(correction.flatten(start_dim=-2), dim=-1)
    return PACTTeacherResult(
        teacher_trajectory=teacher,
        raw_gradient=gradient.detach(),
        normalized_gradient=normalized.detach(),
        displacement_norm=displacement.detach(),
        field=_detach_field(field_result),
        constraint=_detach_constraint(constraint_result),
        teacher_field=(None if projected_field is None else _detach_field(projected_field)),
        teacher_constraint=(
            None if projected_constraint is None else _detach_constraint(projected_constraint)
        ),
    )
