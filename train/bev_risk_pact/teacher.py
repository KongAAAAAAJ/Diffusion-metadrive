from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import RiskPACTConfig
from .constraint import RiskConstraintResult, RiskLevelSetConstraint
from .risk_field import DynamicGaussianRiskField, RiskFieldResult


@dataclass(frozen=True)
class PACTTeacherResult:
    teacher_trajectory: Tensor
    raw_gradient: Tensor
    normalized_gradient: Tensor
    displacement_norm: Tensor
    field: RiskFieldResult
    constraint: RiskConstraintResult


def build_x0_pact_teacher(
    old_trajectory: Tensor,
    actor_state: Tensor,
    actor_valid_mask: Tensor,
    *,
    curriculum_scale: float,
    config: RiskPACTConfig | None = None,
    risk_field: DynamicGaussianRiskField | None = None,
    constraint: RiskLevelSetConstraint | None = None,
) -> PACTTeacherResult:
    """Construct an x0-space PACT-lite teacher.

    ``old_trajectory`` must be the frozen/old policy clean prediction with shape
    [B,R,M,H,3] (or [...,2/3] with the documented leading axes).  Only x/y are
    corrected; heading is kept unchanged in this first pilot.

    This is deliberately a *pilot approximation*: it performs a normalized
    Euclidean gradient step in clean trajectory space.  It does not claim exact
    equivalence to PACT's reverse-KL score-space projection.
    """
    cfg = config or RiskPACTConfig()
    if not 0.0 <= float(curriculum_scale) <= 1.0:
        raise ValueError("curriculum_scale must be in [0,1]")
    if old_trajectory.ndim != 5 or old_trajectory.shape[-1] not in (2, 3):
        raise ValueError("old_trajectory must have shape [B,R,M,H,2/3]")

    xy = old_trajectory[..., :2].detach().clone().requires_grad_(True)
    field_impl = risk_field or DynamicGaussianRiskField(cfg)
    constraint_impl = constraint or RiskLevelSetConstraint(cfg)
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
    teacher_xy = (xy.detach() + correction.detach())

    if old_trajectory.shape[-1] == 3:
        teacher = torch.cat((teacher_xy, old_trajectory[..., 2:3].detach()), dim=-1)
    else:
        teacher = teacher_xy

    displacement = torch.linalg.vector_norm(
        correction.flatten(start_dim=-2), dim=-1
    )
    return PACTTeacherResult(
        teacher_trajectory=teacher,
        raw_gradient=gradient.detach(),
        normalized_gradient=normalized.detach(),
        displacement_norm=displacement.detach(),
        field=RiskFieldResult(
            risk=field_result.risk.detach(),
            per_actor_risk=field_result.per_actor_risk.detach(),
            actor_future_xy=field_result.actor_future_xy.detach(),
        ),
        constraint=RiskConstraintResult(
            trajectory_risk=constraint_result.trajectory_risk.detach(),
            violation=constraint_result.violation.detach(),
            safe_mask=constraint_result.safe_mask.detach(),
            near_or_unsafe_mask=constraint_result.near_or_unsafe_mask.detach(),
            temporal_weights=constraint_result.temporal_weights.detach(),
        ),
    )
