"""PACT-lite distillation loss for ChassisFusion Risk-PACT.

Step 4 intentionally implements only the supervised distillation objective.
It does not call rollout, build the teacher, compose the GRPO total loss, call
``backward()``, or step an optimizer.  Those integration points belong to
Step 5.

The current pilot teacher corrects x/y only.  Heading is therefore excluded
from this loss by design.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .teacher import PACTTeacherResult


@dataclass(frozen=True)
class PACTLiteDistillationLossResult:
    """Outputs of the masked x0-space PACT-lite distillation objective."""

    loss: Tensor
    per_trajectory_mse: Tensor
    active_mask: Tensor
    active_ratio: Tensor
    active_count: Tensor
    teacher_displacement_mean_m: Tensor
    teacher_displacement_max_m: Tensor


def _validate_student_and_teacher(
    student_trajectory: Tensor,
    teacher_result: PACTTeacherResult,
) -> tuple[Tensor, Tensor, Tensor]:
    teacher_trajectory = teacher_result.teacher_trajectory
    active_mask = teacher_result.constraint.near_or_unsafe_mask

    if student_trajectory.ndim != 5:
        raise ValueError(
            "student_trajectory must have shape [B,R,M,H,2/3]"
        )
    if student_trajectory.shape[-1] not in (2, 3):
        raise ValueError(
            "student_trajectory must have final coordinate dimension 2 or 3"
        )
    if tuple(student_trajectory.shape) != tuple(teacher_trajectory.shape):
        raise ValueError(
            "student_trajectory and teacher_trajectory must have identical shapes; "
            f"got {tuple(student_trajectory.shape)} vs {tuple(teacher_trajectory.shape)}"
        )
    if student_trajectory.shape[-2] <= 0:
        raise ValueError("student_trajectory horizon must be non-empty")
    if not student_trajectory.is_floating_point():
        raise TypeError("student_trajectory must be floating point")
    if not teacher_trajectory.is_floating_point():
        raise TypeError("teacher_trajectory must be floating point")
    if student_trajectory.device != teacher_trajectory.device:
        raise ValueError(
            "student_trajectory and teacher_trajectory must be on the same device"
        )
    if not bool(torch.isfinite(student_trajectory).all()):
        raise ValueError("student_trajectory must be finite")
    if not bool(torch.isfinite(teacher_trajectory).all()):
        raise ValueError("teacher_trajectory must be finite")

    expected_mask_shape = student_trajectory.shape[:-2]
    if tuple(active_mask.shape) != tuple(expected_mask_shape):
        raise ValueError(
            "near_or_unsafe_mask must match [B,R,M]; "
            f"expected {tuple(expected_mask_shape)}, got {tuple(active_mask.shape)}"
        )
    if active_mask.device != student_trajectory.device:
        raise ValueError(
            "near_or_unsafe_mask and student_trajectory must be on the same device"
        )
    if active_mask.dtype is not torch.bool:
        raise TypeError("near_or_unsafe_mask must have dtype torch.bool")

    student_xy = student_trajectory[..., :2]
    # The teacher must behave as a frozen target even if a future caller passes
    # a tensor that accidentally retains a graph.
    teacher_xy = teacher_trajectory[..., :2].detach()
    return student_xy, teacher_xy, active_mask.detach()


def pact_lite_distillation_loss(
    student_trajectory: Tensor,
    teacher_result: PACTTeacherResult,
) -> PACTLiteDistillationLossResult:
    """Compute the masked x0-space PACT-lite teacher distillation loss.

    For each candidate trajectory ``g`` the pilot objective is

    ``MSE_g = mean_{h,xy} (tau_student - tau_teacher)^2``.

    Only candidates marked by ``near_or_unsafe_mask`` contribute to the
    reduction.  Clearly safe candidates contribute exactly zero safety
    supervision.  If a batch contains no active candidates, ``loss`` is a
    differentiable zero connected to ``student_trajectory`` so the caller can
    safely include it in a larger loss expression and call ``backward()``.

    Heading is deliberately not supervised in Step 4 because the current
    x0-space teacher only modifies x/y.
    """

    student_xy, teacher_xy, active_mask = _validate_student_and_teacher(
        student_trajectory,
        teacher_result,
    )

    squared_error = (student_xy - teacher_xy).square()
    per_trajectory_mse = squared_error.mean(dim=(-2, -1))

    active_weights = active_mask.to(dtype=per_trajectory_mse.dtype)
    active_count = active_weights.sum()
    total_count = active_weights.new_tensor(float(active_weights.numel()))
    active_ratio = active_count / total_count.clamp_min(1.0)

    if bool(active_mask.any()):
        loss = (
            per_trajectory_mse * active_weights
        ).sum() / active_count.clamp_min(1.0)
    else:
        # Preserve a valid autograd edge while guaranteeing zero gradient.
        loss = student_xy.sum() * 0.0

    displacement = teacher_result.displacement_norm
    if tuple(displacement.shape) != tuple(active_mask.shape):
        raise ValueError(
            "teacher displacement_norm must match [B,R,M]; "
            f"expected {tuple(active_mask.shape)}, got {tuple(displacement.shape)}"
        )
    if displacement.device != student_trajectory.device:
        raise ValueError(
            "teacher displacement_norm and student_trajectory must share a device"
        )
    if not bool(torch.isfinite(displacement).all()):
        raise ValueError("teacher displacement_norm must be finite")

    if bool(active_mask.any()):
        active_displacement = displacement[active_mask]
        displacement_mean = active_displacement.mean()
        displacement_max = active_displacement.max()
    else:
        displacement_mean = displacement.new_zeros(())
        displacement_max = displacement.new_zeros(())

    return PACTLiteDistillationLossResult(
        loss=loss,
        per_trajectory_mse=per_trajectory_mse,
        active_mask=active_mask,
        active_ratio=active_ratio,
        active_count=active_count,
        teacher_displacement_mean_m=displacement_mean,
        teacher_displacement_max_m=displacement_max,
    )


__all__ = [
    "PACTLiteDistillationLossResult",
    "pact_lite_distillation_loss",
]
