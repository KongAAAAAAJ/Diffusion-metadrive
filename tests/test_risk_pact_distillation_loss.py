from __future__ import annotations

import torch
import pytest

from train.bev_risk_pact.constraint import RiskConstraintResult
from train.bev_risk_pact.loss import pact_lite_distillation_loss
from train.bev_risk_pact.risk_field import RiskFieldResult
from train.bev_risk_pact.teacher import PACTTeacherResult


def _teacher_result(
    teacher: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    displacement: torch.Tensor | None = None,
) -> PACTTeacherResult:
    # Minimal but shape-consistent diagnostic tensors.  The loss only consumes
    # teacher_trajectory, near_or_unsafe_mask, and displacement_norm; keeping a
    # complete PACTTeacherResult makes the test match the production contract.
    b, r, m, h, _ = teacher.shape
    device = teacher.device
    dtype = teacher.dtype
    if displacement is None:
        displacement = torch.zeros((b, r, m), device=device, dtype=dtype)

    risk = torch.zeros((b, r, m, h), device=device, dtype=dtype)
    zeros_brm = torch.zeros((b, r, m), device=device, dtype=dtype)
    safe_mask = ~active_mask
    temporal_weights = torch.full_like(risk, 1.0 / float(h))

    return PACTTeacherResult(
        teacher_trajectory=teacher,
        raw_gradient=torch.zeros_like(teacher[..., :2]),
        normalized_gradient=torch.zeros_like(teacher[..., :2]),
        displacement_norm=displacement,
        field=RiskFieldResult(
            risk=risk,
            per_actor_risk=torch.zeros(
                (b, r, m, h, 1), device=device, dtype=dtype
            ),
            actor_future_xy=torch.zeros(
                (b, r, 1, h, 2), device=device, dtype=dtype
            ),
        ),
        constraint=RiskConstraintResult(
            trajectory_risk=zeros_brm,
            violation=zeros_brm,
            safe_mask=safe_mask,
            near_or_unsafe_mask=active_mask,
            temporal_weights=temporal_weights,
        ),
    )


def test_matching_student_and_teacher_has_zero_loss() -> None:
    teacher = torch.randn(1, 3, 2, 8, 3)
    active = torch.ones(1, 3, 2, dtype=torch.bool)
    student = teacher.clone().requires_grad_(True)

    result = pact_lite_distillation_loss(
        student,
        _teacher_result(teacher, active),
    )
    assert result.loss.item() == pytest.approx(0.0)
    assert result.active_ratio.item() == pytest.approx(1.0)


def test_heading_is_not_supervised() -> None:
    teacher = torch.zeros(1, 1, 1, 4, 3)
    active = torch.ones(1, 1, 1, dtype=torch.bool)
    student = teacher.clone()
    student[..., 2] = 100.0
    student.requires_grad_(True)

    result = pact_lite_distillation_loss(
        student,
        _teacher_result(teacher, active),
    )
    assert result.loss.item() == pytest.approx(0.0)


def test_safe_candidates_are_excluded_from_distillation() -> None:
    teacher = torch.zeros(1, 1, 2, 2, 3)
    active = torch.tensor([[[True, False]]])
    student = teacher.clone()

    # Active candidate has xy error 1 -> MSE 1.  Safe candidate has a much
    # larger error but must contribute exactly zero to the safety objective.
    student[:, :, 0, :, :2] = 1.0
    student[:, :, 1, :, :2] = 100.0
    student.requires_grad_(True)

    result = pact_lite_distillation_loss(
        student,
        _teacher_result(teacher, active),
    )
    assert result.loss.item() == pytest.approx(1.0)
    assert result.active_ratio.item() == pytest.approx(0.5)
    assert result.active_count.item() == pytest.approx(1.0)


def test_no_active_candidates_returns_differentiable_zero() -> None:
    teacher = torch.zeros(1, 1, 2, 2, 3)
    active = torch.zeros(1, 1, 2, dtype=torch.bool)
    student = torch.randn_like(teacher, requires_grad=True)

    result = pact_lite_distillation_loss(
        student,
        _teacher_result(teacher, active),
    )
    assert result.loss.item() == pytest.approx(0.0)
    result.loss.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad).item() == 0


def test_gradient_moves_student_toward_teacher_xy() -> None:
    teacher = torch.ones(1, 1, 1, 1, 3)
    teacher[..., 2] = 7.0
    active = torch.ones(1, 1, 1, dtype=torch.bool)
    student = torch.zeros_like(teacher, requires_grad=True)

    result = pact_lite_distillation_loss(
        student,
        _teacher_result(teacher, active),
    )
    result.loss.backward()

    # d/d student (student - 1)^2 is negative at student=0, so gradient descent
    # increases x/y toward the teacher.  Heading has no safety gradient.
    assert bool((student.grad[..., :2] < 0.0).all())
    assert torch.count_nonzero(student.grad[..., 2]).item() == 0


def test_teacher_is_always_detached_from_student_loss() -> None:
    teacher = torch.randn(1, 1, 1, 2, 3, requires_grad=True)
    active = torch.ones(1, 1, 1, dtype=torch.bool)
    student = torch.zeros_like(teacher, requires_grad=True)

    result = pact_lite_distillation_loss(
        student,
        _teacher_result(teacher, active),
    )
    result.loss.backward()

    assert student.grad is not None
    assert teacher.grad is None
