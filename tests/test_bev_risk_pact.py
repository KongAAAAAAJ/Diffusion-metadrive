from __future__ import annotations

import torch

from train.bev_risk_pact import (
    DynamicGaussianRiskField,
    RiskLevelSetConstraint,
    RiskPACTConfig,
    build_x0_pact_teacher,
)


def _fixture():
    cfg = RiskPACTConfig(risk_threshold=0.35, teacher_step_m=0.20)
    x = torch.linspace(0.0, 14.0, 8)
    old = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=-1)
    old = old.reshape(1, 1, 1, 8, 3)
    actor = torch.zeros((1, 1, 1, 8), dtype=torch.float32)
    actor[..., 0] = 8.0
    actor[..., 1] = 1.5
    actor[..., 2] = 1.0
    actor[..., 6] = 4.8
    actor[..., 7] = 2.0
    valid = torch.ones((1, 1, 1), dtype=torch.bool)
    return cfg, old, actor, valid


def test_teacher_reduces_risk():
    cfg, old, actor, valid = _fixture()
    field = DynamicGaussianRiskField(cfg)
    constraint = RiskLevelSetConstraint(cfg)
    result = build_x0_pact_teacher(
        old, actor, valid, curriculum_scale=1.0, config=cfg,
        risk_field=field, constraint=constraint,
    )
    after = constraint(field.query(result.teacher_trajectory[..., :2], actor, valid).risk)
    assert after.trajectory_risk.item() < result.constraint.trajectory_risk.item()


def test_safe_candidate_is_unchanged():
    cfg, old, actor, valid = _fixture()
    old = old.clone()
    old[..., 1] = -20.0
    result = build_x0_pact_teacher(old, actor, valid, curriculum_scale=1.0, config=cfg)
    assert bool(result.constraint.safe_mask.item())
    assert torch.allclose(result.teacher_trajectory, old)


def test_zero_curriculum_is_identity():
    cfg, old, actor, valid = _fixture()
    result = build_x0_pact_teacher(old, actor, valid, curriculum_scale=0.0, config=cfg)
    assert torch.allclose(result.teacher_trajectory, old)
