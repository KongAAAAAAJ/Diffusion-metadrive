from __future__ import annotations

import torch

from train.bev_risk_pact import (
    DynamicGaussianRiskField,
    RiskLevelSetConstraint,
    RiskPACTConfig,
    RiskPACTVisualizationConfig,
    RiskPACTTrainingVisualizer,
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
    actor[..., 2] = 0.0  # sin(dheading)
    actor[..., 3] = 1.0  # cos(dheading), aligned with ego x axis
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


def test_training_visualizer_step_gate(tmp_path):
    vis = RiskPACTTrainingVisualizer(
        run_dir=tmp_path,
        visualization_config=RiskPACTVisualizationConfig(
            enabled=True, interval_steps=5, start_step=2, max_events=2
        ),
    )
    assert not vis.should_save(0)
    assert vis.should_save(2)
    # should_save itself is side-effect free until an event is emitted.
    assert vis.should_save(7)
    assert not vis.should_save(8)


def test_training_visualizer_disabled(tmp_path):
    vis = RiskPACTTrainingVisualizer(
        run_dir=tmp_path,
        visualization_config=RiskPACTVisualizationConfig(enabled=False),
    )
    assert not vis.should_save(0)
    assert not vis.should_save(100)
