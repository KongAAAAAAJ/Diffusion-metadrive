from __future__ import annotations

from pathlib import Path

import torch

from train.bev_risk_pact.config import RiskPACTConfig, RiskPACTVisualizationConfig
from train.bev_risk_pact.road_field import build_drivable_signed_distance
from train.bev_risk_pact.teacher import build_x0_pact_teacher
from train.bev_risk_pact.visualize import save_risk_pact_debug_plots


def _straight_strip(half_width_m: float = 3.0) -> torch.Tensor:
    h = w = 256
    y = 32.0 - torch.arange(w, dtype=torch.float32) / float(w - 1) * 64.0
    bev = torch.zeros((1, 3, 8, h, w), dtype=torch.uint8)
    bev[:, :, 0, :, y.abs() <= half_width_m] = 255
    return bev


def _far_actor(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    actor = torch.zeros((1, 3, count, 8), dtype=torch.float32)
    actor[..., 0] = 100.0
    actor[..., 3] = 1.0
    actor[..., 6] = 5.74
    actor[..., 7] = 2.30
    valid = torch.ones((1, 3, count), dtype=torch.bool)
    return actor, valid


def _teacher_fixture():
    cfg = RiskPACTConfig(
        use_background_actor=True,
        use_platoon_actor=True,
        use_road_boundary=True,
        ego_width_m=2.30,
        road_safety_margin_m=0.4,
        road_temperature_m=0.2,
        teacher_step_m=0.2,
    )
    sdf = build_drivable_signed_distance(_straight_strip())
    old = torch.zeros((1, 3, 1, 8, 3), dtype=torch.float32)
    old[..., 0] = torch.linspace(2.0, 16.0, 8)
    # Become progressively more critical near the right road edge.
    old[..., 1] = torch.linspace(1.8, 2.8, 8)
    bg, bg_valid = _far_actor(1)
    platoon, platoon_valid = _far_actor(2)
    teacher = build_x0_pact_teacher(
        old,
        bg,
        bg_valid,
        platoon_actor_state=platoon,
        platoon_actor_valid_mask=platoon_valid,
        road_sdf=sdf,
        curriculum_scale=1.0,
        config=cfg,
    )
    return cfg, sdf, old, bg, bg_valid, platoon, platoon_valid, teacher


def test_teacher_reports_critical_timestep_and_post_projection_risk() -> None:
    _, _, _, _, _, _, _, teacher = _teacher_fixture()
    assert teacher.constraint.critical_timestep_index is not None
    assert teacher.teacher_constraint is not None
    assert teacher.teacher_field is not None
    expected = teacher.field.risk.argmax(dim=-1)
    assert torch.equal(teacher.constraint.critical_timestep_index, expected)
    active = teacher.constraint.near_or_unsafe_mask
    assert bool(active.any())
    before = teacher.constraint.trajectory_risk[active]
    after = teacher.teacher_constraint.trajectory_risk[active]
    # The normalized gradient step is deliberately small, but it should not
    # increase the level-set risk in this smooth road-only diagnostic scene.
    assert bool(torch.all(after <= before + 1.0e-6))
    assert float((before - after).mean()) > 0.0


def test_teacher_field_exposes_signed_clearance_diagnostics() -> None:
    _, _, _, _, _, _, _, teacher = _teacher_fixture()
    assert teacher.field.background_signed_clearance_m is not None
    assert teacher.field.platoon_signed_clearance_m is not None
    assert teacher.field.road_signed_distance_m is not None
    assert teacher.field.background_signed_clearance_m.shape[-2:] == (8, 1)
    assert teacher.field.platoon_signed_clearance_m.shape[-2:] == (8, 2)


def test_step6_visualization_bundle_writes_scene_risk_clearance_and_summary(tmp_path: Path) -> None:
    cfg, sdf, old_flat, bg, bg_valid, platoon, platoon_valid, teacher = _teacher_fixture()
    # Visualization expects the original [B,R,mode,group,H,D] rollout layout.
    old = old_flat.reshape(1, 3, 1, 1, 8, 3)
    valid_mode = torch.ones((1, 3, 1), dtype=torch.bool)
    vis_cfg = RiskPACTVisualizationConfig(
        enabled=True,
        interval_steps=1,
        start_step=1,
        max_events=1,
        selection="highest_risk",
    )
    summary = save_risk_pact_debug_plots(
        background_actor_state=bg,
        background_actor_valid_mask=bg_valid,
        platoon_actor_state=platoon,
        platoon_actor_valid_mask=platoon_valid,
        road_sdf=sdf,
        old_trajectory=old,
        valid_executable_mode_mask=valid_mode,
        teacher_result=teacher,
        output_dir=tmp_path,
        step=1,
        visualization_config=vis_cfg,
        config=cfg,
    )
    assert summary["trajectory_risk_after"] <= summary["trajectory_risk_before"] + 1.0e-6
    assert len(list(tmp_path.glob("*_scene.png"))) == 1
    assert len(list(tmp_path.glob("*_risk.png"))) == 1
    assert len(list(tmp_path.glob("*_clearance.png"))) == 1
    assert len(list(tmp_path.glob("*_summary.json"))) == 1


def test_teacher_post_projection_evaluation_can_be_disabled() -> None:
    cfg = RiskPACTConfig(use_background_actor=True, use_platoon_actor=False, use_road_boundary=False)
    old = torch.zeros((1, 3, 1, 8, 3), dtype=torch.float32)
    old[..., 0] = torch.linspace(2.0, 16.0, 8)
    bg = torch.zeros((1, 3, 1, 8), dtype=torch.float32)
    bg[..., 0] = 8.0
    bg[..., 3] = 1.0
    bg[..., 6] = 5.74
    bg[..., 7] = 2.30
    valid = torch.ones((1, 3, 1), dtype=torch.bool)
    teacher = build_x0_pact_teacher(
        old,
        bg,
        valid,
        curriculum_scale=0.2,
        config=cfg,
        evaluate_teacher=False,
    )
    assert teacher.teacher_field is None
    assert teacher.teacher_constraint is None
