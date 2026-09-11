from __future__ import annotations

import torch

from train.bev_risk_pact.config import RiskPACTConfig
from train.bev_risk_pact.platoon_actor import build_platoon_actor_state
from train.bev_risk_pact.risk_field import MultiSourceSafetyRiskField
from train.bev_risk_pact.road_field import RoadBoundaryRiskField, build_drivable_signed_distance
from train.bev_risk_pact.teacher import build_x0_pact_teacher


def _bev_with_straight_drivable_strip(half_width_m: float = 3.0) -> torch.Tensor:
    # Build through metric coordinates to stay faithful to SemanticBEV mapping.
    h = w = 256
    y_max, y_min = 32.0, -32.0
    cols = torch.arange(w, dtype=torch.float32)
    y = y_max - cols / float(w - 1) * (y_max - y_min)
    mask_col = y.abs() <= half_width_m
    bev = torch.zeros((1, 3, 8, h, w), dtype=torch.uint8)
    bev[:, :, 0, :, mask_col] = 255
    return bev


def test_platoon_relation_conversion_uses_kmh_delta_speed() -> None:
    cfg = RiskPACTConfig()
    ego = torch.zeros((1, 3, 8), dtype=torch.float32)
    ego[..., 0] = 10.0  # m/s
    relation = torch.zeros((1, 3, 12), dtype=torch.float32)
    relation[..., 0] = 8.0
    relation[..., 3] = 3.6  # +1 m/s
    relation[..., 6] = -8.0
    relation[..., 9] = -3.6  # -1 m/s
    valid = torch.ones((1, 3, 2), dtype=torch.bool)

    state, mask = build_platoon_actor_state(ego, relation, valid, config=cfg)
    assert state.shape == (1, 3, 2, 8)
    assert torch.allclose(state[..., 0, 4], torch.ones((1, 3)), atol=1e-5)
    assert torch.allclose(state[..., 1, 4], -torch.ones((1, 3)), atol=1e-5)
    assert torch.equal(mask, valid)


def test_road_risk_gradient_pushes_trajectory_inward() -> None:
    cfg = RiskPACTConfig(
        use_background_actor=False,
        use_platoon_actor=False,
        use_road_boundary=True,
        ego_width_m=2.0,
        road_safety_margin_m=0.2,
        road_temperature_m=0.25,
    )
    bev = _bev_with_straight_drivable_strip(half_width_m=3.0)
    sdf = build_drivable_signed_distance(bev)
    # Near the +y edge.  Increasing y approaches/exits the boundary, so the
    # risk gradient should be +y and the PACT correction -grad should move inward.
    traj = torch.zeros((1, 3, 1, 8, 2), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        traj[..., 0] = torch.linspace(2.0, 16.0, 8)
        traj[..., 1] = 2.4
    road = RoadBoundaryRiskField(cfg).query(traj, sdf)
    grad = torch.autograd.grad(road.risk.sum(), traj)[0]
    assert float(road.risk.mean()) > 0.05
    assert float(grad[..., 1].mean()) > 0.0


def test_platoon_actor_risk_gradient_points_away_from_neighbor() -> None:
    cfg = RiskPACTConfig(
        use_background_actor=False,
        use_platoon_actor=True,
        use_road_boundary=False,
        inflate_actor_by_ego_footprint=False,
        longitudinal_margin_m=1.0,
        lateral_margin_m=0.5,
    )
    trajectory = torch.zeros((1, 3, 1, 8, 2), dtype=torch.float32, requires_grad=True)
    actor = torch.zeros((1, 3, 2, 8), dtype=torch.float32)
    actor[..., 0, 0] = 4.0
    actor[..., 0, 3] = 1.0
    actor[..., 0, 6] = 4.0
    actor[..., 0, 7] = 2.0
    valid = torch.zeros((1, 3, 2), dtype=torch.bool)
    valid[..., 0] = True

    result = MultiSourceSafetyRiskField(cfg).query(
        trajectory,
        platoon_actor_state=actor,
        platoon_actor_valid_mask=valid,
    )
    grad = torch.autograd.grad(result.risk.sum(), trajectory)[0]
    # Moving ego +x toward a neighbor at +x increases risk; -grad moves away.
    assert float(grad[..., 0].mean()) > 0.0


def test_multisource_union_and_teacher_activate_from_road_only() -> None:
    cfg = RiskPACTConfig(
        use_background_actor=True,
        use_platoon_actor=True,
        use_road_boundary=True,
        ego_width_m=2.0,
        road_safety_margin_m=0.3,
        road_temperature_m=0.2,
        risk_threshold=0.35,
        safe_margin=0.05,
        teacher_step_m=0.2,
    )
    bev = _bev_with_straight_drivable_strip(half_width_m=3.0)
    sdf = build_drivable_signed_distance(bev)
    old = torch.zeros((1, 3, 1, 8, 3), dtype=torch.float32)
    old[..., 0] = torch.linspace(2.0, 16.0, 8)
    old[..., 1] = 2.7

    # Put all actors far away so road is the only meaningful source.
    bg = torch.zeros((1, 3, 1, 8), dtype=torch.float32)
    bg[..., 0] = 100.0
    bg[..., 3] = 1.0
    bg[..., 6] = 4.0
    bg[..., 7] = 2.0
    bg_valid = torch.ones((1, 3, 1), dtype=torch.bool)
    platoon = bg.repeat(1, 1, 2, 1)
    platoon_valid = torch.ones((1, 3, 2), dtype=torch.bool)

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
    assert teacher.field.road_risk is not None
    assert teacher.field.background_risk is not None
    assert teacher.field.platoon_risk is not None
    assert bool(teacher.constraint.near_or_unsafe_mask.any())
    # +y edge => teacher should move toward lower y on average.
    delta_y = teacher.teacher_trajectory[..., 1] - old[..., 1]
    assert float(delta_y.mean()) < 0.0
