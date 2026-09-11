from __future__ import annotations

import torch

from train.bev_risk_pact.config import RiskPACTConfig
from train.bev_risk_pact.risk_field import DynamicActorClearanceRiskField, MultiSourceSafetyRiskField


def _actors(count: int, *, x: float, y: float, length: float = 5.74, width: float = 2.30):
    actor = torch.zeros((1, 3, count, 8), dtype=torch.float32)
    actor[..., 0] = x
    actor[..., 1] = y
    actor[..., 3] = 1.0
    actor[..., 6] = length
    actor[..., 7] = width
    valid = torch.ones((1, 3, count), dtype=torch.bool)
    return actor, valid


def test_vehicle_geometry_defaults_match_reward_contract() -> None:
    cfg = RiskPACTConfig()
    assert cfg.ego_length_m == 5.74
    assert cfg.ego_width_m == 2.30
    assert cfg.platoon_vehicle_length_m == 5.74
    assert cfg.platoon_vehicle_width_m == 2.30


def test_actor_risk_does_not_increase_just_because_actor_is_duplicated() -> None:
    cfg = RiskPACTConfig(
        use_background_actor=True,
        use_platoon_actor=False,
        use_road_boundary=False,
        background_longitudinal_clearance_m=1.0,
        background_lateral_clearance_m=0.2,
    )
    traj = torch.zeros((1, 3, 1, 8, 2), dtype=torch.float32)
    one, one_valid = _actors(1, x=10.0, y=0.0)
    many, many_valid = _actors(16, x=10.0, y=0.0)
    field = DynamicActorClearanceRiskField(cfg)
    r1 = field.query(
        traj, one, one_valid,
        longitudinal_clearance_m=cfg.background_longitudinal_clearance_m,
        lateral_clearance_m=cfg.background_lateral_clearance_m,
    ).risk
    r16 = field.query(
        traj, many, many_valid,
        longitudinal_clearance_m=cfg.background_longitudinal_clearance_m,
        lateral_clearance_m=cfg.background_lateral_clearance_m,
    ).risk
    assert torch.allclose(r1, r16, atol=1e-6)


def test_adjacent_lane_clearance_is_low_with_calibrated_box_geometry() -> None:
    cfg = RiskPACTConfig(
        use_background_actor=True,
        use_platoon_actor=False,
        use_road_boundary=False,
        background_longitudinal_clearance_m=0.0,
        background_lateral_clearance_m=0.4,
        actor_temperature_m=0.5,
    )
    traj = torch.zeros((1, 3, 1, 8, 2), dtype=torch.float32)
    actor, valid = _actors(1, x=0.0, y=3.5)
    result = DynamicActorClearanceRiskField(cfg).query(
        traj, actor, valid,
        longitudinal_clearance_m=cfg.background_longitudinal_clearance_m,
        lateral_clearance_m=cfg.background_lateral_clearance_m,
    )
    # Safety half-width = 1.15 + 1.15 + 0.4 = 2.7 m, so 3.5 m lane separation
    # leaves ~0.8 m clearance and should remain below the 0.35 threshold.
    assert float(result.risk.mean()) < cfg.risk_threshold


def test_actor_risk_is_half_at_declared_safety_boundary() -> None:
    cfg = RiskPACTConfig(
        use_background_actor=True,
        use_platoon_actor=False,
        use_road_boundary=False,
        ego_length_m=4.0,
        ego_width_m=2.0,
        background_longitudinal_clearance_m=1.0,
        background_lateral_clearance_m=0.5,
        actor_temperature_m=0.5,
    )
    actor, valid = _actors(1, x=0.0, y=0.0, length=4.0, width=2.0)
    # Longitudinal safety-box boundary: 2 + 2 + 1 = 5 m.
    traj = torch.zeros((1, 3, 1, 8, 2), dtype=torch.float32)
    traj[..., 0] = 5.0
    result = DynamicActorClearanceRiskField(cfg).query(
        traj, actor, valid,
        longitudinal_clearance_m=1.0,
        lateral_clearance_m=0.5,
    )
    assert torch.allclose(result.risk, torch.full_like(result.risk, 0.5), atol=1e-5)


def test_component_smoothmax_does_not_turn_equal_safe_components_unsafe() -> None:
    cfg = RiskPACTConfig(component_softmax_beta=12.0)
    field = MultiSourceSafetyRiskField(cfg)
    x = torch.full((1, 3, 1, 8), 0.2)
    combined = field._smooth_component_max([x, x, x])
    assert torch.allclose(combined, x, atol=1e-7)
    assert float(combined.max()) < cfg.risk_threshold
