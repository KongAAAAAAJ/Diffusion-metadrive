from __future__ import annotations

import math

import torch

from train.bev_risk_pact.config import RiskPACTConfig
from train.bev_risk_pact.risk_field import MultiSourceSafetyRiskField
from train.bev_risk_pact.road_field import build_drivable_signed_distance


def _stationary_actor(*, x: float = 0.0, y: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    actor = torch.zeros((1, 1, 1, 8), dtype=torch.float32)
    actor[..., 0] = x
    actor[..., 1] = y
    actor[..., 3] = 1.0  # cos(delta_heading)
    actor[..., 6] = 5.74
    actor[..., 7] = 2.30
    valid = torch.ones((1, 1, 1), dtype=torch.bool)
    return actor, valid


def _straight_strip(half_width_m: float = 3.0) -> torch.Tensor:
    h = w = 256
    y = 32.0 - torch.arange(w, dtype=torch.float32) / float(w - 1) * 64.0
    bev = torch.zeros((1, 3, 8, h, w), dtype=torch.uint8)
    bev[:, :, 0, :, y.abs() <= half_width_m] = 255
    return bev


def test_actor_confidence_schedule_matches_step65_defaults() -> None:
    cfg = RiskPACTConfig(
        use_background_actor=True,
        use_platoon_actor=False,
        use_road_boundary=False,
        actor_confidence_decay_enabled=True,
        actor_full_confidence_horizon_s=2.5,
        actor_confidence_decay_rate_per_s=0.60,
        actor_min_confidence=0.40,
    )
    field = MultiSourceSafetyRiskField(cfg)
    gamma = field.actor_prediction_confidence(
        8, dtype=torch.float32, device=torch.device("cpu")
    )
    expected = torch.tensor(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            math.exp(-0.60 * 0.5),
            math.exp(-0.60 * 1.0),
            math.exp(-0.60 * 1.5),
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(gamma, expected, atol=1.0e-6, rtol=1.0e-6)
    assert float(gamma[-1]) >= cfg.actor_min_confidence


def test_actor_risk_is_attenuated_after_full_confidence_horizon() -> None:
    cfg = RiskPACTConfig(
        use_background_actor=True,
        use_platoon_actor=False,
        use_road_boundary=False,
        actor_confidence_decay_enabled=True,
        actor_full_confidence_horizon_s=2.5,
        actor_confidence_decay_rate_per_s=0.60,
        actor_min_confidence=0.40,
    )
    field = MultiSourceSafetyRiskField(cfg)
    trajectory = torch.zeros((1, 1, 1, 8, 2), dtype=torch.float32)
    actor, valid = _stationary_actor(x=0.0, y=0.0)
    result = field.query(
        trajectory,
        background_actor_state=actor,
        background_actor_valid_mask=valid,
    )
    assert result.background_risk_raw is not None
    assert result.background_risk is not None
    assert result.actor_confidence is not None
    # Static geometry gives the same raw CRV risk at every step. Step 6.5 only
    # attenuates the effective actor component after 2.5 s.
    raw = result.background_risk_raw[0, 0, 0]
    effective = result.background_risk[0, 0, 0]
    gamma = result.actor_confidence[0, 0, 0]
    assert torch.allclose(raw, raw[0].expand_as(raw), atol=1.0e-6)
    assert torch.allclose(effective, raw * gamma, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(effective[:5], raw[:5], atol=1.0e-6)
    assert float(effective[-1]) < float(raw[-1])


def test_hard_long_horizon_actor_overlap_remains_safety_relevant() -> None:
    cfg = RiskPACTConfig(
        use_background_actor=True,
        use_platoon_actor=False,
        use_road_boundary=False,
        actor_confidence_decay_enabled=True,
        actor_full_confidence_horizon_s=2.5,
        actor_confidence_decay_rate_per_s=0.60,
        actor_min_confidence=0.40,
        risk_threshold=0.35,
        safe_margin=0.05,
    )
    field = MultiSourceSafetyRiskField(cfg)
    trajectory = torch.zeros((1, 1, 1, 8, 2), dtype=torch.float32)
    actor, valid = _stationary_actor(x=0.0, y=0.0)
    result = field.query(
        trajectory,
        background_actor_state=actor,
        background_actor_valid_mask=valid,
    )
    assert result.background_risk is not None
    # A near-certain geometric overlap at 4 s should not be erased by the
    # confidence decay. It remains above the level-set threshold.
    assert float(result.background_risk[0, 0, 0, -1]) > cfg.risk_threshold


def test_road_risk_is_not_attenuated_by_actor_confidence() -> None:
    base_kwargs = dict(
        use_background_actor=False,
        use_platoon_actor=False,
        use_road_boundary=True,
        road_safety_margin_m=0.4,
        road_temperature_m=0.30,
    )
    cfg_decay = RiskPACTConfig(**base_kwargs, actor_confidence_decay_enabled=True)
    cfg_no_decay = RiskPACTConfig(**base_kwargs, actor_confidence_decay_enabled=False)
    sdf = build_drivable_signed_distance(_straight_strip())
    trajectory = torch.zeros((1, 3, 1, 8, 2), dtype=torch.float32)
    trajectory[..., 0] = torch.linspace(1.0, 8.0, 8)
    trajectory[..., 1] = torch.linspace(0.0, 2.8, 8)
    decay = MultiSourceSafetyRiskField(cfg_decay).query(trajectory, road_sdf=sdf)
    no_decay = MultiSourceSafetyRiskField(cfg_no_decay).query(trajectory, road_sdf=sdf)
    assert decay.road_risk is not None and no_decay.road_risk is not None
    assert torch.allclose(decay.road_risk, no_decay.road_risk, atol=1.0e-7)
    assert torch.allclose(decay.risk, no_decay.risk, atol=1.0e-7)
