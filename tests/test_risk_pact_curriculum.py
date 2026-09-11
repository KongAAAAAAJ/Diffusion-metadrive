from dataclasses import dataclass

import pytest

from train.bev_risk_pact.curriculum import (
    risk_pact_curriculum_scale,
    risk_pact_curriculum_state,
)


@dataclass(frozen=True)
class _Cfg:
    start_scale: float = 0.2
    end_scale: float = 1.0
    warmup_updates: int = 0
    ramp_updates: int = 100
    schedule: str = "linear"


def test_default_linear_schedule_key_steps() -> None:
    cfg = _Cfg()
    assert risk_pact_curriculum_scale(0, cfg) == pytest.approx(0.2)
    assert risk_pact_curriculum_scale(50, cfg) == pytest.approx(0.6)
    assert risk_pact_curriculum_scale(100, cfg) == pytest.approx(1.0)
    assert risk_pact_curriculum_scale(200, cfg) == pytest.approx(1.0)


def test_warmup_holds_start_scale_then_ramps() -> None:
    cfg = _Cfg(warmup_updates=10, ramp_updates=20)
    assert risk_pact_curriculum_scale(0, cfg) == pytest.approx(0.2)
    assert risk_pact_curriculum_scale(9, cfg) == pytest.approx(0.2)
    assert risk_pact_curriculum_scale(10, cfg) == pytest.approx(0.2)
    assert risk_pact_curriculum_scale(20, cfg) == pytest.approx(0.6)
    assert risk_pact_curriculum_scale(30, cfg) == pytest.approx(1.0)


def test_state_exposes_absolute_and_incremental_strength() -> None:
    cfg = _Cfg()
    state0 = risk_pact_curriculum_state(0, cfg)
    assert state0.scale == pytest.approx(0.2)
    assert state0.previous_scale == pytest.approx(0.0)
    assert state0.delta_scale == pytest.approx(0.2)
    assert state0.phase == "ramp"

    state50 = risk_pact_curriculum_state(50, cfg)
    assert state50.scale == pytest.approx(0.6)
    assert state50.delta_scale == pytest.approx(0.008)
    assert state50.progress == pytest.approx(0.5)
    assert state50.phase == "ramp"

    state100 = risk_pact_curriculum_state(100, cfg)
    assert state100.scale == pytest.approx(1.0)
    assert state100.delta_scale == pytest.approx(0.008)
    assert state100.phase == "steady"

    state101 = risk_pact_curriculum_state(101, cfg)
    assert state101.scale == pytest.approx(1.0)
    assert state101.delta_scale == pytest.approx(0.0)
    assert state101.phase == "steady"


def test_rejects_invalid_step() -> None:
    cfg = _Cfg()
    with pytest.raises(ValueError):
        risk_pact_curriculum_scale(-1, cfg)
    with pytest.raises(TypeError):
        risk_pact_curriculum_scale(1.5, cfg)  # type: ignore[arg-type]
