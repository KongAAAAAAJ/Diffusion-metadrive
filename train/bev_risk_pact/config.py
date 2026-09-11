from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RiskPACTConfig:
    """Small-pilot defaults for risk-field PACT.

    The first pilot intentionally uses a simple constant-velocity actor forecast
    and center-point ego risk queries.  Vehicle footprint support can be added
    after the gradient sanity check passes.
    """

    horizon_dt_s: float = 0.5
    longitudinal_margin_m: float = 3.0
    lateral_margin_m: float = 1.2
    minimum_sigma_x_m: float = 2.5
    minimum_sigma_y_m: float = 1.2
    temporal_softmax_beta: float = 12.0
    risk_threshold: float = 0.35
    violation_temperature: float = 0.04
    safe_margin: float = 0.05
    teacher_step_m: float = 0.20
    gradient_eps: float = 1.0e-6
    gradient_clip_norm: float = 10.0

    def __post_init__(self) -> None:
        positive = (
            "horizon_dt_s",
            "longitudinal_margin_m",
            "lateral_margin_m",
            "minimum_sigma_x_m",
            "minimum_sigma_y_m",
            "temporal_softmax_beta",
            "violation_temperature",
            "teacher_step_m",
            "gradient_eps",
            "gradient_clip_norm",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < float(self.risk_threshold) < 1.0:
            raise ValueError("risk_threshold must be in (0,1)")
        if not 0.0 <= float(self.safe_margin) < float(self.risk_threshold):
            raise ValueError("safe_margin must be in [0, risk_threshold)")
