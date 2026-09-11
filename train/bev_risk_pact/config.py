from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RiskPACTConfig:
    """Risk-PACT-lite safety-field configuration.

    Step 5.6 calibrates the multi-source field around physically interpretable
    clearance geometry:
      * actor risk uses an oriented-box signed-clearance field rather than a
        Gaussian whose scale changes with actor count;
      * actor and component aggregation use softmax-weighted smooth maxima;
      * ego/platoon dimensions match the shared reward/environment geometry.

    Scene geometry stays detached. Gradients flow only through queried
    trajectory coordinates.
    """

    # Component switches (useful for ablations).
    use_background_actor: bool = True
    use_platoon_actor: bool = True
    use_road_boundary: bool = True

    # Shared horizon and vehicle geometry. 5.74 x 2.30 m is the common XL
    # vehicle geometry already used by VehicleModeRewardConfig / mode_contract.
    horizon_dt_s: float = 0.5
    ego_length_m: float = 5.74
    ego_width_m: float = 2.30

    # Background-traffic signed-clearance field.
    background_longitudinal_clearance_m: float = 5.0
    background_lateral_clearance_m: float = 0.4

    # Platoon-neighbor geometry and signed-clearance field.
    platoon_vehicle_length_m: float = 5.74
    platoon_vehicle_width_m: float = 2.30
    platoon_longitudinal_clearance_m: float = 7.0
    platoon_lateral_clearance_m: float = 0.4

    # Maps oriented-box signed clearance (m) to a bounded risk in (0,1).
    # At clearance=0, risk=0.5; positive clearance decreases risk.
    actor_temperature_m: float = 0.50

    # Smooth-max temperatures. Larger beta approaches a hard max while
    # avoiding actor-count / component-count inflation from probabilistic union.
    actor_softmax_beta: float = 12.0
    component_softmax_beta: float = 12.0

    # Road-boundary field from DRIVABLE signed distance.
    road_safety_margin_m: float = 0.4
    road_temperature_m: float = 0.30

    # Trajectory-level risk / PACT teacher.
    temporal_softmax_beta: float = 12.0
    risk_threshold: float = 0.35
    violation_temperature: float = 0.04
    safe_margin: float = 0.05
    teacher_step_m: float = 0.20
    gradient_eps: float = 1.0e-6
    gradient_clip_norm: float = 10.0

    def __post_init__(self) -> None:
        for name in (
            "use_background_actor",
            "use_platoon_actor",
            "use_road_boundary",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        if not (self.use_background_actor or self.use_platoon_actor or self.use_road_boundary):
            raise ValueError("at least one Risk-PACT risk component must be enabled")

        positive = (
            "horizon_dt_s",
            "ego_length_m",
            "ego_width_m",
            "platoon_vehicle_length_m",
            "platoon_vehicle_width_m",
            "actor_temperature_m",
            "actor_softmax_beta",
            "component_softmax_beta",
            "road_temperature_m",
            "temporal_softmax_beta",
            "violation_temperature",
            "teacher_step_m",
            "gradient_eps",
            "gradient_clip_norm",
        )
        non_negative = (
            "background_longitudinal_clearance_m",
            "background_lateral_clearance_m",
            "platoon_longitudinal_clearance_m",
            "platoon_lateral_clearance_m",
            "road_safety_margin_m",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in non_negative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
        if not 0.0 < float(self.risk_threshold) < 1.0:
            raise ValueError("risk_threshold must be in (0,1)")
        if not 0.0 <= float(self.safe_margin) < float(self.risk_threshold):
            raise ValueError("safe_margin must be in [0, risk_threshold)")


@dataclass(frozen=True)
class RiskPACTVisualizationConfig:
    """Training-time debug visualization controls."""

    enabled: bool = False
    interval_steps: int = 5
    start_step: int = 0
    max_events: int | None = 4
    batch_index: int = 0
    role_index: int = 0
    mode_index: int = 0
    output_subdir: str = "risk_field_visualizations"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be bool")
        if isinstance(self.interval_steps, bool) or int(self.interval_steps) <= 0:
            raise ValueError("interval_steps must be a positive integer")
        if isinstance(self.start_step, bool) or int(self.start_step) < 0:
            raise ValueError("start_step must be a non-negative integer")
        if self.max_events is not None:
            if isinstance(self.max_events, bool) or int(self.max_events) <= 0:
                raise ValueError("max_events must be None or a positive integer")
        for name in ("batch_index", "role_index", "mode_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.output_subdir, str) or not self.output_subdir.strip():
            raise ValueError("output_subdir must be a non-empty string")
