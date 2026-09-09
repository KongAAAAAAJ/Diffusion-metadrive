"""Configuration and validation for vehicle-mode counterfactual rewards."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass


class JointRewardError(RuntimeError):
    """Raised when the strict reward contract is violated."""


@dataclass(frozen=True)
class VehicleModeRewardConfig:
    """Configuration for target-vehicle counterfactual trajectory rewards."""

    trajectories_per_mode: int = 48
    trajectory_dt_s: float = 0.5
    interpolation_dt_s: float = 0.1
    vehicle_length_m: float = 5.74
    vehicle_width_m: float = 2.3
    platoon_safe_gap_m: float = 7.0
    background_safe_gap_m: float = 5.0
    gap_softness_m: float = 0.5
    ttc_warning_s: float = 4.0
    ttc_softness_s: float = 0.5
    closing_speed_epsilon_mps: float = 0.1
    road_margin_warning_m: float = 1.0
    road_margin_softness_m: float = 0.25
    tracking_longitudinal_margin_m: float = 0.0
    tracking_lateral_margin_m: float = 0.0
    tracking_heading_margin_rad: float = 0.0
    progress_norm_m: float = 30.0
    progress_weight: float = 0.47
    gap_weight: float = 1.185
    ttc_weight: float = 0.5
    road_weight: float = 0.5
    comfort_weight: float = 0.0225
    collision_penalty: float = 5.0
    out_of_drivable_penalty: float = 4.0
    temporal_max_weight: float = 0.7
    temporal_mean_weight: float = 0.3
    no_risk_gap_m: float = 1.0e6
    no_risk_ttc_s: float = 1.0e6

    def __post_init__(self) -> None:
        if (
            isinstance(self.trajectories_per_mode, bool)
            or not isinstance(self.trajectories_per_mode, int)
            or self.trajectories_per_mode <= 0
        ):
            raise JointRewardError("trajectories_per_mode must be a positive integer")

        positive = (
            "trajectory_dt_s",
            "interpolation_dt_s",
            "vehicle_length_m",
            "vehicle_width_m",
            "platoon_safe_gap_m",
            "background_safe_gap_m",
            "gap_softness_m",
            "ttc_warning_s",
            "ttc_softness_s",
            "closing_speed_epsilon_mps",
            "road_margin_warning_m",
            "road_margin_softness_m",
            "progress_norm_m",
            "no_risk_gap_m",
            "no_risk_ttc_s",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise JointRewardError(f"{name} must be positive and finite")

        margin_limits = {
            "tracking_longitudinal_margin_m": 1.5,
            "tracking_lateral_margin_m": 1.0,
            "tracking_heading_margin_rad": 0.15,
        }
        for name, maximum in margin_limits.items():
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0 or value > maximum:
                raise JointRewardError(
                    f"{name} must be finite and within [0,{maximum}]"
                )

        if self.interpolation_dt_s > self.trajectory_dt_s:
            raise JointRewardError(
                "interpolation_dt_s cannot exceed trajectory_dt_s"
            )

        for name in (
            "progress_weight",
            "gap_weight",
            "ttc_weight",
            "road_weight",
            "comfort_weight",
            "collision_penalty",
            "out_of_drivable_penalty",
            "temporal_max_weight",
            "temporal_mean_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise JointRewardError(f"{name} must be non-negative and finite")

        if not math.isclose(
            self.temporal_max_weight + self.temporal_mean_weight,
            1.0,
        ):
            raise JointRewardError(
                "temporal_max_weight and temporal_mean_weight must sum to one"
            )


def vehicle_mode_reward_config_sha256(config: VehicleModeRewardConfig) -> str:
    """Return the canonical digest for the counterfactual reward settings."""

    if not isinstance(config, VehicleModeRewardConfig):
        raise JointRewardError("config must be VehicleModeRewardConfig")
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
