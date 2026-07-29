"""Tracking-error envelopes and false-safe attribution for joint branches."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


class TrackingDiagnosticError(RuntimeError):
    """Raised when tracking diagnostics violate their strict contract."""


@dataclass(frozen=True)
class TrackingEnvelope:
    longitudinal_margin_m: float
    lateral_margin_m: float
    heading_margin_rad: float
    raw_longitudinal_p99_m: float
    raw_lateral_p99_m: float
    raw_heading_p99_rad: float
    within_controller_limits: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "longitudinal_margin_m": self.longitudinal_margin_m,
            "lateral_margin_m": self.lateral_margin_m,
            "heading_margin_rad": self.heading_margin_rad,
            "raw_longitudinal_p99_m": self.raw_longitudinal_p99_m,
            "raw_lateral_p99_m": self.raw_lateral_p99_m,
            "raw_heading_p99_rad": self.raw_heading_p99_rad,
            "within_controller_limits": self.within_controller_limits,
        }


def derive_tracking_envelope(
    longitudinal_errors_m: np.ndarray,
    lateral_errors_m: np.ndarray,
    heading_errors_rad: np.ndarray,
) -> TrackingEnvelope:
    values = []
    for name, source in (
        ("longitudinal", longitudinal_errors_m),
        ("lateral", lateral_errors_m),
        ("heading", heading_errors_rad),
    ):
        array = np.asarray(source, dtype=np.float64).reshape(-1)
        if array.size == 0 or not np.isfinite(array).all():
            raise TrackingDiagnosticError(
                f"{name} tracking errors must be non-empty and finite"
            )
        values.append(float(np.percentile(np.abs(array), 99)))
    longitudinal, lateral, heading = values
    return TrackingEnvelope(
        longitudinal_margin_m=min(longitudinal + 0.2, 1.5),
        lateral_margin_m=min(lateral + 0.2, 1.0),
        heading_margin_rad=min(heading + 0.02, 0.15),
        raw_longitudinal_p99_m=longitudinal,
        raw_lateral_p99_m=lateral,
        raw_heading_p99_rad=heading,
        within_controller_limits=(
            longitudinal <= 1.5 and lateral <= 1.0 and heading <= 0.15
        ),
    )


def tracking_percentiles(values: np.ndarray) -> dict[str, float]:
    array = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    if array.size == 0 or not np.isfinite(array).all():
        raise TrackingDiagnosticError(
            "tracking percentile input must be non-empty and finite"
        )
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def classify_false_safe(
    *,
    simulator_collision: bool,
    simulator_out_of_drivable: bool,
    failure_reasons: Sequence[str],
    replay_position_error_m: float,
    replay_heading_error_rad: float,
    maximum_lateral_error_m: float,
    maximum_heading_error_rad: float,
    minimum_platoon_gap_m: float,
    minimum_background_gap_m: float,
) -> str:
    finite = (
        replay_position_error_m,
        replay_heading_error_rad,
        maximum_lateral_error_m,
        maximum_heading_error_rad,
        minimum_platoon_gap_m,
        minimum_background_gap_m,
    )
    if not all(math.isfinite(float(value)) for value in finite):
        raise TrackingDiagnosticError("false-safe inputs must be finite")
    reasons = tuple(str(value) for value in failure_reasons)
    if replay_position_error_m > 0.01 or replay_heading_error_rad > 0.01:
        return "replay_or_state_mismatch"
    if any("agent_removed_without_failure_flag" in value for value in reasons):
        return "termination_semantics"
    if maximum_lateral_error_m > 0.5 or maximum_heading_error_rad > 0.1:
        return "controller_tracking"
    if simulator_out_of_drivable:
        return "road_footprint"
    if simulator_collision and minimum_background_gap_m <= 0.0:
        return "background_prediction"
    if simulator_collision and minimum_platoon_gap_m <= 0.0:
        return "platoon_interaction"
    return "termination_semantics"


def aggregate_tracking_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise TrackingDiagnosticError("tracking report has no rows")
    long_values: list[float] = []
    lateral_values: list[float] = []
    heading_values: list[float] = []
    by_scenario_role: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        scenario = str(row["scenario"])
        role = int(row["role"])
        key = f"{scenario}/agent{role}"
        bucket = by_scenario_role.setdefault(
            key, {"longitudinal": [], "lateral": [], "heading": []}
        )
        for target, field, aggregate in (
            ("longitudinal", "longitudinal_errors_m", long_values),
            ("lateral", "lateral_errors_m", lateral_values),
            ("heading", "heading_errors_rad", heading_values),
        ):
            array = np.asarray(row[field], dtype=np.float64).reshape(-1)
            if array.size == 0 or not np.isfinite(array).all():
                raise TrackingDiagnosticError(
                    f"{key} contains invalid {target} tracking values"
                )
            absolute = np.abs(array).tolist()
            bucket[target].extend(absolute)
            aggregate.extend(absolute)
    envelope = derive_tracking_envelope(
        np.asarray(long_values),
        np.asarray(lateral_values),
        np.asarray(heading_values),
    )
    return {
        "overall": {
            "longitudinal_m": tracking_percentiles(np.asarray(long_values)),
            "lateral_m": tracking_percentiles(np.asarray(lateral_values)),
            "heading_rad": tracking_percentiles(np.asarray(heading_values)),
        },
        "by_scenario_role": {
            key: {
                "longitudinal_m": tracking_percentiles(
                    np.asarray(bucket["longitudinal"])
                ),
                "lateral_m": tracking_percentiles(
                    np.asarray(bucket["lateral"])
                ),
                "heading_rad": tracking_percentiles(
                    np.asarray(bucket["heading"])
                ),
            }
            for key, bucket in sorted(by_scenario_role.items())
        },
        "recommended_envelope": envelope.as_dict(),
    }


__all__ = [
    "TrackingDiagnosticError",
    "TrackingEnvelope",
    "aggregate_tracking_rows",
    "classify_false_safe",
    "derive_tracking_envelope",
    "tracking_percentiles",
]
