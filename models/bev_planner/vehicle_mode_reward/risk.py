"""Continuous safety-risk transforms and temporal aggregation."""

from __future__ import annotations

import math

import numpy as np

from .config import JointRewardError

def soft_threshold_risk(
    values: np.ndarray,
    *,
    warning_threshold: float,
    softness: float,
) -> np.ndarray:
    """Map a larger-is-safer metric to a stable continuous risk in [0,1]."""

    metric = np.asarray(values, dtype=np.float64)
    threshold = float(warning_threshold)
    width = float(softness)
    if (
        not np.isfinite(metric).all()
        or not math.isfinite(threshold)
        or threshold <= 0.0
        or not math.isfinite(width)
        or width <= 0.0
    ):
        raise JointRewardError("soft risk inputs and parameters must be finite/positive")
    scaled = (threshold - metric) / width
    softplus = np.maximum(scaled, 0.0) + np.log1p(
        np.exp(-np.abs(scaled))
    )
    return np.clip((width / threshold) * softplus, 0.0, 1.0)


def closing_ttc_from_gap_series(
    gap_series: np.ndarray,
    *,
    dt_s: float,
    closing_speed_epsilon_mps: float,
    no_risk_gap_m: float,
    no_risk_ttc_s: float,
) -> np.ndarray:
    """Return finite TTC using adjacent finite shared-corridor gap samples."""

    gap = np.asarray(gap_series, dtype=np.float64)
    if gap.ndim != 1 or not np.isfinite(gap).all() or gap.size == 0:
        raise JointRewardError("gap_series must be a non-empty finite vector")
    dt = float(dt_s)
    epsilon = float(closing_speed_epsilon_mps)
    gap_sentinel = float(no_risk_gap_m)
    sentinel = float(no_risk_ttc_s)
    if (
        not math.isfinite(dt)
        or dt <= 0.0
        or not math.isfinite(epsilon)
        or epsilon <= 0.0
        or not math.isfinite(gap_sentinel)
        or gap_sentinel <= 0.0
        or not math.isfinite(sentinel)
        or sentinel <= 0.0
    ):
        raise JointRewardError("TTC parameters must be positive and finite")
    ttc = np.full(gap.shape, sentinel, dtype=np.float64)
    closing_speed = (gap[:-1] - gap[1:]) / dt
    adjacent_shared_corridor = (
        (gap[:-1] < gap_sentinel) & (gap[1:] < gap_sentinel)
    )
    closing = adjacent_shared_corridor & (closing_speed > epsilon)
    indices = np.flatnonzero(closing) + 1
    if len(indices):
        ttc[indices] = np.maximum(gap[indices], 0.0) / closing_speed[closing]
    return ttc


def ttc_risk_from_gap_series(
    gap_series: np.ndarray,
    *,
    dt_s: float,
    warning_threshold_s: float,
    softness_s: float,
    closing_speed_epsilon_mps: float,
    no_risk_gap_m: float,
    no_risk_ttc_s: float,
) -> np.ndarray:
    ttc = closing_ttc_from_gap_series(
        gap_series,
        dt_s=dt_s,
        closing_speed_epsilon_mps=closing_speed_epsilon_mps,
        no_risk_gap_m=no_risk_gap_m,
        no_risk_ttc_s=no_risk_ttc_s,
    )
    return soft_threshold_risk(
        ttc,
        warning_threshold=warning_threshold_s,
        softness=softness_s,
    )


def aggregate_temporal_risk(
    per_time_risk: np.ndarray,
    *,
    max_weight: float,
    mean_weight: float,
) -> float:
    values = np.asarray(per_time_risk, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise JointRewardError("temporal risk must be a non-empty [0,1] vector")
    if (
        not math.isfinite(max_weight)
        or not math.isfinite(mean_weight)
        or max_weight < 0.0
        or mean_weight < 0.0
        or not math.isclose(max_weight + mean_weight, 1.0)
    ):
        raise JointRewardError("temporal aggregation weights must sum to one")
    return float(max_weight * np.max(values) + mean_weight * np.mean(values))


# Preserve the historical public re-export used by callers.
from models.platoon_planner.collision_geometry import shared_corridor_gap_series
