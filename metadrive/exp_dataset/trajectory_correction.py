from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


class TrajectoryMode(IntEnum):
    KEEP_LANE = 0
    FOLLOW = 1
    LANE_CHANGE_LEFT = 2
    LANE_CHANGE_RIGHT = 3
    OVERTAKE = 4
    MERGE = 5


STRONG_CORRECTION_MODES = {
    TrajectoryMode.KEEP_LANE,
    TrajectoryMode.FOLLOW,
}


@dataclass(frozen=True)
class TrajectoryCorrectionContext:
    current_lane_index: Optional[int]
    future_lane_indices: Tuple[Optional[int], ...]
    current_ref_lane_count: int
    next_ref_lane_count: Optional[int]
    lane_width: float
    front_object_distance: Optional[float]
    ego_speed_km_h: float
    front_object_speed_km_h: Optional[float] = None


def _non_none_lane_indices(indices: Sequence[Optional[int]]) -> Tuple[int, ...]:
    return tuple(int(idx) for idx in indices if idx is not None)


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _smooth_1d(values: np.ndarray, strength: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).copy()
    if arr.size <= 2:
        return arr
    alpha = float(np.clip(strength, 0.0, 0.95))
    if alpha <= 0.0:
        return arr

    for idx in range(1, arr.shape[0]):
        arr[idx] = (1.0 - alpha) * arr[idx] + alpha * arr[idx - 1]
    for idx in range(arr.shape[0] - 2, -1, -1):
        arr[idx] = (1.0 - alpha) * arr[idx] + alpha * arr[idx + 1]
    return arr


def _recompute_heading(xy: np.ndarray, fallback_heading: np.ndarray) -> np.ndarray:
    if xy.shape[0] == 1:
        return fallback_heading.astype(np.float32, copy=True)
    delta = np.diff(xy, axis=0)
    delta_heading = np.arctan2(delta[:, 1], np.maximum(delta[:, 0], 1e-3))
    heading = np.empty((xy.shape[0],), dtype=np.float32)
    heading[:-1] = delta_heading
    heading[-1] = delta_heading[-1]
    blended = 0.65 * heading + 0.35 * fallback_heading.astype(np.float32, copy=False)
    return _wrap_to_pi(blended).astype(np.float32, copy=False)


def classify_trajectory_mode(
    raw_trajectory: np.ndarray,
    context: TrajectoryCorrectionContext,
) -> TrajectoryMode:
    raw_trajectory = np.asarray(raw_trajectory, dtype=np.float32)
    lane_indices = _non_none_lane_indices(context.future_lane_indices)
    current_lane_index = context.current_lane_index
    final_lane_index = lane_indices[-1] if lane_indices else current_lane_index
    lane_delta = None
    if current_lane_index is not None and final_lane_index is not None:
        lane_delta = int(final_lane_index) - int(current_lane_index)

    next_ref_lane_count = context.next_ref_lane_count
    if (
        lane_delta is not None
        and lane_delta != 0
        and next_ref_lane_count is not None
        and next_ref_lane_count < max(int(context.current_ref_lane_count), 1)
    ):
        return TrajectoryMode.MERGE

    front_distance = context.front_object_distance
    front_speed = context.front_object_speed_km_h
    ego_speed = context.ego_speed_km_h
    blocked_by_front_vehicle = (
        front_distance is not None
        and front_distance < max(context.lane_width * 4.5, 12.0)
        and (
            front_speed is None
            or ego_speed >= front_speed - 1.0
        )
    )

    if lane_delta is not None:
        if lane_delta < 0:
            return TrajectoryMode.OVERTAKE if blocked_by_front_vehicle else TrajectoryMode.LANE_CHANGE_LEFT
        if lane_delta > 0:
            return TrajectoryMode.OVERTAKE if blocked_by_front_vehicle else TrajectoryMode.LANE_CHANGE_RIGHT

    final_lateral = float(raw_trajectory[-1, 1]) if raw_trajectory.size else 0.0
    lateral_threshold = max(context.lane_width * 0.55, 1.6)
    if final_lateral <= -lateral_threshold:
        return TrajectoryMode.OVERTAKE if blocked_by_front_vehicle else TrajectoryMode.LANE_CHANGE_LEFT
    if final_lateral >= lateral_threshold:
        return TrajectoryMode.OVERTAKE if blocked_by_front_vehicle else TrajectoryMode.LANE_CHANGE_RIGHT
    if blocked_by_front_vehicle:
        return TrajectoryMode.FOLLOW
    return TrajectoryMode.KEEP_LANE


def correct_trajectory_geometry(
    raw_trajectory: np.ndarray,
    trajectory_mode: TrajectoryMode,
    reference_trajectory: Optional[np.ndarray] = None,
    attraction_strength: float = 0.85,
    smoothing_strength: float = 0.25,
) -> tuple[np.ndarray, Dict[str, float]]:
    raw_trajectory = np.asarray(raw_trajectory, dtype=np.float32)
    corrected = raw_trajectory.copy()

    if corrected.ndim != 2 or corrected.shape[0] == 0:
        return corrected, {
            "strong_correction": 0.0,
            "mean_abs_lateral_before": 0.0,
            "mean_abs_lateral_after": 0.0,
            "final_abs_lateral_before": 0.0,
            "final_abs_lateral_after": 0.0,
            "mean_point_shift": 0.0,
        }

    reference = None if reference_trajectory is None else np.asarray(reference_trajectory, dtype=np.float32)
    strong_correction = trajectory_mode in STRONG_CORRECTION_MODES and reference is not None

    corrected[:, 0] = np.maximum.accumulate(np.maximum(corrected[:, 0], 0.0))
    if strong_correction:
        num_points = corrected.shape[0]
        weights = np.linspace(0.35, 1.0, num_points, dtype=np.float32) * float(np.clip(attraction_strength, 0.0, 1.0))
        corrected[:, 1] = (1.0 - weights) * corrected[:, 1] + weights * reference[:, 1]
        corrected[:, 1] = _smooth_1d(corrected[:, 1], smoothing_strength)
        ref_heading = np.unwrap(reference[:, 2].astype(np.float64))
        corrected_heading = np.unwrap(corrected[:, 2].astype(np.float64))
        corrected[:, 2] = _wrap_to_pi(
            ((1.0 - weights) * corrected_heading + weights * ref_heading).astype(np.float32)
        )
    else:
        corrected[:, 1] = _smooth_1d(corrected[:, 1], smoothing_strength)
        corrected[:, 2] = _smooth_1d(np.unwrap(corrected[:, 2].astype(np.float64)).astype(np.float32), smoothing_strength)
        corrected[:, 2] = _wrap_to_pi(corrected[:, 2])

    corrected[:, 0] = _smooth_1d(corrected[:, 0], min(float(smoothing_strength) * 0.5, 0.2))
    corrected[:, 0] = np.maximum.accumulate(np.maximum(corrected[:, 0], 0.0))
    corrected[:, 2] = _recompute_heading(corrected[:, :2], corrected[:, 2])

    metrics = {
        "strong_correction": float(bool(strong_correction)),
        "mean_abs_lateral_before": float(np.mean(np.abs(raw_trajectory[:, 1]))),
        "mean_abs_lateral_after": float(np.mean(np.abs(corrected[:, 1]))),
        "final_abs_lateral_before": float(np.abs(raw_trajectory[-1, 1])),
        "final_abs_lateral_after": float(np.abs(corrected[-1, 1])),
        "mean_point_shift": float(np.linalg.norm(corrected[:, :2] - raw_trajectory[:, :2], axis=1).mean()),
    }
    return corrected.astype(np.float32, copy=False), metrics
