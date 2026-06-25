from __future__ import annotations

import numpy as np
import pytest

from expert_dataset.trajectory_correction import (
    TrajectoryCorrectionContext,
    TrajectoryMode,
    classify_trajectory_mode,
    correct_trajectory_geometry,
)


def test_classify_keep_lane_when_future_lane_is_stable():
    trajectory = np.asarray(
        [
            [2.0, 0.4, 0.03],
            [4.5, 0.45, 0.04],
            [7.5, 0.5, 0.03],
        ],
        dtype=np.float32,
    )
    context = TrajectoryCorrectionContext(
        current_lane_index=1,
        future_lane_indices=(1, 1, 1),
        current_ref_lane_count=3,
        next_ref_lane_count=3,
        lane_width=4.0,
        front_object_distance=35.0,
        ego_speed_km_h=28.0,
        front_object_speed_km_h=28.0,
    )

    mode = classify_trajectory_mode(trajectory, context)

    assert mode == TrajectoryMode.KEEP_LANE


def test_classify_follow_when_front_vehicle_blocks_progress():
    trajectory = np.asarray(
        [
            [1.8, 0.15, 0.02],
            [3.4, 0.12, 0.02],
            [5.0, 0.10, 0.01],
        ],
        dtype=np.float32,
    )
    context = TrajectoryCorrectionContext(
        current_lane_index=1,
        future_lane_indices=(1, 1, 1),
        current_ref_lane_count=3,
        next_ref_lane_count=3,
        lane_width=4.0,
        front_object_distance=11.0,
        ego_speed_km_h=30.0,
        front_object_speed_km_h=18.0,
    )

    mode = classify_trajectory_mode(trajectory, context)

    assert mode == TrajectoryMode.FOLLOW


def test_classify_lane_change_left_from_lane_indices():
    trajectory = np.asarray(
        [
            [2.0, -0.6, -0.05],
            [4.0, -1.5, -0.10],
            [6.2, -2.3, -0.08],
        ],
        dtype=np.float32,
    )
    context = TrajectoryCorrectionContext(
        current_lane_index=1,
        future_lane_indices=(1, 0, 0),
        current_ref_lane_count=3,
        next_ref_lane_count=3,
        lane_width=4.0,
        front_object_distance=40.0,
        ego_speed_km_h=26.0,
        front_object_speed_km_h=None,
    )

    mode = classify_trajectory_mode(trajectory, context)

    assert mode == TrajectoryMode.LANE_CHANGE_LEFT


def test_classify_merge_when_lane_count_drops():
    trajectory = np.asarray(
        [
            [2.0, -0.2, -0.02],
            [4.5, -0.9, -0.06],
            [7.2, -1.4, -0.04],
        ],
        dtype=np.float32,
    )
    context = TrajectoryCorrectionContext(
        current_lane_index=2,
        future_lane_indices=(2, 1, 1),
        current_ref_lane_count=3,
        next_ref_lane_count=2,
        lane_width=4.0,
        front_object_distance=25.0,
        ego_speed_km_h=24.0,
        front_object_speed_km_h=23.0,
    )

    mode = classify_trajectory_mode(trajectory, context)

    assert mode == TrajectoryMode.MERGE


def test_strong_correction_pulls_keep_lane_toward_centerline():
    raw_trajectory = np.asarray(
        [
            [2.0, 1.1, 0.12],
            [4.0, 1.5, 0.15],
            [6.5, 1.8, 0.18],
            [9.0, 2.1, 0.22],
        ],
        dtype=np.float32,
    )
    reference_trajectory = np.asarray(
        [
            [2.0, 0.0, 0.00],
            [4.0, 0.0, 0.00],
            [6.5, 0.0, 0.00],
            [9.0, 0.0, 0.00],
        ],
        dtype=np.float32,
    )

    corrected, metrics = correct_trajectory_geometry(
        raw_trajectory,
        TrajectoryMode.KEEP_LANE,
        reference_trajectory=reference_trajectory,
        attraction_strength=0.9,
        smoothing_strength=0.3,
    )

    assert corrected.shape == raw_trajectory.shape
    assert np.all(np.diff(corrected[:, 0]) >= -1e-6)
    assert np.abs(corrected[-1, 1]) < np.abs(raw_trajectory[-1, 1])
    assert metrics["strong_correction"] == pytest.approx(1.0)
    assert metrics["mean_abs_lateral_after"] < metrics["mean_abs_lateral_before"]


def test_weak_correction_preserves_lane_change_shape():
    raw_trajectory = np.asarray(
        [
            [2.0, -0.2, -0.02],
            [4.2, -1.1, -0.11],
            [6.4, -2.2, -0.14],
            [8.8, -3.0, -0.16],
        ],
        dtype=np.float32,
    )
    reference_trajectory = np.asarray(
        [
            [2.0, 0.0, 0.00],
            [4.2, 0.0, 0.00],
            [6.4, 0.0, 0.00],
            [8.8, 0.0, 0.00],
        ],
        dtype=np.float32,
    )

    corrected, metrics = correct_trajectory_geometry(
        raw_trajectory,
        TrajectoryMode.LANE_CHANGE_LEFT,
        reference_trajectory=reference_trajectory,
        attraction_strength=0.95,
        smoothing_strength=0.2,
    )

    assert corrected.shape == raw_trajectory.shape
    assert corrected[-1, 1] < -2.0
    assert np.all(np.diff(corrected[:, 0]) >= -1e-6)
    assert metrics["strong_correction"] == pytest.approx(0.0)
    assert abs(corrected[-1, 1] - raw_trajectory[-1, 1]) < 0.75
