from __future__ import annotations

import numpy as np
import pytest

from evaluation.joint_simulator_branch import (
    _fixed_world_longitudinal_reference,
    _local_reference_to_world,
    _world_reference_to_current_local,
)
from models.bev_planner.mode_contract import validate_trajectory_kinematics
from models.controller.longitudinal_reference import (
    EXECUTABLE_MAX_ACCEL_MPS2,
    EXECUTABLE_MIN_ACCEL_MPS2,
    LongitudinalReferenceError,
    LongitudinalCascadeController,
    LongitudinalTrackingReference,
    build_feedback_executable_profile,
    project_point_to_path_arc,
    sample_path_at_arc,
    trajectory_to_longitudinal_reference,
)


def _trajectory(speed_mps: float) -> np.ndarray:
    result = np.zeros((8, 3), dtype=np.float32)
    result[:, 0] = np.arange(1, 9, dtype=np.float32) * 0.5 * speed_mps
    return result


def test_fixed_reference_does_not_turn_two_metre_lag_into_speed() -> None:
    world = _local_reference_to_world(
        _trajectory(4.0), np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    )
    current = np.asarray([2.0, 0.0, 0.0], dtype=np.float64)
    local = _world_reference_to_current_local(world, current, elapsed_s=1.0)
    assert np.linalg.norm(local[1, :2]) == pytest.approx(6.0)
    reference = _fixed_world_longitudinal_reference(world, current, 1.0)
    assert reference.target_speed_mps == pytest.approx(4.0)
    assert reference.original_arc_error_m == pytest.approx(2.0)


def test_online_reference_uses_current_origin_and_fixed_timestamps() -> None:
    reference = trajectory_to_longitudinal_reference(
        _trajectory(6.0), 6.0, source="online_trajectory"
    )
    np.testing.assert_allclose(reference.arc_position_m, np.arange(9) * 3.0)
    np.testing.assert_allclose(reference.speed_mps, 6.0)
    assert reference.target_speed_mps == pytest.approx(6.0)


def test_feedback_governor_reanchors_small_lag_to_valid_trajectory() -> None:
    times = np.arange(81, dtype=np.float64) * 0.1
    path = np.column_stack((4.0 * times, np.zeros_like(times), np.zeros_like(times)))
    arc = path[:, 0].copy()
    actual_pose = np.asarray([0.368, 0.0, 0.0], dtype=np.float64)
    actual_arc, error = project_point_to_path_arc(actual_pose[:2], path[:, :2], arc)
    assert error == pytest.approx(0.0)
    reference = build_feedback_executable_profile(
        path_times_s=times,
        path_arc_m=arc,
        elapsed_s=0.1,
        actual_arc_m=actual_arc,
        actual_speed_mps=4.0,
    )
    output = sample_path_at_arc(path, arc, reference.arc_position_m[1:])
    audit = validate_trajectory_kinematics(output, 4.0, actual_pose)
    assert audit.valid, audit.violations
    assert reference.original_arc_error_m == pytest.approx(0.032)
    assert np.all(reference.acceleration_mps2 >= EXECUTABLE_MIN_ACCEL_MPS2)
    assert np.all(reference.acceleration_mps2 <= EXECUTABLE_MAX_ACCEL_MPS2)


def test_stop_reference_freezes_after_zero_speed() -> None:
    stop = trajectory_to_longitudinal_reference(
        np.zeros((8, 3), dtype=np.float32), 2.0, source="online_trajectory"
    )
    assert stop.stop_requested
    np.testing.assert_allclose(stop.speed_mps[1:], 0.0)
    np.testing.assert_allclose(stop.arc_position_m, 0.0)


def test_terminal_stop_does_not_request_zero_speed_before_stop_time() -> None:
    trajectory = np.zeros((8, 3), dtype=np.float32)
    times = np.arange(1, 9, dtype=np.float32) * 0.5
    active = np.minimum(times, 3.0)
    trajectory[:, 0] = 6.0 * active - active**2
    reference = trajectory_to_longitudinal_reference(
        trajectory,
        6.0,
        source="decelerating_stop",
    )
    assert reference.stop_requested
    assert reference.target_speed_mps == pytest.approx(5.9)

    throttle, debug = LongitudinalCascadeController().compute(
        "agent0",
        6.0,
        reference,
    )
    assert debug["reference_speed_mps"] == pytest.approx(5.9)
    assert throttle > -1.0


def test_terminal_stop_holds_zero_only_after_reference_reaches_zero() -> None:
    reference = trajectory_to_longitudinal_reference(
        np.zeros((8, 3), dtype=np.float32),
        0.05,
        source="stationary_stop",
    )
    throttle, debug = LongitudinalCascadeController().compute(
        "agent0",
        0.05,
        reference,
    )
    assert debug["reference_speed_mps"] == pytest.approx(0.04)
    assert throttle == pytest.approx(0.0)


def test_reference_rejects_invalid_time_or_acceleration() -> None:
    with pytest.raises(LongitudinalReferenceError):
        LongitudinalTrackingReference(
            sample_times_s=np.arange(9, dtype=np.float64),
            arc_position_m=np.arange(9, dtype=np.float64),
            speed_mps=np.ones(9),
            acceleration_mps2=np.full(9, 6.0),
            original_arc_error_m=0.0,
            stop_requested=False,
            source="test",
        )
