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
    BRAKE_ACCELERATION_SCALE_MPS2,
    DRIVE_ACCELERATION_SCALE_MPS2,
    EXECUTABLE_MAX_ACCEL_MPS2,
    EXECUTABLE_MIN_ACCEL_MPS2,
    PROFILE_ACCELERATION_GUARD_MPS2,
    LongitudinalReferenceError,
    LongitudinalCascadeController,
    LongitudinalTrackingReference,
    build_feedback_executable_profile,
    project_point_to_path_arc,
    sample_path_at_arc,
    signed_longitudinal_speed_mps,
    trajectory_to_longitudinal_reference,
)


def test_executable_authority_uses_latest_asymmetric_actuator_calibration() -> None:
    assert EXECUTABLE_MAX_ACCEL_MPS2 == pytest.approx(
        DRIVE_ACCELERATION_SCALE_MPS2
    )
    assert EXECUTABLE_MIN_ACCEL_MPS2 == pytest.approx(
        max(-8.0, -BRAKE_ACCELERATION_SCALE_MPS2)
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


def test_feedback_governor_keeps_float32_guard_inside_hard_brake_boundary() -> None:
    times = np.arange(81, dtype=np.float64) * 0.1
    arc = np.zeros_like(times)
    reference = build_feedback_executable_profile(
        path_times_s=times,
        path_arc_m=arc,
        elapsed_s=0.0,
        actual_arc_m=0.0,
        actual_speed_mps=20.0,
    )

    assert np.min(reference.acceleration_mps2) == pytest.approx(
        EXECUTABLE_MIN_ACCEL_MPS2 + PROFILE_ACCELERATION_GUARD_MPS2
    )


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
    expected_speed, _ = reference.sample_speed_acceleration_at_time(0.3)
    assert debug["reference_speed_mps"] == pytest.approx(expected_speed)
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
    expected_speed, _ = reference.sample_speed_acceleration_at_time(0.3)
    assert debug["reference_speed_mps"] == pytest.approx(expected_speed)
    assert throttle == pytest.approx(0.0)


def test_signed_longitudinal_speed_preserves_reverse_direction() -> None:
    vehicle = type(
        "Vehicle",
        (),
        {
            "velocity": np.asarray([-2.0, 0.0]),
            "heading": np.asarray([1.0, 0.0]),
            "speed_km_h": 7.2,
        },
    )()
    assert vehicle.speed_km_h > 0.0
    assert signed_longitudinal_speed_mps(vehicle) == pytest.approx(-2.0)


def test_time_preview_does_not_mix_arc_position_error_into_speed() -> None:
    reference = trajectory_to_longitudinal_reference(
        _trajectory(4.0), 4.0, source="preview"
    )
    speed, acceleration = reference.sample_speed_acceleration_at_time(0.3)
    assert speed == pytest.approx(4.0)
    assert acceleration == pytest.approx(0.0)


def test_acceleration_bias_compensates_zero_acceleration_mapping() -> None:
    reference = trajectory_to_longitudinal_reference(
        _trajectory(4.0), 4.0, source="bias"
    )
    default_throttle, default_debug = LongitudinalCascadeController().compute(
        "agent0", 4.0, reference
    )
    corrected_throttle, corrected_debug = LongitudinalCascadeController(
        acceleration_bias_mps2=-0.08
    ).compute("agent0", 4.0, reference)
    assert default_throttle == pytest.approx(0.0)
    assert default_debug["acceleration_bias_mps2"] == pytest.approx(0.0)
    assert corrected_debug["desired_acceleration_mps2"] == pytest.approx(0.0)
    assert corrected_debug["compensated_acceleration_mps2"] == pytest.approx(0.08)
    assert corrected_debug["acceleration_bias_mps2"] == pytest.approx(-0.08)
    assert corrected_throttle == pytest.approx(
        0.08 / DRIVE_ACCELERATION_SCALE_MPS2
    )


def test_brake_mapping_uses_independent_calibrated_scale() -> None:
    trajectory = np.zeros((8, 3), dtype=np.float32)
    trajectory[:, 0] = np.asarray(
        [3.5, 6.0, 7.5, 8.0, 8.0, 8.0, 8.0, 8.0], dtype=np.float32
    )
    reference = trajectory_to_longitudinal_reference(
        trajectory, 8.0, source="brake-scale"
    )
    calibrated, calibrated_debug = LongitudinalCascadeController(
        brake_acceleration_scale_mps2=10.0
    ).compute("agent0", 8.0, reference)
    legacy, _ = LongitudinalCascadeController(
        brake_acceleration_scale_mps2=2.6
    ).compute("agent0", 8.0, reference)
    assert calibrated_debug["desired_acceleration_mps2"] < 0.0
    assert calibrated_debug["brake_acceleration_scale_mps2"] == pytest.approx(10.0)
    assert calibrated == pytest.approx(
        calibrated_debug["compensated_acceleration_mps2"] / 10.0
    )
    assert abs(calibrated) < abs(legacy)


def test_drive_mapping_uses_independent_calibrated_scale() -> None:
    reference = trajectory_to_longitudinal_reference(
        _trajectory(6.0), 4.0, source="drive-scale"
    )
    throttle, debug = LongitudinalCascadeController(
        drive_acceleration_scale_mps2=1.6
    ).compute("agent0", 4.0, reference)
    assert debug["desired_acceleration_mps2"] > 0.0
    assert debug["drive_acceleration_scale_mps2"] == pytest.approx(1.6)
    assert throttle == pytest.approx(
        debug["compensated_acceleration_mps2"] / 1.6
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"drive_acceleration_scale_mps2": 0.0},
        {"brake_acceleration_scale_mps2": 0.0},
    ),
)
def test_actuator_scales_must_be_strictly_positive(kwargs) -> None:
    with pytest.raises(LongitudinalReferenceError, match="timing and limits"):
        LongitudinalCascadeController(**kwargs)


def test_drive_and_brake_pi_use_separate_regimes_and_stop_overzero_guard() -> None:
    controller = LongitudinalCascadeController()
    cruise = trajectory_to_longitudinal_reference(
        _trajectory(6.0), 4.0, source="drive"
    )
    _, drive = controller.compute("agent0", 4.0, cruise)
    assert drive["control_regime"] == "acceleration"

    braking_trajectory = np.zeros((8, 3), dtype=np.float32)
    braking_trajectory[:, 0] = np.asarray(
        [3.5, 6.0, 7.5, 8.0, 8.0, 8.0, 8.0, 8.0], dtype=np.float32
    )
    brake = trajectory_to_longitudinal_reference(
        braking_trajectory, 8.0, source="brake"
    )
    _, brake_debug = controller.compute("agent0", 8.0, brake)
    assert brake_debug["control_regime"] == "braking"

    stationary = trajectory_to_longitudinal_reference(
        np.zeros((8, 3), dtype=np.float32), 0.0, source="stop"
    )
    throttle, stopped = controller.compute("agent0", 0.05, stationary)
    assert stopped["speed_overzero_guard"] is True
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
