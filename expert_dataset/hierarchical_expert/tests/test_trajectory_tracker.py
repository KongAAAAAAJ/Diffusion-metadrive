from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from metadrive.component.lane.point_lane import PointLane
from metadrive.component.lane.straight_lane import StraightLane
from expert_dataset.expert_idm_policy import MockVehicle, _simulate_step
from expert_dataset.hierarchical_expert.driving_style import DrivingStyleProfile
from expert_dataset.hierarchical_expert.trajectory_planner import QuinticLaneChangePlanner
from expert_dataset.hierarchical_expert.trajectory_tracker import PurePursuitTracker


def build_point_lane(y_offset: float = 0.0, length: float = 80.0, samples: int = 100) -> PointLane:
    points = np.stack(
        [
            np.linspace(0.0, length, samples, dtype=np.float64),
            np.full(samples, y_offset, dtype=np.float64),
        ],
        axis=1,
    )
    return PointLane(center_line_points=points, width=3.7)


def build_parallel_straight_lanes():
    source = StraightLane([0.0, 0.0], [80.0, 0.0], width=3.7)
    target = StraightLane([0.0, 3.7], [80.0, 3.7], width=3.7)
    return source, target


def simulate_tracker(vehicle: MockVehicle, trajectory, tracker: PurePursuitTracker, steps: int = 50, dt: float = 0.05):
    lateral_errors = []
    steering_history = []
    for _ in range(steps):
        current_s, current_lat = trajectory.local_coordinates(vehicle.position)
        if current_s >= trajectory.length:
            break
        steering = tracker.compute_steering(vehicle.position, vehicle.heading_theta, vehicle.speed, trajectory)
        steering_history.append(float(steering))
        lateral_errors.append(float(current_lat))
        _simulate_step(vehicle, steering, dt=dt, wheel_base=tracker.PP_WHEELBASE)
    return np.asarray(steering_history), np.asarray(lateral_errors)


def has_oscillation(steering_history, threshold=0.05, max_sign_changes=4):
    if len(steering_history) < 3:
        return False
    deltas = np.diff(steering_history)
    significant = deltas[np.abs(deltas) > threshold]
    if len(significant) < 2:
        return False
    sign_changes = np.sum(np.diff(np.sign(significant)) != 0)
    return sign_changes > max_sign_changes


def test_straight_trajectory_centerline_tracking_returns_near_zero_steering():
    tracker = PurePursuitTracker()
    trajectory = build_point_lane()
    steering = tracker.compute_steering(
        ego_position=np.asarray([5.0, 0.0], dtype=np.float64),
        ego_heading=0.0,
        ego_speed=8.0,
        trajectory=trajectory,
    )

    assert abs(steering) < 0.01


def test_tracker_uses_conservative_lookahead_defaults():
    tracker = PurePursuitTracker()

    assert tracker.PP_LOOKAHEAD_SPEED_GAIN == 0.65
    assert tracker.PP_MIN_LOOKAHEAD == 3.5
    assert tracker.PP_MAX_LOOKAHEAD == 10.0
    assert tracker.MAX_STEER_DELTA_PER_STEP == 0.025


def test_straight_trajectory_offset_converges_within_50_steps():
    tracker = PurePursuitTracker()
    trajectory = build_point_lane()
    vehicle = MockVehicle(position=np.asarray([5.0, 1.0], dtype=np.float64), heading_theta=0.0, speed=8.0)

    steering_history, lateral_errors = simulate_tracker(vehicle, trajectory, tracker, steps=50)

    assert steering_history[0] < 0.0
    assert abs(lateral_errors[-1]) < 0.3


def test_lane_change_trajectory_is_bounded_and_non_oscillatory():
    planner = QuinticLaneChangePlanner(DrivingStyleProfile())
    tracker = PurePursuitTracker()
    source_lane, target_lane = build_parallel_straight_lanes()
    ego_position = np.asarray(source_lane.position(5.0, 0.0), dtype=np.float64)
    trajectory = planner.plan(
        ego_position=ego_position,
        ego_heading=0.0,
        ego_speed=8.0,
        source_lane=source_lane,
        target_lane=target_lane,
        direction=-1,
        urgency=0.0,
    )
    vehicle = MockVehicle(position=ego_position.copy(), heading_theta=0.0, speed=8.0)

    steering_history, _ = simulate_tracker(vehicle, trajectory, tracker, steps=80)

    assert np.all(np.abs(steering_history) <= tracker.MAX_STEERING + 1e-6)
    assert not has_oscillation(steering_history)


def test_low_speed_tracking_returns_finite_steering():
    tracker = PurePursuitTracker()
    trajectory = build_point_lane(y_offset=1.0)
    steering = tracker.compute_steering(
        ego_position=np.asarray([0.0, 0.0], dtype=np.float64),
        ego_heading=0.0,
        ego_speed=2.0,
        trajectory=trajectory,
    )

    assert np.isfinite(steering)


def test_high_speed_tracking_remains_smooth_and_bounded():
    tracker = PurePursuitTracker()
    trajectory = build_point_lane(y_offset=1.0)
    vehicle = MockVehicle(position=np.asarray([5.0, 0.0], dtype=np.float64), heading_theta=0.0, speed=30.0)

    steering_history, _ = simulate_tracker(vehicle, trajectory, tracker, steps=20)

    assert np.all(np.isfinite(steering_history))
    assert np.all(np.abs(steering_history) <= tracker.MAX_STEERING + 1e-6)


def test_trajectory_end_detection_triggers_near_end():
    tracker = PurePursuitTracker()
    trajectory = build_point_lane(length=20.0)

    assert tracker.is_trajectory_ended(np.asarray([19.5, 0.0], dtype=np.float64), trajectory)
    assert not tracker.is_trajectory_ended(np.asarray([10.0, 0.0], dtype=np.float64), trajectory)


def test_lookahead_is_inflated_near_trajectory_end():
    tracker = PurePursuitTracker()
    trajectory = build_point_lane(length=20.0)

    normal = tracker._compute_lookahead(ego_speed=8.0, current_s=5.0, trajectory_length=20.0)
    near_end = tracker._compute_lookahead(ego_speed=8.0, current_s=15.0, trajectory_length=20.0)

    assert near_end > normal
    assert near_end == min(normal * 1.25, tracker.PP_MAX_LOOKAHEAD)


def test_steering_rate_limit_caps_single_step_change():
    tracker = PurePursuitTracker()
    trajectory = build_point_lane(y_offset=3.0)

    first = tracker.compute_steering(
        ego_position=np.asarray([0.0, 0.0], dtype=np.float64),
        ego_heading=0.0,
        ego_speed=8.0,
        trajectory=trajectory,
    )
    second = tracker.compute_steering(
        ego_position=np.asarray([0.0, 0.0], dtype=np.float64),
        ego_heading=0.0,
        ego_speed=8.0,
        trajectory=trajectory,
    )

    assert abs(second - first) <= tracker.MAX_STEER_DELTA_PER_STEP + 1e-6


def test_abort_trajectory_tracking_is_smooth_and_without_overshoot():
    planner = QuinticLaneChangePlanner(DrivingStyleProfile())
    tracker = PurePursuitTracker()
    source_lane, _ = build_parallel_straight_lanes()
    ego_position = np.asarray(source_lane.position(10.0, 1.6), dtype=np.float64)
    trajectory = planner.plan_abort(
        ego_position=ego_position,
        ego_heading=0.0,
        ego_speed=8.0,
        source_lane=source_lane,
    )
    vehicle = MockVehicle(position=ego_position.copy(), heading_theta=0.0, speed=8.0)

    steering_history, lateral_errors = simulate_tracker(vehicle, trajectory, tracker, steps=80)

    assert np.all(np.abs(steering_history) <= tracker.MAX_STEERING + 1e-6)
    assert not has_oscillation(steering_history)
    assert abs(lateral_errors[-1]) < 0.5
