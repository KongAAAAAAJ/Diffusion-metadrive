from __future__ import annotations

import numpy as np

from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.lane.point_lane import PointLane
from expert_dataset.hierarchical_expert.driving_style import DrivingStyleProfile
from expert_dataset.hierarchical_expert.trajectory_planner import (
    LaneChangeTrajectoryConfig,
    QuinticLaneChangePlanner,
)


def build_parallel_lanes(width=3.7):
    source = StraightLane([0.0, 0.0], [80.0, 0.0], width=width)
    target = StraightLane([0.0, width], [80.0, width], width=width)
    return source, target


def build_curved_lanes(width=3.7):
    source = CircularLane(center=[0.0, 0.0], radius=25.0, start_phase=0.0, angle=np.pi, clockwise=False, width=width)
    outer = CircularLane(center=[0.0, 0.0], radius=25.0 + width, start_phase=0.0, angle=np.pi, clockwise=False, width=width)
    inner = CircularLane(center=[0.0, 0.0], radius=25.0 - width, start_phase=0.0, angle=np.pi, clockwise=False, width=width)
    return source, outer, inner


def sample_path(lane, num_samples=60):
    longs = np.linspace(0.0, float(lane.length), num_samples)
    return np.stack([np.asarray(lane.position(longitudinal, 0.0), dtype=np.float64) for longitudinal in longs], axis=0)


def path_geometry(points: np.ndarray):
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    heading_jumps = np.abs(np.diff(np.unwrap(headings)))
    return lengths, heading_jumps


def terminal_heading(points: np.ndarray) -> float:
    delta = points[-1] - points[-2]
    return float(np.arctan2(delta[1], delta[0]))


def wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def test_default_trajectory_config_is_more_conservative():
    config = LaneChangeTrajectoryConfig()

    assert config.min_duration == 4.0
    assert config.max_duration == 4.0


def test_default_lane_change_duration_is_fixed_to_four_seconds():
    planner = QuinticLaneChangePlanner(DrivingStyleProfile())

    assert planner._compute_duration(ego_speed=8.0, lateral_distance=3.7, urgency=0.0) == 4.0
    assert planner._compute_duration(ego_speed=20.0, lateral_distance=1.5, urgency=1.0) == 4.0


def test_quintic_coefficients_satisfy_boundary_conditions():
    planner = QuinticLaneChangePlanner(DrivingStyleProfile())
    coeffs = planner._solve_quintic_coefficients(0.0, 0.0, 0.0, 3.7, 0.0, 0.0, 4.0)

    assert coeffs.shape == (6,)
    d0, d0_dot, d0_ddot = planner._evaluate_quintic(coeffs, 0.0)
    dT, dT_dot, dT_ddot = planner._evaluate_quintic(coeffs, 4.0)

    assert abs(d0 - 0.0) < 1e-10
    assert abs(dT - 3.7) < 1e-6
    assert abs(dT_dot) < 1e-6
    assert abs(dT_ddot) < 1e-6
    assert abs(d0_dot) < 1e-10
    assert abs(d0_ddot) < 1e-10


def test_curvature_validation_distinguishes_reasonable_and_extreme_paths():
    planner = QuinticLaneChangePlanner(DrivingStyleProfile())
    good = planner._solve_quintic_coefficients(0.0, 0.0, 0.0, 3.7, 0.0, 0.0, 4.0)
    bad = planner._solve_quintic_coefficients(0.0, 0.0, 0.0, 3.7, 0.0, 0.0, 0.5)

    assert planner._validate_curvature(good, 4.0, ego_speed=8.0) is True
    assert planner._validate_curvature(bad, 0.5, ego_speed=30.0) is False


def test_sample_frenet_points_keeps_spacing_and_heading_smooth_on_curved_lane():
    planner = QuinticLaneChangePlanner(DrivingStyleProfile())
    source_lane, target_lane, _ = build_curved_lanes()
    ego_position = np.asarray(source_lane.position(5.0, 0.0), dtype=np.float64)
    longitudinal, lateral = source_lane.local_coordinates(ego_position)
    ref_position = source_lane.position(longitudinal, 0.0)
    target_s, _ = target_lane.local_coordinates(ref_position)
    target_center = target_lane.position(target_s, 0.0)
    _, target_offset = source_lane.local_coordinates(target_center)

    points = planner._sample_frenet_points(
        source_lane=source_lane,
        start_longitudinal=float(longitudinal),
        ego_speed=8.0,
        start_lateral=float(lateral),
        target_offset=float(target_offset),
        duration=4.5,
    )

    lengths, heading_jumps = path_geometry(points)

    assert points.shape[0] >= 20
    assert float(np.min(lengths)) >= 0.5
    assert float(np.max(heading_jumps)) < 0.3


def test_plan_returns_point_lane_that_starts_near_ego_and_ends_near_target_center():
    planner = QuinticLaneChangePlanner(DrivingStyleProfile())
    source_lane, target_lane = build_parallel_lanes()
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

    assert isinstance(trajectory, PointLane)
    assert np.linalg.norm(np.asarray(trajectory.position(0.0, 0.0)) - ego_position) < 1.0
    _, final_lateral = target_lane.local_coordinates(trajectory.position(trajectory.length, 0.0))
    assert abs(final_lateral) < 0.5


def test_plan_rejects_lane_change_when_too_close_to_lane_end():
    planner = QuinticLaneChangePlanner(DrivingStyleProfile())
    source_lane, target_lane = build_parallel_lanes()
    ego_position = np.asarray(source_lane.position(79.0, 0.0), dtype=np.float64)

    trajectory = planner.plan(
        ego_position=ego_position,
        ego_heading=0.0,
        ego_speed=8.0,
        source_lane=source_lane,
        target_lane=target_lane,
        direction=-1,
        urgency=0.0,
    )

    assert trajectory is None


def test_plan_supports_mirrored_right_lane_change_on_parallel_lanes():
    planner = QuinticLaneChangePlanner(DrivingStyleProfile())
    target_lane, source_lane = build_parallel_lanes()
    ego_position = np.asarray(source_lane.position(5.0, 0.0), dtype=np.float64)

    trajectory = planner.plan(
        ego_position=ego_position,
        ego_heading=0.0,
        ego_speed=8.0,
        source_lane=source_lane,
        target_lane=target_lane,
        direction=1,
        urgency=0.0,
    )

    assert isinstance(trajectory, PointLane)
    _, final_lateral = target_lane.local_coordinates(trajectory.position(trajectory.length, 0.0))
    assert abs(final_lateral) < 0.5


def test_curved_left_lane_change_to_outer_lane_ends_on_target_center_without_fold():
    planner = QuinticLaneChangePlanner(DrivingStyleProfile())
    source_lane, target_lane, _ = build_curved_lanes()
    ego_position = np.asarray(source_lane.position(5.0, 0.0), dtype=np.float64)
    ego_heading = float(source_lane.heading_theta_at(5.0))

    trajectory = planner.plan(
        ego_position=ego_position,
        ego_heading=ego_heading,
        ego_speed=8.0,
        source_lane=source_lane,
        target_lane=target_lane,
        direction=-1,
        urgency=0.0,
    )

    assert isinstance(trajectory, PointLane)
    path = sample_path(trajectory)
    end_point = path[-1]
    _, final_lateral = target_lane.local_coordinates(end_point)
    target_longitudinal, _ = target_lane.local_coordinates(end_point)
    end_heading = terminal_heading(path)
    target_heading = float(target_lane.heading_theta_at(np.clip(target_longitudinal, 0.0, float(target_lane.length))))
    heading_deltas = np.diff(np.unwrap(np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0]))))
    assert abs(final_lateral) < 0.3
    assert abs(wrap_to_pi(end_heading - target_heading)) < 0.15
    assert np.max(np.abs(heading_deltas)) < 0.2
    for point in path:
        _, src_lat = source_lane.local_coordinates(point)
        _, tgt_lat = target_lane.local_coordinates(point)
        assert min(abs(float(src_lat)), abs(float(tgt_lat))) <= source_lane.width / 2 + 0.2


def test_curved_right_lane_change_to_inner_lane_ends_on_target_center_without_fold():
    planner = QuinticLaneChangePlanner(DrivingStyleProfile())
    source_lane, _, target_lane = build_curved_lanes()
    ego_position = np.asarray(source_lane.position(5.0, 0.0), dtype=np.float64)
    ego_heading = float(source_lane.heading_theta_at(5.0))

    trajectory = planner.plan(
        ego_position=ego_position,
        ego_heading=ego_heading,
        ego_speed=8.0,
        source_lane=source_lane,
        target_lane=target_lane,
        direction=1,
        urgency=0.0,
    )

    assert isinstance(trajectory, PointLane)
    path = sample_path(trajectory)
    end_point = path[-1]
    _, final_lateral = target_lane.local_coordinates(end_point)
    target_longitudinal, _ = target_lane.local_coordinates(end_point)
    end_heading = terminal_heading(path)
    target_heading = float(target_lane.heading_theta_at(np.clip(target_longitudinal, 0.0, float(target_lane.length))))
    _, heading_jumps = path_geometry(path)

    assert abs(final_lateral) < 0.3
    assert abs(wrap_to_pi(end_heading - target_heading)) < 0.15
    assert float(np.max(heading_jumps)) < 0.3


def test_plan_abort_returns_lane_recenter_trajectory():
    planner = QuinticLaneChangePlanner(DrivingStyleProfile())
    source_lane, _ = build_parallel_lanes()
    ego_position = np.asarray(source_lane.position(10.0, 1.8), dtype=np.float64)

    trajectory = planner.plan_abort(
        ego_position=ego_position,
        ego_heading=0.0,
        ego_speed=8.0,
        source_lane=source_lane,
    )

    assert isinstance(trajectory, PointLane)
    path = sample_path(trajectory)
    _, heading_jumps = path_geometry(path)
    _, final_lateral = source_lane.local_coordinates(trajectory.position(trajectory.length, 0.0))
    terminal_lane_heading = float(source_lane.heading_theta_at(float(source_lane.local_coordinates(path[-1])[0])))
    end_heading = terminal_heading(path)
    assert abs(final_lateral) < 0.5
    assert abs(wrap_to_pi(end_heading - terminal_lane_heading)) < 0.15
    assert float(np.max(heading_jumps)) < 0.3
