from __future__ import annotations

import numpy as np

from metadrive.policy.diffusion_policy.mode_context import DynamicObstacle, ModeContext, build_mode_context_from_sample
from metadrive.policy.diffusion_policy.mode_trajectory_generator import ModeTrajectoryGenerator


def _straight_polyline(y_offset: float = 0.0, length: float = 60.0, num_points: int = 31) -> np.ndarray:
    xs = np.linspace(0.0, length, num_points, dtype=np.float32)
    ys = np.full((num_points,), y_offset, dtype=np.float32)
    return np.stack([xs, ys], axis=1)


def _branch_polyline(direction: str = "left", length: float = 40.0, num_points: int = 31) -> np.ndarray:
    xs = np.linspace(0.0, length, num_points, dtype=np.float32)
    sign = 1.0 if direction == "left" else -1.0
    ys = sign * np.linspace(0.0, 8.0, num_points, dtype=np.float32)
    return np.stack([xs, ys], axis=1)


def _context(**overrides) -> ModeContext:
    base = dict(
        ego_speed_mps=10.0,
        ego_heading=0.0,
        current_lane_polyline=_straight_polyline(0.0),
        current_lane_width=4.0,
        left_lane_polyline=_straight_polyline(4.0),
        right_lane_polyline=_straight_polyline(-4.0),
        left_branch_polyline=_branch_polyline("left"),
        right_branch_polyline=_branch_polyline("right"),
        front_object_distance=18.0,
        front_object_speed_mps=8.0,
        left_lane_gap=20.0,
        right_lane_gap=20.0,
        current_ref_lane_count=2,
        next_ref_lane_count=2,
        has_left_adjacent=True,
        has_right_adjacent=True,
        has_left_branch=True,
        has_right_branch=True,
        dynamic_obstacles=(),
    )
    base.update(overrides)
    return ModeContext(**base)


def test_generator_marks_emergency_stop_always_valid_and_stationary_lateral():
    generator = ModeTrajectoryGenerator()

    output = generator.generate(_context(left_lane_polyline=None, has_left_adjacent=False))

    assert output.mode_valid_mask[9]
    np.testing.assert_allclose(output.coarse_trajectories[9][:, 1], 0.0)
    assert np.all(np.diff(output.coarse_trajectories[9][:, 0]) >= -1e-6)


def test_generator_lane_change_left_ends_near_left_lane_center():
    generator = ModeTrajectoryGenerator()

    output = generator.generate(_context())

    lane_change = output.coarse_trajectories[4]
    assert output.mode_valid_mask[3]
    assert lane_change[-1, 1] > 3.0
    assert lane_change[-1, 0] > lane_change[0, 0]


def test_generator_lane_change_speed_profiles_scale_longitudinal_progress():
    generator = ModeTrajectoryGenerator()

    output = generator.generate(_context())

    left_high = output.coarse_trajectories[3]
    left_medium = output.coarse_trajectories[4]
    left_low = output.coarse_trajectories[5]

    assert output.mode_valid_mask[3]
    assert output.mode_valid_mask[4]
    assert output.mode_valid_mask[5]
    assert left_high[-1, 0] >= left_medium[-1, 0] >= left_low[-1, 0]


def test_generator_uses_branch_polyline_when_left_adjacent_lane_is_missing():
    generator = ModeTrajectoryGenerator()

    output = generator.generate(_context(left_lane_polyline=None, has_left_adjacent=False, has_left_branch=True))

    assert output.mode_valid_mask[3]
    assert output.mode_valid_mask[4]
    assert output.mode_valid_mask[5]
    assert float(output.coarse_trajectories[4][-1, 1]) > 3.0


def test_generator_filters_collision_with_constant_velocity_prediction():
    # Obstacle at x=20m, approaching at 5 m/s.  KEEP_LANE_HIGH and KEEP_LANE_MEDIUM
    # are subject to collision checking (they become invalid when the trajectory
    # intersects the obstacle).  KEEP_LANE_LOW and EMERGENCY_STOP always remain valid
    # so the vehicle always has at least one actionable mode.
    obstacle = DynamicObstacle(
        initial_position_xy=np.asarray([20.0, 0.0], dtype=np.float32),
        velocity_xy=np.asarray([-5.0, 0.0], dtype=np.float32),
        radius_m=1.0,
    )
    generator = ModeTrajectoryGenerator(
        collision_speed_scale_candidates=(1.0, 0.5, 0.25),
        ego_collision_radius_m=1.0,
        collision_check_clearance_m=0.2,
    )

    output = generator.generate(_context(dynamic_obstacles=(obstacle,)))

    # KEEP_LANE_HIGH (slot 0): now subject to collision checking; at 13 m/s
    # the trajectory reaches the obstacle → invalid.
    assert not output.mode_valid_mask[0]
    # KEEP_LANE_LOW (slot 2): always valid regardless of collisions.
    assert output.mode_valid_mask[2]


def test_generator_keeps_emergency_stop_valid_even_when_other_modes_collide():
    obstacle = DynamicObstacle(
        initial_position_xy=np.asarray([0.0, 0.0], dtype=np.float32),
        velocity_xy=np.asarray([0.0, 0.0], dtype=np.float32),
        radius_m=5.0,
    )
    generator = ModeTrajectoryGenerator()

    output = generator.generate(_context(dynamic_obstacles=(obstacle,)))

    assert output.mode_valid_mask[9]
    np.testing.assert_allclose(output.coarse_trajectories[9][:, 1], 0.0)


def test_build_mode_context_from_sample_falls_back_to_reference_poses():
    sample = {
        "reference_pose_world": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "future_reference_pose_world": np.asarray(
            [
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "ego_speed_km_h": np.asarray(36.0, dtype=np.float32),
    }

    ctx = build_mode_context_from_sample(sample)

    assert ctx.current_lane_polyline.shape == (4, 2)
    np.testing.assert_allclose(ctx.current_lane_polyline[0], np.asarray([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(ctx.current_lane_polyline[-1], np.asarray([6.0, 0.0], dtype=np.float32))
