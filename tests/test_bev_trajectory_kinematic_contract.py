from __future__ import annotations

import numpy as np
import pytest

from models.bev_planner.mode_contract import (
    ModeContractError,
    validate_trajectory_kinematics,
)


def _straight(speed_mps: float) -> np.ndarray:
    times = np.arange(1, 9, dtype=np.float32) * 0.5
    return np.column_stack(
        (
            speed_mps * times,
            np.zeros_like(times),
            np.zeros_like(times),
        )
    ).astype(np.float32)


def test_local_and_world_origins_have_identical_kinematics() -> None:
    local = _straight(8.0)
    angle = 0.7
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    origin = np.asarray([13.0, -7.0, angle], dtype=np.float64)
    world = local.copy()
    world[:, :2] = local[:, :2] @ rotation.T + origin[:2]
    world[:, 2] = angle
    local_result = validate_trajectory_kinematics(
        local, 8.0, np.zeros(3)
    )
    world_result = validate_trajectory_kinematics(world, 8.0, origin)
    assert local_result.valid and world_result.valid
    np.testing.assert_allclose(
        local_result.segment_distance_m,
        world_result.segment_distance_m,
        atol=1.0e-6,
    )


def test_origin_to_first_point_and_reachability_are_enforced() -> None:
    shifted = _straight(8.0)
    shifted[0] = 0.0
    result = validate_trajectory_kinematics(shifted, 8.0, np.zeros(3))
    assert "first_waypoint_at_time_zero" in result.violations
    unreachable = _straight(10.0)
    result = validate_trajectory_kinematics(unreachable, 8.0, np.zeros(3))
    assert "outside_reachable_distance" in result.violations


def test_acceleration_reverse_and_turning_violations_are_named() -> None:
    spike = _straight(8.0)
    spike[0, 0] = 8.0
    assert "acceleration_above_max" in validate_trajectory_kinematics(
        spike, 8.0, np.zeros(3)
    ).violations
    reverse = _straight(2.0)
    reverse[2:, 0] -= 5.0
    assert "non_forward_motion" in validate_trajectory_kinematics(
        reverse, 2.0, np.zeros(3)
    ).violations
    sharp = _straight(4.0)
    sharp[:, 1] = np.asarray([0, 2, -2, 2, -2, 2, -2, 2])
    sharp[:, 2] = np.asarray([0, 1, -1, 1, -1, 1, -1, 1])
    result = validate_trajectory_kinematics(sharp, 4.0, np.zeros(3))
    assert {"yaw_rate_limit", "curvature_limit"} & set(result.violations)


def test_invalid_tensor_contract_raises_without_fallback() -> None:
    with pytest.raises(ModeContractError, match="shape"):
        validate_trajectory_kinematics(
            np.zeros((7, 3), dtype=np.float32), 0.0, np.zeros(3)
        )
    with pytest.raises(ModeContractError, match="finite"):
        validate_trajectory_kinematics(
            np.full((8, 3), np.nan, dtype=np.float32), 0.0, np.zeros(3)
        )
