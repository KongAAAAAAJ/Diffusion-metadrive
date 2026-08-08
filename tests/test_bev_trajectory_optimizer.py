from __future__ import annotations

import numpy as np
import pytest

from models.bev_planner.mode_contract import (
    ModeIndex,
    validate_trajectory_kinematics,
)
from models.bev_planner.trajectory_optimizer import (
    KinematicTrajectoryOptimizer,
    KinematicTrajectoryOptimizerConfig,
    TrajectoryOptimizationError,
)
from models.controller.longitudinal_reference import EXECUTABLE_MAX_ACCEL_MPS2


def _coarse(speed: float = 8.0) -> np.ndarray:
    values = np.zeros((3, 10, 8, 3), dtype=np.float32)
    values[..., 0] = speed * 0.5 * np.arange(1, 9, dtype=np.float32)[None, None, :]
    # Contract-valid stop: average segment speeds 6, 2, 0, ... from 8 m/s.
    values[:, int(ModeIndex.STOP), :, 0] = np.asarray(
        [3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0], dtype=np.float32
    )
    return values


def test_optimizer_projects_raw_groups_without_changing_modes() -> None:
    optimizer = KinematicTrajectoryOptimizer()
    coarse = _coarse()
    raw = np.broadcast_to(coarse[:, 0], (4, 3, 8, 3)).copy()
    raw[..., 0] *= -2.0
    raw[..., 1] = np.linspace(-20.0, 20.0, 8, dtype=np.float32)
    raw[..., 2] = 2.5
    original = raw.copy()
    modes = np.zeros((4, 3), dtype=np.int64)

    result = optimizer.optimize(raw, coarse, np.full(3, 8.0), modes)

    assert np.array_equal(raw, original)
    assert np.array_equal(result.raw_trajectories, original)
    assert np.array_equal(result.selected_modes, modes)
    assert not result.raw_valid.any()
    assert result.optimized_valid.all()
    assert result.optimized_trajectories.shape == raw.shape
    for trajectory in result.optimized_trajectories.reshape(-1, 8, 3):
        assert validate_trajectory_kinematics(trajectory, 8.0, np.zeros(3)).valid


def test_stop_mode_is_stationary_after_reachable_braking() -> None:
    optimizer = KinematicTrajectoryOptimizer()
    coarse = _coarse()
    raw = coarse[:, int(ModeIndex.STOP)].copy()
    raw[:, :, 0] += 20.0
    modes = np.full(3, int(ModeIndex.STOP), dtype=np.int64)

    result = optimizer.optimize(raw, coarse, np.full(3, 8.0), modes)

    optimized = result.optimized_trajectories
    assert np.allclose(optimized[:, -1, :2], optimized[:, -2, :2])
    for trajectory in optimized:
        audit = validate_trajectory_kinematics(trajectory, 8.0, np.zeros(3))
        assert audit.valid
        assert audit.speed_mps[-1] == pytest.approx(0.0)


def test_optimizer_rejects_contract_mismatch_without_fallback() -> None:
    optimizer = KinematicTrajectoryOptimizer()
    with pytest.raises(TrajectoryOptimizationError, match="selected modes"):
        optimizer.optimize(
            np.zeros((3, 8, 3), dtype=np.float32),
            _coarse(),
            np.zeros(3, dtype=np.float32),
            np.zeros((1, 3), dtype=np.int64),
        )


def test_execution_mode_mask_guarantees_every_retained_anchor_projects() -> None:
    optimizer = KinematicTrajectoryOptimizer()
    coarse = _coarse(speed=8.0)
    hard_mask = np.ones((3, 10), dtype=np.bool_)
    execution_mask = optimizer.execution_mode_valid_mask(
        coarse,
        np.full(3, 8.0, dtype=np.float32),
        hard_mask,
    )

    assert execution_mask.shape == (3, 10)
    assert execution_mask.dtype == np.bool_
    assert not execution_mask.flags.writeable
    assert execution_mask[:, int(ModeIndex.STOP)].all()
    for role in range(3):
        for mode in np.flatnonzero(execution_mask[role]):
            optimizer._project_one(
                coarse[role, mode],
                coarse[role, mode],
                8.0,
                int(mode),
            )


def test_optimizer_uses_measured_drive_authority_and_strict_pose_alignment() -> None:
    optimizer = KinematicTrajectoryOptimizer()
    coarse = _coarse(speed=6.0)
    raw = coarse[:, 0].copy()
    # The generic hard mask permits this initial path/heading disagreement,
    # but it is not stable for the preview controller.
    raw[:, 0, 1] = -1.0
    modes = np.zeros(3, dtype=np.int64)

    result = optimizer.optimize(raw, coarse, np.full(3, 5.0), modes)

    assert KinematicTrajectoryOptimizerConfig().max_accel_mps2 == pytest.approx(1.0)
    assert (
        KinematicTrajectoryOptimizerConfig().max_accel_mps2 < EXECUTABLE_MAX_ACCEL_MPS2
    )
    assert np.all(result.retained_raw_fraction < 1.0)
    for trajectory in result.optimized_trajectories:
        audit = validate_trajectory_kinematics(
            trajectory,
            5.0,
            np.zeros(3),
            optimizer._audit_config,
        )
        assert audit.valid
        assert audit.acceleration_mps2.max() <= EXECUTABLE_MAX_ACCEL_MPS2 + 1e-6


def test_optimizer_repairs_heading_only_anchor_without_changing_xy_or_mode() -> None:
    optimizer = KinematicTrajectoryOptimizer()
    coarse = _coarse(speed=4.0)
    x = np.arange(1, 9, dtype=np.float32) * 2.0
    y = 0.015 * x**2
    curved = np.column_stack((x, y, np.zeros(8, dtype=np.float32)))
    coarse[:, int(ModeIndex.RIGHT_LOW)] = curved
    raw = coarse[:, int(ModeIndex.RIGHT_LOW)].copy()
    modes = np.full(3, int(ModeIndex.RIGHT_LOW), dtype=np.int64)

    generic = validate_trajectory_kinematics(curved, 4.0, np.zeros(3))
    strict = validate_trajectory_kinematics(
        curved, 4.0, np.zeros(3), optimizer._audit_config
    )
    assert generic.valid
    assert strict.violations == ("heading_alignment",)

    result = optimizer.optimize(raw, coarse, np.full(3, 4.0), modes)

    assert np.array_equal(result.selected_modes, modes)
    assert np.allclose(result.optimized_trajectories[..., :2], raw[..., :2])
    for trajectory in result.optimized_trajectories:
        assert validate_trajectory_kinematics(
            trajectory, 4.0, np.zeros(3), optimizer._audit_config
        ).valid


def test_brake_recovery_profile_is_actuator_lag_and_jerk_aware() -> None:
    """Regression for the B/S5 seed17 state2 rear-role blocker."""

    optimizer = KinematicTrajectoryOptimizer()
    coarse = _coarse(speed=5.1506)
    blocker_arc = np.asarray(
        [
            2.57531645,
            3.19063309,
            4.04758842,
            5.15290489,
            6.50320827,
            8.09852455,
            9.93774851,
            12.02069504,
        ],
        dtype=np.float32,
    )
    coarse[:, 0, :, 0] = blocker_arc
    raw = np.broadcast_to(coarse[:, 0], (1, 3, 8, 3)).copy()
    modes = np.zeros((1, 3), dtype=np.int64)

    result = optimizer.optimize(
        raw,
        coarse,
        np.full(3, 5.1506, dtype=np.float32),
        modes,
    )

    assert not np.array_equal(result.optimized_trajectories, raw)
    assert np.all(result.profile_regularization >= 0.3)
    assert np.all(
        result.predicted_max_positive_jerk_mps3
        <= optimizer.config.max_drive_jerk_mps3 + 1.0e-6
    )
    assert np.all(
        result.predicted_command_acceleration_max_mps2
        <= optimizer.config.max_accel_mps2 + 1.0e-6
    )
    assert np.all(
        result.predicted_command_acceleration_min_mps2
        >= optimizer.config.min_accel_mps2 - 1.0e-6
    )
    assert np.allclose(result.terminal_arc_error_m, 0.0, atol=1.0e-6)
    assert np.all(result.brake_to_drive_transition_s >= 2.0)
    assert np.allclose(
        result.optimized_trajectories[..., -1, 0],
        blocker_arc[-1],
        atol=1.0e-5,
    )


def test_constant_speed_profile_is_unchanged_and_deterministic() -> None:
    optimizer = KinematicTrajectoryOptimizer()
    coarse = _coarse(speed=4.0)
    raw = np.broadcast_to(coarse[:, 0], (2, 3, 8, 3)).copy()
    modes = np.zeros((2, 3), dtype=np.int64)

    first = optimizer.optimize(raw, coarse, np.full(3, 4.0), modes)
    second = optimizer.optimize(raw, coarse, np.full(3, 4.0), modes)

    assert np.array_equal(first.optimized_trajectories, second.optimized_trajectories)
    assert np.allclose(first.optimized_trajectories, raw, atol=1.0e-6)
    assert np.allclose(first.predicted_max_positive_jerk_mps3, 0.0, atol=1.0e-7)
    assert first.config_sha256 == second.config_sha256


def test_actuator_profile_config_rejects_invalid_contract() -> None:
    with pytest.raises(TrajectoryOptimizationError, match="integer multiple"):
        KinematicTrajectoryOptimizerConfig(internal_dt_s=0.3)
    with pytest.raises(TrajectoryOptimizationError, match="non-negative"):
        KinematicTrajectoryOptimizerConfig(drive_time_constant_s=-0.1)
    with pytest.raises(TrajectoryOptimizationError, match="increasing"):
        KinematicTrajectoryOptimizerConfig(profile_jerk_regularizations=(0.3, 0.1))
