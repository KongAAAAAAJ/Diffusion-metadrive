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
    values[..., 0] = (
        speed * 0.5 * np.arange(1, 9, dtype=np.float32)[None, None, :]
    )
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
        assert validate_trajectory_kinematics(
            trajectory, 8.0, np.zeros(3)
        ).valid


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
        KinematicTrajectoryOptimizerConfig().max_accel_mps2
        < EXECUTABLE_MAX_ACCEL_MPS2
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
