from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from envs.platoon_env import PlatoonEnv


class _ControlHarness:
    _trajectory_target_speed_mps = staticmethod(
        PlatoonEnv._trajectory_target_speed_mps
    )
    _longitudinal_lqr = PlatoonEnv._longitudinal_lqr
    _lateral_pd = PlatoonEnv._lateral_pd
    _solve_lqr_gain = PlatoonEnv._solve_lqr_gain
    trajectory_to_control = PlatoonEnv.trajectory_to_control
    _agent_speed_km_h = PlatoonEnv._agent_speed_km_h
    _agent_pose = PlatoonEnv._agent_pose
    _desired_center_spacing_m = PlatoonEnv._desired_center_spacing_m
    _vehicle_length_m = PlatoonEnv._vehicle_length_m
    _cfg = PlatoonEnv._cfg
    _cfg_float = PlatoonEnv._cfg_float

    def __init__(self, *, speed_kmh: float = 18.0, follower_x: float = -15.74):
        self._agent_ids = ["agent0", "agent1", "agent2"]
        self.config = {
            "vehicle_length_m": 5.74,
            "initial_speed_km_h": 25.0,
        }
        self.platoon_config = SimpleNamespace(
            vehicle_length_m=5.74,
            initial_speed_km_h=25.0,
        )
        self.agents = {
            "agent0": SimpleNamespace(
                position=np.asarray([0.0, 0.0], dtype=np.float32),
                heading_theta=0.0,
                speed_km_h=float(speed_kmh),
                FRONT_WHEELBASE=1.4,
                REAR_WHEELBASE=1.4,
                max_steering=60.0,
            ),
            "agent1": SimpleNamespace(
                position=np.asarray([follower_x, 0.0], dtype=np.float32),
                heading_theta=0.0,
                speed_km_h=float(speed_kmh),
                FRONT_WHEELBASE=1.4,
                REAR_WHEELBASE=1.4,
                max_steering=60.0,
            ),
            "agent2": SimpleNamespace(
                position=np.asarray([2.0 * follower_x, 0.0], dtype=np.float32),
                heading_theta=0.0,
                speed_km_h=float(speed_kmh),
                FRONT_WHEELBASE=1.4,
                REAR_WHEELBASE=1.4,
                max_steering=60.0,
            ),
        }


def _trajectory(speed_mps: float) -> np.ndarray:
    times = np.arange(1, 9, dtype=np.float32) * 0.5
    value = np.zeros((8, 3), dtype=np.float32)
    value[:, 0] = times * float(speed_mps)
    return value


def test_trajectory_target_speed_uses_first_second_arc_length() -> None:
    trajectory = _trajectory(8.0)
    assert PlatoonEnv._trajectory_target_speed_mps(trajectory) == pytest.approx(
        8.0
    )
    trajectory[:2, 1] = (1.5, 3.0)
    expected = 2.0 * np.hypot(4.0, 1.5)
    assert PlatoonEnv._trajectory_target_speed_mps(trajectory) == pytest.approx(
        expected
    )


def test_stop_and_speed_limit() -> None:
    assert PlatoonEnv._trajectory_target_speed_mps(_trajectory(0.0)) == 0.0
    assert PlatoonEnv._trajectory_target_speed_mps(
        _trajectory(100.0)
    ) == pytest.approx(100.0 / 3.6)


def test_leader_control_tracks_trajectory_speed() -> None:
    env = _ControlHarness(speed_kmh=18.0)
    stop = env.trajectory_to_control("agent0", _trajectory(0.0))
    cruise = env.trajectory_to_control("agent0", _trajectory(8.0))
    assert stop.shape == (2,)
    assert stop.dtype == np.float32
    assert stop[1] < cruise[1]
    assert stop[1] < 0.0 < cruise[1]


def test_follower_combines_gap_and_trajectory_speed() -> None:
    desired = _ControlHarness(speed_kmh=18.0, follower_x=-15.74)
    close = _ControlHarness(speed_kmh=18.0, follower_x=-9.0)
    fast_reference = desired.trajectory_to_control("agent1", _trajectory(8.0))
    slow_reference = desired.trajectory_to_control("agent1", _trajectory(2.0))
    close_reference = close.trajectory_to_control("agent1", _trajectory(8.0))
    assert slow_reference[1] < fast_reference[1]
    assert close_reference[1] < fast_reference[1]


def test_pure_pursuit_turn_sign_and_straight_zero() -> None:
    env = _ControlHarness(speed_kmh=18.0)
    straight = _trajectory(6.0)
    left = straight.copy()
    left[:, 1] = np.linspace(0.2, 4.0, 8, dtype=np.float32)
    left[:, 2] = np.linspace(0.02, 0.35, 8, dtype=np.float32)
    right = left.copy()
    right[:, 1:] *= -1.0
    assert env._lateral_pd("agent0", straight) == pytest.approx(0.0)
    assert env._lateral_pd("agent0", left) > 0.0
    assert env._lateral_pd("agent0", right) < 0.0


def test_curve_preview_caps_speed_by_lateral_acceleration() -> None:
    straight = _trajectory(20.0)
    curve = straight.copy()
    curve[:, 2] = np.linspace(0.15, 1.2, 8, dtype=np.float32)
    straight_speed = PlatoonEnv._trajectory_target_speed_mps(straight)
    curve_speed = PlatoonEnv._trajectory_target_speed_mps(curve)
    assert curve_speed < straight_speed
    distance = np.linalg.norm(
        np.diff(
            np.concatenate(
                (np.zeros((1, 2), dtype=np.float32), curve[:, :2]), axis=0
            ),
            axis=0,
        ),
        axis=1,
    )
    curvature = np.abs(
        np.arctan2(
            np.sin(np.diff(np.concatenate(([0.0], curve[:, 2])))),
            np.cos(np.diff(np.concatenate(([0.0], curve[:, 2])))),
        )
    ) / np.maximum(distance, 1.0e-3)
    assert curve_speed**2 * curvature.max() <= 6.0 + 1.0e-5


@pytest.mark.parametrize(
    "trajectory",
    [
        np.zeros((7, 3), dtype=np.float32),
        np.zeros((8, 3), dtype=np.int64),
        np.full((8, 3), np.nan, dtype=np.float32),
    ],
)
def test_invalid_trajectory_is_rejected(trajectory: np.ndarray) -> None:
    env = _ControlHarness()
    with pytest.raises(ValueError):
        env.trajectory_to_control("agent0", trajectory)
