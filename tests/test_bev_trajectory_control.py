from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from envs.platoon_env import PlatoonEnv
from models.controller.longitudinal_reference import (
    trajectory_to_longitudinal_reference,
)


class _ControlHarness:
    _trajectory_target_speed_mps = staticmethod(
        PlatoonEnv._trajectory_target_speed_mps
    )
    _longitudinal_lqr = PlatoonEnv._longitudinal_lqr
    _lateral_preview_pid = PlatoonEnv._lateral_preview_pid
    _solve_lqr_gain = PlatoonEnv._solve_lqr_gain
    trajectory_to_control = PlatoonEnv.trajectory_to_control
    trajectory_reference_to_control = PlatoonEnv.trajectory_reference_to_control
    trajectory_formation_constraint_enabled = (
        PlatoonEnv.trajectory_formation_constraint_enabled
    )
    _agent_speed_km_h = PlatoonEnv._agent_speed_km_h
    _agent_longitudinal_speed_mps = (
        PlatoonEnv._agent_longitudinal_speed_mps
    )
    _agent_pose = PlatoonEnv._agent_pose
    _desired_center_spacing_m = PlatoonEnv._desired_center_spacing_m
    _vehicle_length_m = PlatoonEnv._vehicle_length_m
    _cfg = PlatoonEnv._cfg
    _cfg_float = PlatoonEnv._cfg_float
    _cfg_int = PlatoonEnv._cfg_int

    def __init__(self, *, speed_kmh: float = 18.0, follower_x: float = -15.74):
        self._agent_ids = ["agent0", "agent1", "agent2"]
        self.config = {
            "vehicle_length_m": 5.74,
            "initial_speed_km_h": 25.0,
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
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
                velocity=np.asarray(
                    [float(speed_kmh) / 3.6, 0.0], dtype=np.float32
                ),
                FRONT_WHEELBASE=1.4,
                REAR_WHEELBASE=1.4,
                max_steering=60.0,
            ),
            "agent1": SimpleNamespace(
                position=np.asarray([follower_x, 0.0], dtype=np.float32),
                heading_theta=0.0,
                speed_km_h=float(speed_kmh),
                velocity=np.asarray(
                    [float(speed_kmh) / 3.6, 0.0], dtype=np.float32
                ),
                FRONT_WHEELBASE=1.4,
                REAR_WHEELBASE=1.4,
                max_steering=60.0,
            ),
            "agent2": SimpleNamespace(
                position=np.asarray([2.0 * follower_x, 0.0], dtype=np.float32),
                heading_theta=0.0,
                speed_km_h=float(speed_kmh),
                velocity=np.asarray(
                    [float(speed_kmh) / 3.6, 0.0], dtype=np.float32
                ),
                FRONT_WHEELBASE=1.4,
                REAR_WHEELBASE=1.4,
                max_steering=60.0,
            ),
        }
        self._lateral_preview_pid_state = {}
        self._last_longitudinal_control_debug = {}


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
    fast_reference = desired.trajectory_to_control("agent1", _trajectory(5.0))
    slow_reference = desired.trajectory_to_control("agent1", _trajectory(2.0))
    close_reference = close.trajectory_to_control("agent1", _trajectory(5.0))
    assert slow_reference[1] < fast_reference[1]
    assert close_reference[1] < fast_reference[1]


def test_independent_follower_disables_gap_feedback() -> None:
    locked = _ControlHarness(speed_kmh=18.0, follower_x=-9.0)
    independent = _ControlHarness(speed_kmh=18.0, follower_x=-9.0)
    trajectory = _trajectory(5.0)
    reference = trajectory_to_longitudinal_reference(
        trajectory,
        5.0,
        source="test_independent",
    )
    locked_control = locked.trajectory_reference_to_control(
        "agent1", trajectory, reference
    )
    independent_control = independent.trajectory_reference_to_control(
        "agent1",
        trajectory,
        reference,
        formation_constraint_enabled=False,
    )
    assert independent_control[1] > locked_control[1]
    assert independent._last_longitudinal_control_debug["agent1"][
        "gap_feedback_mps2"
    ] == pytest.approx(0.0)


def test_preview_pid_turn_sign_and_straight_zero() -> None:
    straight = _trajectory(6.0)
    left = straight.copy()
    left[:, 1] = np.linspace(0.2, 4.0, 8, dtype=np.float32)
    left[:, 2] = np.linspace(0.02, 0.35, 8, dtype=np.float32)
    right = left.copy()
    right[:, 1:] *= -1.0
    assert _ControlHarness()._lateral_preview_pid(
        "agent0", straight
    ) == pytest.approx(0.0)
    assert _ControlHarness()._lateral_preview_pid("agent0", left) > 0.0
    assert _ControlHarness()._lateral_preview_pid("agent0", right) < 0.0


def test_preview_pid_uses_future_curvature() -> None:
    near_straight = _trajectory(8.0)
    future_curve = near_straight.copy()
    future_curve[1:, 1] = np.linspace(0.8, 5.0, 7, dtype=np.float32)
    future_curve[1:, 2] = np.linspace(0.08, 0.5, 7, dtype=np.float32)
    straight_command = _ControlHarness(
        speed_kmh=36.0
    )._lateral_preview_pid("agent0", near_straight)
    curve_command = _ControlHarness(
        speed_kmh=36.0
    )._lateral_preview_pid("agent0", future_curve)
    assert curve_command > straight_command


def test_preview_pid_integral_is_bounded_and_resettable() -> None:
    env = _ControlHarness()
    left = _trajectory(6.0)
    left[:, 1] = 3.0
    left[:, 2] = 0.3
    for _ in range(100):
        command = env._lateral_preview_pid("agent0", left)
        assert -1.0 <= command <= 1.0
    integral, _, initialized = env._lateral_preview_pid_state["agent0"]
    assert initialized
    assert abs(integral) <= 1.0
    env._lateral_preview_pid_state = {}
    assert env._lateral_preview_pid_state == {}


def test_terminal_target_behind_vehicle_holds_heading_without_full_turn() -> None:
    env = _ControlHarness(speed_kmh=3.6)
    terminal = np.tile(
        np.asarray([-0.1, 0.2, -0.2], dtype=np.float32), (8, 1)
    )
    command = env._lateral_preview_pid("agent0", terminal)
    assert command < 0.0
    assert abs(command) < 1.0


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


def test_three_agent_control_mapping_does_not_recompute_scenario_summary() -> None:
    class AttrConfig(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

    class Orchestrator:
        summary = SimpleNamespace(scenario_realized=True)

        def get_episode_summary(self):
            raise AssertionError("control mapping must not advance functional evidence")

    env = _ControlHarness()
    env.config = AttrConfig(env.config)
    env.config["scenario_id"] = "S5_hard_brake_lead"
    env._scenario_orchestrator = Orchestrator()

    controls = {
        agent_id: env.trajectory_to_control(agent_id, _trajectory(8.0))
        for agent_id in env._agent_ids
    }

    assert set(controls) == set(env._agent_ids)
    assert all(value.shape == (2,) for value in controls.values())
