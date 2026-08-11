from __future__ import annotations

import importlib

import numpy as np
import pytest

from models.controller import BaseController, LQRFollowerController, PIDTrajectoryController

lqr_module = importlib.import_module("models.controller.LQRFollowerController")


class _FakeLane:
    """Minimal lane stub: s = x, t = y."""
    def __init__(self, index=("A", "B", 0), width: float = 3.5) -> None:
        self.index = index
        self.width = float(width)

    def local_coordinates(self, pos):
        return float(pos[0]), float(pos[1])


class _FakeVehicle:
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        heading: float = 0.0,
        speed_km_h: float = 0.0,
        lane: "_FakeLane | None" = None,
    ) -> None:
        self.position = np.asarray([x, y], dtype=np.float32)
        self.heading_theta = float(heading)
        self.speed_km_h = float(speed_km_h)
        self.heading = np.asarray(
            [np.cos(self.heading_theta), np.sin(self.heading_theta)],
            dtype=np.float32,
        )
        self.velocity = self.heading * (self.speed_km_h / 3.6)
        self.LENGTH = 5.0
        self.lane = lane


class _FakeEnv:
    def __init__(self) -> None:
        self.agents = {"agent0": _FakeVehicle()}


def test_base_controller_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseController()


def test_pid_trajectory_controller_implements_base_controller() -> None:
    controller = PIDTrajectoryController({"pid_dt": 0.2})

    assert isinstance(controller, BaseController)
    assert controller.config["pid_dt"] == 0.2


def test_pid_default_dt_matches_simulator_decision_period() -> None:
    controller = PIDTrajectoryController(
        {"physics_world_step_size": 0.02, "decision_repeat": 5}
    )

    assert controller.dt == pytest.approx(0.1)


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan")])
def test_pid_rejects_invalid_control_period(value: float) -> None:
    with pytest.raises(ValueError, match="pid_dt"):
        PIDTrajectoryController({"pid_dt": value})


def test_pid_compute_actions_returns_low_level_action_dict() -> None:
    controller = PIDTrajectoryController({"pid_dt": 0.5})
    env = _FakeEnv()
    trajectory = np.asarray([[float(i), 0.0, 0.0] for i in range(8)], dtype=np.float32)

    actions = controller.compute_actions(env, {"agent0": trajectory})

    assert set(actions) == {"agent0"}
    assert actions["agent0"].shape == (2,)
    assert actions["agent0"].dtype == np.float32


def test_pid_compute_actions_skips_missing_agent_and_preserves_empty_trajectory_behavior() -> None:
    controller = PIDTrajectoryController()
    env = _FakeEnv()

    actions = controller.compute_actions(
        env,
        {
            "agent0": np.zeros((0, 3), dtype=np.float32),
            "agent_missing": np.zeros((8, 3), dtype=np.float32),
        },
    )

    assert set(actions) == {"agent0"}
    np.testing.assert_array_equal(actions["agent0"], np.zeros((2,), dtype=np.float32))


def test_pid_does_not_double_count_fixed_time_heading_on_top_of_preview() -> None:
    trajectory = np.asarray(
        [
            [4.0, -0.1, 0.12],
            [8.0, -0.8, -0.08],
            [12.0, -2.0, -0.18],
            [16.0, -3.4, -0.24],
            [20.0, -4.8, -0.28],
            [24.0, -6.0, -0.30],
            [28.0, -7.0, -0.30],
            [32.0, -7.8, -0.30],
        ],
        dtype=np.float32,
    )
    controller = PIDTrajectoryController({"pid_dt": 0.1})
    vehicle = _FakeVehicle(speed_km_h=30.0)

    action, debug = controller._single_control_with_debug(
        "agent0", vehicle, trajectory, None
    )

    assert np.isfinite(action).all()
    assert debug["tracking_heading_error_rad"] == pytest.approx(0.12)
    assert debug["heading_correction"] == 0.0


def test_pid_cross_track_feedback_opposes_lateral_path_error() -> None:
    trajectory = np.asarray(
        [[4.0 * (index + 1), 0.0, 0.0] for index in range(8)],
        dtype=np.float32,
    )
    controller = PIDTrajectoryController({"pid_dt": 0.1})
    vehicle = _FakeVehicle(speed_km_h=30.0)

    action, debug = controller._single_control_with_debug(
        "agent0",
        vehicle,
        trajectory,
        None,
        cross_track_error_m=-0.5,
    )

    assert float(action[0]) > 0.0
    assert debug["cross_track_correction"] == pytest.approx(0.2)


def test_pid_uses_s9_specific_cross_track_gain() -> None:
    trajectory = np.asarray(
        [[4.0 * (index + 1), 0.0, 0.0] for index in range(8)],
        dtype=np.float32,
    )
    controller = PIDTrajectoryController(
        {
            "pid_dt": 0.1,
            "pid_cross_track_kp": 0.4,
            "s9_pid_cross_track_kp": 0.1,
        }
    )
    env = _FakeEnv()
    env.config = {"scenario_id": "S9_narrow_channel_negotiation"}

    controller.compute_actions(
        env,
        {"agent0": trajectory},
        lateral_tracking_errors_m={"agent0": 0.5},
    )
    debug = controller.get_last_debug()["agent0"]

    assert debug["cross_track_kp"] == pytest.approx(0.1)
    assert debug["cross_track_correction"] == pytest.approx(-0.05)


@pytest.mark.parametrize("key", ["pid_cross_track_kp", "s9_pid_cross_track_kp"])
def test_pid_rejects_invalid_cross_track_gain(key: str) -> None:
    with pytest.raises(ValueError, match="cross-track gains"):
        PIDTrajectoryController({key: -0.1})


def test_pid_preview_does_not_skip_initial_opposite_curve_direction() -> None:
    trajectory = np.asarray(
        [
            [4.6055, 0.2317, 0.04176],
            [8.4292, -0.0534, -0.28153],
            [11.4874, -1.1772, -0.40417],
            [13.7265, -2.0773, -0.32140],
            [15.1328, -2.4798, -0.20691],
            [15.7167, -2.5901, -0.15362],
            [16.5216, -2.6833, -0.05916],
            [18.5729, -2.6734, 0.04691],
        ],
        dtype=np.float32,
    )
    controller = PIDTrajectoryController({"pid_dt": 0.1})
    vehicle = _FakeVehicle(speed_km_h=36.0)

    action, debug = controller._single_control_with_debug(
        "agent0", vehicle, trajectory, None
    )

    assert debug["preview_direction_guarded"] is True
    assert debug["preview_query_m"] == pytest.approx(
        float(np.linalg.norm(trajectory[0, :2]))
    )
    assert debug["first_path_heading_rad"] > 0.0
    assert debug["preview_heading_rad"] > 0.0
    assert float(action[0]) > 0.0


def test_pid_debug_separates_world_reference_tangent_and_actual_yaw_response() -> None:
    trajectory = np.asarray(
        [[4.0 * (index + 1), 0.2 * (index + 1), 0.05] for index in range(8)],
        dtype=np.float32,
    )
    controller = PIDTrajectoryController({"pid_dt": 0.1})
    vehicle = _FakeVehicle(heading=0.4, speed_km_h=20.0)

    _, first = controller._single_control_with_debug(
        "agent0", vehicle, trajectory, None
    )
    vehicle.heading_theta = 0.42
    _, second = controller._single_control_with_debug(
        "agent0", vehicle, trajectory, None
    )

    assert first["actual_yaw_rate_rad_s"] == pytest.approx(0.0)
    assert second["actual_yaw_rate_rad_s"] == pytest.approx(0.2)
    assert second["preview_reference_heading_world_rad"] == pytest.approx(
        0.42 + second["preview_heading_rad"]
    )
    assert second["first_reference_heading_world_rad"] == pytest.approx(0.47)


def test_lqr_follower_controller_records_lateral_debug_for_followers() -> None:
    shared_lane = _FakeLane()
    controller = LQRFollowerController()
    env = type(
        "Env",
        (),
        {
            "_agent_ids": ["agent0", "agent1"],
            "agents": {
                "agent0": _FakeVehicle(x=10.0, y=0.0, heading=0.0, speed_km_h=30.0, lane=shared_lane),
                "agent1": _FakeVehicle(x=-2.0, y=1.0, heading=0.1, speed_km_h=28.0, lane=shared_lane),
            },
        },
    )()

    trajectories = {
        agent_id: np.asarray(
            [
                [vehicle.position[0] + 4.0 * (index + 1), vehicle.position[1], 0.0]
                for index in range(8)
            ],
            dtype=np.float32,
        )
        for agent_id, vehicle in env.agents.items()
    }
    actions = controller.compute_actions(env, trajectories)
    debug = controller.get_last_debug()

    assert set(actions) == {"agent0", "agent1"}
    assert debug["agent0"]["mode"] == "trajectory_no_front"
    # agent1 finds agent0 (platoon member) ahead
    follower_debug = debug["agent1"]
    assert follower_debug["mode"] == "follower_platoon"
    assert follower_debug["leader_id"] == "agent0"
    assert follower_debug["desired_gap_m"] == pytest.approx(10.0)
    assert abs(follower_debug["gap_feedback_mps2"]) <= 1.0
    assert follower_debug["longitudinal_reference_source"] == "online_trajectory"
    assert -1.0 <= follower_debug["clipped_steering"] <= 1.0


def test_lqr_follower_controller_uses_bounded_gap_cascade_debug() -> None:
    shared_lane = _FakeLane()
    controller = LQRFollowerController(
        {
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
            "lqr_max_accel_mps2": 4.0,
        }
    )
    env = type(
        "Env",
        (),
        {
            "_agent_ids": ["agent0", "agent1"],
            "agents": {
                "agent0": _FakeVehicle(x=10.0, y=0.0, heading=0.0, speed_km_h=30.0, lane=shared_lane),
                "agent1": _FakeVehicle(x=-2.0, y=0.0, heading=0.0, speed_km_h=28.0, lane=shared_lane),
            },
        },
    )()

    def trajectories():
        return {
            agent_id: np.asarray(
                [
                    [vehicle.position[0] + 4.0 * (index + 1), vehicle.position[1], 0.0]
                    for index in range(8)
                ],
                dtype=np.float32,
            )
            for agent_id, vehicle in env.agents.items()
        }

    controller.compute_actions(env, trajectories())
    env.agents["agent1"].speed_km_h = 31.6
    actions = controller.compute_actions(env, trajectories())
    follower_debug = controller.get_last_debug()["agent1"]

    assert set(actions) == {"agent0", "agent1"}
    assert abs(follower_debug["gap_feedback_mps2"]) <= 1.0
    assert np.isfinite(follower_debug["desired_acceleration_mps2"])
    assert -2.6 <= follower_debug["desired_acceleration_mps2"] <= 0.4
    assert follower_debug["longitudinal_reference_source"] == "online_trajectory"
    assert -1.0 <= follower_debug["clipped_throttle"] <= 1.0


def test_lqr_follower_controller_ignores_config_for_physical_lqr_constants() -> None:
    controller = LQRFollowerController(
        {
            "lqr_desired_gap_m": 99.0,
            "lqr_max_accel_mps2": 88.0,
            "lqr_min_accel_mps2": -99.0,
            "lqr_wheelbase_m": 77.0,
            "physics_world_step_size": 0.5,
            "decision_repeat": 100,
        }
    )

    assert controller.desired_gap_m == pytest.approx(10.0)
    assert controller.max_accel_mps2 == pytest.approx(1.0)
    assert controller.min_accel_mps2 == pytest.approx(-2.0)
    assert controller.wheelbase_m == pytest.approx(2.8)
    assert controller.dt == pytest.approx(0.1)


def test_lqr_follower_controller_maps_positive_and_negative_accel_asymmetrically(monkeypatch) -> None:
    controller = LQRFollowerController()
    monkeypatch.setattr(
        lqr_module,
        "_solve_discrete_lqr",
        lambda *args, **kwargs: np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float64),
    )

    positive_throttle, positive_debug = controller._lon_lqr(
        gap_error=-0.5,
        speed_diff=0.0,
        ego_id="agent1",
        ego_spd_ms=10.0,
    )
    negative_throttle, negative_debug = controller._lon_lqr(
        gap_error=0.5,
        speed_diff=0.0,
        ego_id="agent1",
        ego_spd_ms=10.0,
    )

    assert positive_debug["desired_accel_mps2"] == pytest.approx(0.5)
    assert positive_throttle == pytest.approx(0.5 / 1.0)
    assert negative_debug["desired_accel_mps2"] == pytest.approx(-0.5)
    assert negative_throttle == pytest.approx(-0.5 / 2.0)
