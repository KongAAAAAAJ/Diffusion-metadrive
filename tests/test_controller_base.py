from __future__ import annotations

import numpy as np
import pytest

from models.controller import BaseController, LQRFollowerController, PIDTrajectoryController


class _FakeVehicle:
    def __init__(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0, speed_km_h: float = 0.0) -> None:
        self.position = np.asarray([x, y], dtype=np.float32)
        self.heading_theta = float(heading)
        self.speed_km_h = float(speed_km_h)
        self.LENGTH = 5.0


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


def test_lqr_follower_controller_records_lateral_debug_for_followers() -> None:
    controller = LQRFollowerController({"lqr_lat_q1": 2.0, "lqr_lat_q2": 1.0, "lqr_lat_r": 0.2})
    env = type(
        "Env",
        (),
        {
            "_agent_ids": ["agent0", "agent1"],
            "agents": {
                "agent0": _FakeVehicle(x=10.0, y=0.0, heading=0.0, speed_km_h=30.0),
                "agent1": _FakeVehicle(x=-2.0, y=1.0, heading=0.1, speed_km_h=28.0),
            },
        },
    )()
    trajectory = np.asarray([[float(i), 0.0, 0.0] for i in range(8)], dtype=np.float32)

    actions = controller.compute_actions(env, {"agent0": trajectory})
    debug = controller.get_last_debug()

    assert set(actions) == {"agent0", "agent1"}
    assert debug["agent0"]["mode"] == "leader_pid"
    follower_debug = debug["agent1"]
    assert follower_debug["mode"] == "follower_lqr"
    assert follower_debug["leader_id"] == "agent0"
    assert follower_debug["lat_error"] == pytest.approx(1.0)
    assert follower_debug["heading_error"] == pytest.approx(0.1)
    assert follower_debug["q"] == {"lat": 2.0, "heading": 1.0}
    assert follower_debug["r"] == 0.2
    assert len(follower_debug["K_lat"]) == 2
    assert -1.0 <= follower_debug["clipped_steering"] <= 1.0
