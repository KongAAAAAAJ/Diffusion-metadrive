from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from models.platoon_planner.platoon_normal_planner import PlatoonNormalPlanner


class FakeLane:
    def __init__(self, lane_id: int, y: float, length: float = 120.0, width: float = 3.5):
        self.index = ("A", "B", lane_id)
        self.length = float(length)
        self.width = float(width)
        self.y = float(y)

    def local_coordinates(self, position):
        return float(position[0]), float(position[1] - self.y)

    def position(self, longitudinal: float, lateral: float):
        return np.asarray([float(longitudinal), self.y + float(lateral)], dtype=np.float32)

    def heading_theta_at(self, longitudinal: float):  # noqa: ARG002
        return 0.0


class FakeRoadNetwork:
    def __init__(self):
        self._lanes = {
            ("A", "B", 0): FakeLane(0, 3.5),
            ("A", "B", 1): FakeLane(1, 0.0),
            ("A", "B", 2): FakeLane(2, -3.5),
        }

    def get_lane(self, lane_index):
        return self._lanes[tuple(lane_index)]


class S8HardcodedTargetFakeLane(FakeLane):
    def __init__(self, from_node: str, to_node: str, lane_id: int, y: float):
        super().__init__(lane_id, y)
        self.index = (from_node, to_node, lane_id)


class S8HardcodedTargetFakeRoadNetwork:
    def __init__(self):
        self._lanes = {
            ("3C0_1_", "4G0_0_", 2): S8HardcodedTargetFakeLane("3C0_1_", "4G0_0_", 2, -3.5),
            ("3C0_1_", "4G1_0_", 0): S8HardcodedTargetFakeLane("3C0_1_", "4G1_0_", 0, -7.0),
            ("4G0_0_", "4G1_1_", 0): S8HardcodedTargetFakeLane("4G0_0_", "4G1_1_", 0, -3.5),
        }

    def get_lane(self, lane_index):
        return self._lanes[tuple(lane_index)]


def _vehicle(name: str, x: float, y: float, lane, speed_km_h: float = 18.0):
    return SimpleNamespace(
        name=name,
        position=np.asarray([x, y], dtype=np.float32),
        heading_theta=0.0,
        speed_km_h=float(speed_km_h),
        lane=lane,
        lane_index=lane.index,
    )


def _env(agent_lane_id: int = 1):
    road_network = FakeRoadNetwork()
    lane = road_network.get_lane(("A", "B", agent_lane_id))
    agents = {
        "agent0": _vehicle("agent0", 10.0, lane.y, lane),
    }
    return SimpleNamespace(
        agents=agents,
        engine=SimpleNamespace(current_map=SimpleNamespace(road_network=road_network)),
    )


def _env_s8_hardcoded_target():
    road_network = S8HardcodedTargetFakeRoadNetwork()
    return SimpleNamespace(
        engine=SimpleNamespace(current_map=SimpleNamespace(road_network=road_network)),
    )


def test_keep_returns_8x3_forward_trajectory():
    env = _env(agent_lane_id=1)
    planner = PlatoonNormalPlanner()

    result = planner.plan(
        env,
        {"agent0": {"action": 0, "target_point": np.asarray([25.0, 0.0], dtype=np.float32)}},
    )

    traj = result["agent0"]
    assert traj.shape == (8, 3)
    assert traj.dtype == np.float32
    assert traj[-1, 0] > traj[0, 0]
    assert np.allclose(traj[:, 1], 0.0, atol=0.6)
    assert np.max(np.abs(np.diff(traj[:, 2]))) < 0.2


def test_left_action_converges_toward_left_lane_center():
    env = _env(agent_lane_id=1)
    planner = PlatoonNormalPlanner()

    result = planner.plan(
        env,
        {"agent0": {"action": -1, "target_point": np.asarray([25.0, 0.0], dtype=np.float32)}},
    )

    traj = result["agent0"]
    assert traj[-1, 1] > 2.5


def test_right_action_converges_toward_right_lane_center():
    env = _env(agent_lane_id=1)
    planner = PlatoonNormalPlanner()

    result = planner.plan(
        env,
        {"agent0": {"action": 1, "target_point": np.asarray([25.0, 0.0], dtype=np.float32)}},
    )

    traj = result["agent0"]
    assert traj[-1, 1] < -2.5


def test_missing_adjacent_lane_falls_back_to_keep():
    env = _env(agent_lane_id=0)
    planner = PlatoonNormalPlanner()

    result = planner.plan(
        env,
        {"agent0": {"action": -1, "target_point": np.asarray([25.0, 0.0], dtype=np.float32)}},
    )

    traj = result["agent0"]
    assert np.allclose(traj[:, 1], 3.5, atol=0.6)


def test_resolve_target_lane_uses_s8_hardcoded_branch_for_3c0_right_lane():
    env = _env_s8_hardcoded_target()
    source_lane = env.engine.current_map.road_network.get_lane(("3C0_1_", "4G0_0_", 2))

    target_lane = PlatoonNormalPlanner._resolve_target_lane(env, source_lane, action=1)

    assert target_lane is not None
    assert tuple(target_lane.index) == ("3C0_1_", "4G1_0_", 0)


def test_target_point_changes_terminal_progress():
    env = _env(agent_lane_id=1)
    planner = PlatoonNormalPlanner()

    near = planner.plan(
        env,
        {"agent0": {"action": 0, "target_point": np.asarray([16.0, 0.0], dtype=np.float32)}},
    )["agent0"]
    far = planner.plan(
        env,
        {"agent0": {"action": 0, "target_point": np.asarray([40.0, 0.0], dtype=np.float32)}},
    )["agent0"]

    assert far[-1, 0] > near[-1, 0]


def test_candidate_failure_falls_back_to_keep_lane_trajectory():
    env = _env(agent_lane_id=1)
    planner = PlatoonNormalPlanner()
    planner._candidate_durations = lambda: ()  # type: ignore[method-assign]

    result = planner.plan(
        env,
        {"agent0": {"action": 1, "target_point": np.asarray([20.0, 0.0], dtype=np.float32)}},
    )
    debug = planner.get_last_debug()

    traj = result["agent0"]
    assert traj.shape == (8, 3)
    assert np.allclose(traj[:, 1], 0.0, atol=0.6)
    assert debug is not None
    assert debug["agent0"]["fallback_used"] is True


def test_debug_records_lattice_candidates_and_selected_trajectory():
    env = _env(agent_lane_id=1)
    planner = PlatoonNormalPlanner()

    result = planner.plan(
        env,
        {"agent0": {"action": 0, "target_point": np.asarray([25.0, 0.0], dtype=np.float32)}},
    )
    debug = planner.get_last_debug()

    assert debug is not None
    agent_debug = debug["agent0"]
    assert agent_debug["fallback_used"] is False
    assert agent_debug["candidate_count"] == len(agent_debug["candidates"])
    assert agent_debug["candidate_count"] > 1
    selected = [candidate for candidate in agent_debug["candidates"] if candidate["selected"]]
    assert len(selected) == 1
    assert np.allclose(np.asarray(selected[0]["trajectory_world"], dtype=np.float32), result["agent0"])
    assert all("score" in candidate for candidate in agent_debug["candidates"])


def test_score_candidate_adds_safety_distance_penalty_for_nearby_agent():
    env = _env(agent_lane_id=1)
    ego = env.agents["agent0"]
    lane = ego.lane
    env.agents["traffic"] = _vehicle("traffic", 24.0, 0.0, lane, speed_km_h=18.0)
    planner = PlatoonNormalPlanner(safety_distance_m=8.0, safety_weight=10.0, ttc_weight=0.0)
    candidate = np.asarray(
        [
            [10.0, 0.0, 0.0],
            [18.0, 0.0, 0.0],
            [24.0, 0.0, 0.0],
            [32.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    score_with_agent = planner._score_candidate(
        candidate,
        target_world=np.asarray([32.0, 0.0], dtype=np.float32),
        source_lane=lane,
        target_lane=lane,
        action=0,
        desired_end_d=0.0,
        env=env,
        vehicle=ego,
        duration=4.0,
    )
    env.agents.pop("traffic")
    score_without_agent = planner._score_candidate(
        candidate,
        target_world=np.asarray([32.0, 0.0], dtype=np.float32),
        source_lane=lane,
        target_lane=lane,
        action=0,
        desired_end_d=0.0,
        env=env,
        vehicle=ego,
        duration=4.0,
    )

    assert score_with_agent > score_without_agent + 9.0


def test_score_candidate_adds_ttc_penalty_for_slow_lead_vehicle():
    env = _env(agent_lane_id=1)
    ego = env.agents["agent0"]
    ego.speed_km_h = 36.0
    lane = ego.lane
    env.agents["slow_lead"] = _vehicle("slow_lead", 22.0, 0.0, lane, speed_km_h=0.0)
    planner = PlatoonNormalPlanner(safety_weight=0.0, ttc_threshold_s=3.0, ttc_weight=12.0)
    candidate = np.asarray(
        [
            [10.0, 0.0, 0.0],
            [18.0, 0.0, 0.0],
            [26.0, 0.0, 0.0],
            [34.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    slow_score = planner._score_candidate(
        candidate,
        target_world=np.asarray([34.0, 0.0], dtype=np.float32),
        source_lane=lane,
        target_lane=lane,
        action=0,
        desired_end_d=0.0,
        env=env,
        vehicle=ego,
        duration=4.0,
    )
    env.agents["slow_lead"].speed_km_h = 60.0
    fast_score = planner._score_candidate(
        candidate,
        target_world=np.asarray([34.0, 0.0], dtype=np.float32),
        source_lane=lane,
        target_lane=lane,
        action=0,
        desired_end_d=0.0,
        env=env,
        vehicle=ego,
        duration=4.0,
    )

    assert slow_score > fast_score + 3.0
