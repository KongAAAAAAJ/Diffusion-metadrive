from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import models.decisioner.rule_decisioner as rule_maker_module
from models.decisioner.risk import RiskDetector, SimpleRuleRiskDetector
from models.decisioner.rule_decisioner_helper import (
    save_candidate_debug_plot,
    save_lane_pair_debug_plot,
    save_s8_route_lanes_debug_plot,
)
from models.decisioner.rule_decisioner import MultiAgentRuleMaker, make_rule_maker


def test_rule_risk_detector_has_single_package_entrypoint():
    assert RiskDetector is not None
    assert SimpleRuleRiskDetector is not None
    assert importlib.util.find_spec("models.decisioner.risk_detect") is None


def test_rule_maker_factory_uses_multi_agent_type_without_legacy_alias():
    assert not hasattr(rule_maker_module, "KeepLaneFallbackRuleMaker")
    assert isinstance(make_rule_maker({}), MultiAgentRuleMaker)
    assert isinstance(make_rule_maker({"rule_maker_type": "multi_agent"}), MultiAgentRuleMaker)
    try:
        make_rule_maker({"rule_maker_type": "keep_lane_fallback"})
    except ValueError as exc:
        assert "keep_lane_fallback" in str(exc)
    else:
        raise AssertionError("legacy keep_lane_fallback rule_maker_type should be rejected")


class FakeLane:
    def __init__(self, lane_id: int, y: float, length: float = 200.0, width: float = 3.5):
        self.index = ("A", "B", lane_id)
        self.length = float(length)
        self.width = float(width)
        self.y = float(y)

    def local_coordinates(self, position):
        return float(position[0]), float(position[1] - self.y)

    def position(self, longitudinal: float, lateral: float):
        return np.asarray([float(longitudinal), self.y + float(lateral)], dtype=np.float32)


class FakeRoadNetwork:
    def __init__(self):
        self._lanes = {
            ("A", "B", 0): FakeLane(0, 3.5),
            ("A", "B", 1): FakeLane(1, 0.0),
            ("A", "B", 2): FakeLane(2, -3.5),
        }
        self.graph = {"A": {"B": [self._lanes[("A", "B", 0)], self._lanes[("A", "B", 1)], self._lanes[("A", "B", 2)]]}}

    def get_lane(self, lane_index):
        return self._lanes[tuple(lane_index)]


def _vehicle(name: str, x: float, y: float, lane_id: int, speed_km_h: float = 20.0):
    return SimpleNamespace(
        name=name,
        position=np.asarray([x, y], dtype=np.float32),
        heading_theta=0.0,
        lane=FakeLane(lane_id, y),
        speed_km_h=float(speed_km_h),
        LENGTH=4.5,
        WIDTH=1.8,
    )


def _env(agents, traffic):
    return SimpleNamespace(
        agents=agents,
        engine=SimpleNamespace(
            current_map=SimpleNamespace(road_network=FakeRoadNetwork()),
            traffic_manager=SimpleNamespace(_traffic_vehicles=list(traffic)),
        ),
    )


class ConnectedFakeLane(FakeLane):
    def __init__(self, lane_id: int, y: float, x_offset: float, from_node: str, to_node: str, length: float = 30.0, width: float = 3.5):
        super().__init__(lane_id=lane_id, y=y, length=length, width=width)
        self.index = (from_node, to_node, lane_id)
        self.x_offset = float(x_offset)

    def local_coordinates(self, position):
        return float(position[0] - self.x_offset), float(position[1] - self.y)

    def position(self, longitudinal: float, lateral: float):
        return np.asarray([self.x_offset + float(longitudinal), self.y + float(lateral)], dtype=np.float32)


class ConnectedFakeRoadNetwork:
    def __init__(self):
        self.graph = {
            "A": {
                "B": [ConnectedFakeLane(1, 0.0, 0.0, "A", "B", length=30.0)],
            },
            "B": {
                "C": [ConnectedFakeLane(1, 0.0, 30.0, "B", "C", length=30.0)],
            },
        }

    def get_lane(self, lane_index):
        return self.graph[lane_index[0]][lane_index[1]][lane_index[2] if len(self.graph[lane_index[0]][lane_index[1]]) > 1 else 0]


class TwoLaneConnectedFakeRoadNetwork:
    def __init__(self):
        self.graph = {
            "A": {
                "B": [
                    ConnectedFakeLane(0, 3.5, 0.0, "A", "B", length=30.0),
                    ConnectedFakeLane(1, 0.0, 0.0, "A", "B", length=30.0),
                    ConnectedFakeLane(2, -3.5, 0.0, "A", "B", length=30.0),
                ],
            },
            "B": {
                "C": [
                    ConnectedFakeLane(0, 3.5, 30.0, "B", "C", length=30.0),
                    ConnectedFakeLane(1, 0.0, 30.0, "B", "C", length=30.0),
                    ConnectedFakeLane(2, -5.0, 30.0, "B", "C", length=30.0),
                ],
            },
        }

    def get_lane(self, lane_index):
        return self.graph[lane_index[0]][lane_index[1]][lane_index[2]]


class S8ExitFakeRoadNetwork:
    def __init__(self):
        self.graph = {
            "A": {
                "B": [
                    ConnectedFakeLane(0, 3.5, 0.0, "A", "B", length=30.0),
                    ConnectedFakeLane(1, 0.0, 0.0, "A", "B", length=30.0),
                    ConnectedFakeLane(2, -3.5, 0.0, "A", "B", length=30.0),
                ],
            },
            "B": {
                "C": [
                    ConnectedFakeLane(0, 3.5, 30.0, "B", "C", length=30.0),
                    ConnectedFakeLane(1, 0.0, 30.0, "B", "C", length=30.0),
                    ConnectedFakeLane(2, -3.5, 30.0, "B", "C", length=30.0),
                ],
            },
            "C": {
                "D": [
                    ConnectedFakeLane(0, -3.5, 60.0, "C", "D", length=30.0),
                ],
            },
        }

    def get_lane(self, lane_index):
        return self.graph[lane_index[0]][lane_index[1]][lane_index[2]]


class S8ImmediateRampFakeRoadNetwork:
    def __init__(self):
        self.graph = {
            "A": {
                "B": [
                    ConnectedFakeLane(0, 3.5, 0.0, "A", "B", length=30.0),
                    ConnectedFakeLane(1, 0.0, 0.0, "A", "B", length=30.0),
                    ConnectedFakeLane(2, -3.5, 0.0, "A", "B", length=30.0),
                ],
            },
            "B": {
                "C": [
                    ConnectedFakeLane(0, -3.5, 30.0, "B", "C", length=30.0),
                ],
            },
        }

    def get_lane(self, lane_index):
        return self.graph[lane_index[0]][lane_index[1]][lane_index[2]]


class S8HardcodedBranchFakeRoadNetwork:
    def __init__(self):
        self.graph = {
            "3C0_1_": {
                "4G0_0_": [
                    ConnectedFakeLane(0, 3.5, 0.0, "3C0_1_", "4G0_0_", length=30.0),
                    ConnectedFakeLane(1, 0.0, 0.0, "3C0_1_", "4G0_0_", length=30.0),
                    ConnectedFakeLane(2, -3.5, 0.0, "3C0_1_", "4G0_0_", length=30.0),
                ],
                "4G1_0_": [
                    ConnectedFakeLane(0, -7.0, 0.0, "3C0_1_", "4G1_0_", length=30.0),
                ],
            },
            "4G0_0_": {
                "4G1_1_": [
                    ConnectedFakeLane(0, -3.5, 30.0, "4G0_0_", "4G1_1_", length=30.0),
                ],
            },
        }

    def get_lane(self, lane_index):
        return self.graph[lane_index[0]][lane_index[1]][lane_index[2]]


def _env_connected(vehicle):
    road_network = ConnectedFakeRoadNetwork()
    vehicle.lane = road_network.graph["A"]["B"][0]
    vehicle.position = np.asarray([25.0, 0.0], dtype=np.float32)
    vehicle.navigation = SimpleNamespace(checkpoints=["A", "B", "C"])
    return SimpleNamespace(
        agents={"agent0": vehicle},
        engine=SimpleNamespace(
            current_map=SimpleNamespace(road_network=road_network),
            traffic_manager=SimpleNamespace(_traffic_vehicles=[]),
        ),
    )


def _env_connected_two_lane(vehicle):
    road_network = TwoLaneConnectedFakeRoadNetwork()
    vehicle.lane = road_network.graph["A"]["B"][1]
    vehicle.position = np.asarray([25.0, 0.0], dtype=np.float32)
    vehicle.navigation = SimpleNamespace(checkpoints=["A", "B", "C"])
    return SimpleNamespace(
        agents={"agent0": vehicle},
        engine=SimpleNamespace(
            current_map=SimpleNamespace(road_network=road_network),
            traffic_manager=SimpleNamespace(_traffic_vehicles=[]),
        ),
    )


def _env_s8_exit(vehicle, scenario_id="S8_ego_exit_to_ramp", local_route="R6_exit_to_ramp"):
    road_network = S8ExitFakeRoadNetwork()
    vehicle.lane = road_network.graph["A"]["B"][1]
    vehicle.position = np.asarray([25.0, 0.0], dtype=np.float32)
    vehicle.navigation = SimpleNamespace(checkpoints=["A", "B", "C", "D"])
    return SimpleNamespace(
        config={"scenario_id": scenario_id, "local_route": local_route},
        agents={"agent0": vehicle},
        engine=SimpleNamespace(
            current_map=SimpleNamespace(road_network=road_network),
            traffic_manager=SimpleNamespace(_traffic_vehicles=[]),
        ),
    )


def _env_s8_immediate_ramp(vehicle, scenario_id="S8_ego_exit_to_ramp", local_route="R6_exit_to_ramp"):
    road_network = S8ImmediateRampFakeRoadNetwork()
    vehicle.lane = road_network.graph["A"]["B"][2]
    vehicle.position = np.asarray([25.0, -3.5], dtype=np.float32)
    vehicle.navigation = SimpleNamespace(checkpoints=["A", "B", "C"])
    return SimpleNamespace(
        config={"scenario_id": scenario_id, "local_route": local_route},
        agents={"agent0": vehicle},
        engine=SimpleNamespace(
            current_map=SimpleNamespace(road_network=road_network),
            traffic_manager=SimpleNamespace(_traffic_vehicles=[]),
        ),
    )


def _env_s8_hardcoded_branch(vehicle):
    road_network = S8HardcodedBranchFakeRoadNetwork()
    vehicle.lane = road_network.graph["3C0_1_"]["4G0_0_"][2]
    vehicle.position = np.asarray([25.0, -3.5], dtype=np.float32)
    vehicle.navigation = SimpleNamespace(checkpoints=["3C0_1_", "4G0_0_", "4G1_1_"])
    return SimpleNamespace(
        config={"scenario_id": "S8_ego_exit_to_ramp", "local_route": "R6_exit_to_ramp"},
        agents={"agent0": vehicle},
        engine=SimpleNamespace(
            current_map=SimpleNamespace(road_network=road_network),
            traffic_manager=SimpleNamespace(_traffic_vehicles=[]),
        ),
    )


def test_rule_maker_returns_target_points_for_all_agents():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 0.0, 0.0, 1),
            "agent1": _vehicle("agent1", -10.0, 0.0, 1),
        },
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0)

    decisions = rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})

    assert set(decisions) == {"agent0", "agent1"}
    assert decisions["agent0"]["target_point"].shape == (2,)
    assert decisions["agent1"]["target_point"].shape == (2,)
    assert decisions["agent0"]["target_point"][0] > 0.0


def test_rule_maker_scores_joint_actions_and_avoids_blocked_left_lane():
    env = _env(
        agents={"agent0": _vehicle("agent0", 0.0, 0.0, 1)},
        traffic=[_vehicle("left_blocker", 16.0, 3.5, 0)],
    )
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        lane_change_preference=2.0,
    )

    decisions = rule_maker.compute(env, ["agent0"], planner_batch={})

    assert decisions["agent0"]["target_point"][0] > 0.0
    assert decisions["agent0"]["target_point"][1] < -1.0


def test_rule_maker_marks_rear_agent_as_follower_when_same_decision_same_lane_and_clear():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 0.0, 0.0, 1),
        },
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0)

    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert debug is not None
    assert debug["dynamic_roles"]["agent0"] == "leader"
    assert debug["dynamic_roles"]["agent1"] == "follower"


def test_rule_maker_marks_rear_agent_as_leader_when_blocked_between_agents():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 0.0, 0.0, 1),
        },
        traffic=[_vehicle("blocker", 5.0, 0.0, 1)],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0)

    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert debug is not None
    assert debug["dynamic_roles"]["agent0"] == "leader"
    assert debug["dynamic_roles"]["agent1"] == "leader"


def test_rule_maker_marks_rear_agent_as_leader_when_agents_are_not_in_same_lane():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 0.0, -3.5, 2),
        },
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, locked_on_reset=False)

    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert debug is not None
    assert debug["dynamic_roles"]["agent0"] == "leader"
    assert debug["dynamic_roles"]["agent1"] == "leader"


def test_rule_maker_prefers_coherent_joint_combo_for_close_platoon_even_with_lane_change_bias():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 2.0, 0.0, 1),
        },
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        lane_change_preference=2.0,
    )

    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert debug is not None
    assert debug["best_actions"]["agent0"] == debug["best_actions"]["agent1"]


def test_debug_compute_selects_requested_action_candidate():
    env = _env(
        agents={"agent0": _vehicle("agent0", 10.0, 0.0, 1)},
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, locked_on_reset=False)

    decisions = rule_maker.debug_compute(env, ["agent0"], planner_batch={}, manual_actions={"agent0": 1})
    debug = rule_maker.get_last_debug()

    assert decisions["agent0"]["action"] == 1
    assert decisions["agent0"]["target_point"].shape == (2,)
    assert debug is not None
    assert debug["manual_mode"] is True
    assert debug["requested_actions"] == {"agent0": 1}
    assert debug["best_actions"] == {"agent0": 1}
    assert debug["invalid_actions"] == {}


def test_rule_maker_helper_saves_candidate_debug_plot_png(tmp_path):
    env = _env(
        agents={"agent0": _vehicle("candidate_plot_agent", 10.0, 0.0, 1)},
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, locked_on_reset=False)
    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], [])

    output_path = save_candidate_debug_plot(
        env.agents["agent0"],
        candidates,
        counter=1,
        output_dir=tmp_path,
    )

    assert output_path == tmp_path / "candidate_candidate_plot_agent_000001.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_rule_maker_helper_saves_lane_pair_debug_plot_png(tmp_path):
    vehicle = _vehicle("lane_pair_plot_agent", 10.0, 0.0, 1)
    source_lane = FakeLane(1, 0.0)
    target_lane = FakeLane(2, -3.5)

    output_path = save_lane_pair_debug_plot(
        vehicle,
        source_lane,
        target_lane,
        action=1,
        counter=1,
        output_dir=tmp_path,
    )

    assert output_path == tmp_path / "lane_pair_lane_pair_plot_agent_action_1_000001.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_rule_maker_helper_saves_s8_route_lanes_debug_plot_png(tmp_path, monkeypatch):
    vehicle = _vehicle("s8_route_plot_agent", 25.0, 0.0, 1)
    env = _env_s8_exit(vehicle)
    seen_lane_indices = []
    original_lane_centerline_points = rule_maker_module.save_s8_route_lanes_debug_plot.__globals__["_lane_centerline_points"]

    def record_lane_centerline_points(lane):
        seen_lane_indices.append(tuple(getattr(lane, "index", ()) or ()))
        return original_lane_centerline_points(lane)

    monkeypatch.setitem(
        rule_maker_module.save_s8_route_lanes_debug_plot.__globals__,
        "_lane_centerline_points",
        record_lane_centerline_points,
    )

    output_path = save_s8_route_lanes_debug_plot(
        vehicle,
        env.engine.current_map.road_network,
        ["A", "B"],
        vehicle.lane,
        counter=1,
        output_dir=tmp_path,
    )

    assert output_path == tmp_path / "s8_route_lanes_s8_route_plot_agent_000001.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert ("B", "C", 2) in seen_lane_indices
    assert ("C", "D", 0) in seen_lane_indices


def test_rule_maker_idm_profile_slows_terminal_speed_when_slow_front_vehicle_exists():
    env = _env(
        agents={"agent0": _vehicle("agent0", 0.0, 0.0, 1, speed_km_h=28.0)},
        traffic=[_vehicle("front_slow", 18.0, 0.0, 1, speed_km_h=10.0)],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0)

    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], rule_maker._traffic_vehicles(env))
    keep_candidate = next(candidate for candidate in candidates if int(candidate["action"]) == 0)

    assert keep_candidate["terminal_speed_km_h"] < float(env.agents["agent0"].speed_km_h)
    assert keep_candidate["trajectory_world"][-1, 0] < 16.0


def test_rule_maker_mobil_gain_prefers_free_adjacent_lane_over_blocked_keep_lane():
    env = _env(
        agents={"agent0": _vehicle("agent0", 0.0, 0.0, 1, speed_km_h=28.0)},
        traffic=[_vehicle("front_slow", 18.0, 0.0, 1, speed_km_h=10.0)],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0)

    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], rule_maker._traffic_vehicles(env))
    keep_candidate = next(candidate for candidate in candidates if int(candidate["action"]) == 0)
    right_candidate = next(candidate for candidate in candidates if int(candidate["action"]) == 1)

    assert right_candidate["mobil_gain"] > keep_candidate["mobil_gain"]


def test_rule_maker_mobil_penalizes_lane_change_when_target_lane_rear_vehicle_must_hard_brake():
    env = _env(
        agents={"agent0": _vehicle("agent0", 0.0, 0.0, 1, speed_km_h=20.0)},
        traffic=[
            _vehicle("front_slow", 18.0, 0.0, 1, speed_km_h=10.0),
            _vehicle("right_rear_fast", -5.0, -3.5, 2, speed_km_h=32.0),
        ],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0)

    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], rule_maker._traffic_vehicles(env))
    keep_candidate = next(candidate for candidate in candidates if int(candidate["action"]) == 0)
    right_candidate = next(candidate for candidate in candidates if int(candidate["action"]) == 1)

    assert right_candidate["rear_vehicle_post_accel_mps2"] <= -rule_maker.idm_comfortable_brake_mps2
    assert right_candidate["mobil_gain"] < keep_candidate["mobil_gain"]


def test_rule_maker_lane_change_trajectory_uses_smooth_lateral_transition():
    env = _env(
        agents={"agent0": _vehicle("agent0", 0.0, 0.0, 1, speed_km_h=20.0)},
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], rule_maker._traffic_vehicles(env))
    right_candidate = next(candidate for candidate in candidates if int(candidate["action"]) == 1)
    traj = np.asarray(right_candidate["trajectory_world"], dtype=np.float32)

    lateral = traj[:, 1]
    assert lateral[0] > -0.2
    assert lateral[-1] < -3.0
    assert np.all(np.diff(lateral) <= 1e-4)
    assert np.any(np.abs(lateral[1:-1]) < 3.0)


def test_rule_maker_trajectory_can_continue_across_connected_lane_segments():
    vehicle = _vehicle("agent0", 25.0, 0.0, 1, speed_km_h=25.0)
    env = _env_connected(vehicle)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], [])
    keep_candidate = next(candidate for candidate in candidates if int(candidate["action"]) == 0)
    traj = np.asarray(keep_candidate["trajectory_world"], dtype=np.float32)

    assert traj[-1, 0] > 30.0


def test_rule_maker_lane_change_continues_on_target_lane_family_across_blocks():
    vehicle = _vehicle("agent0", 25.0, 0.0, 1, speed_km_h=25.0)
    env = _env_connected_two_lane(vehicle)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], [])
    right_candidate = next(candidate for candidate in candidates if int(candidate["action"]) == 1)
    traj = np.asarray(right_candidate["trajectory_world"], dtype=np.float32)

    assert traj[-1, 0] > 30.0
    assert traj[-1, 1] < -4.0


def test_s8_reference_lane_chain_uses_rightmost_exit_lane_then_ramp_lane():
    vehicle = _vehicle("agent0", 25.0, 0.0, 1, speed_km_h=25.0)
    env = _env_s8_exit(vehicle)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    lane_chain = rule_maker._reference_lane_chain(env, env.agents["agent0"], env.agents["agent0"].lane)
    lane_indices = [tuple(lane.index) for lane in lane_chain]

    assert lane_indices == [
        ("A", "B", 1),
        ("B", "C", 2),
        ("C", "D", 0),
    ]


def test_reference_lane_chain_keeps_current_lane_slot_for_non_s8_routes():
    vehicle = _vehicle("agent0", 25.0, 0.0, 1, speed_km_h=25.0)
    env = _env_s8_exit(vehicle, scenario_id="S7_ego_merge_from_ramp", local_route="R7_merge_core")
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    lane_chain = rule_maker._reference_lane_chain(env, env.agents["agent0"], env.agents["agent0"].lane)
    lane_indices = [tuple(lane.index) for lane in lane_chain]

    assert lane_indices == [
        ("A", "B", 1),
        ("B", "C", 1),
        ("C", "D", 0),
    ]


def test_s8_action_one_targets_downstream_ramp_lane_when_current_right_lane_has_no_neighbor():
    vehicle = _vehicle("agent0", 25.0, -3.5, 2, speed_km_h=25.0)
    env = _env_s8_immediate_ramp(vehicle)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    candidate = rule_maker._build_coarse_trajectory(env, env.agents["agent0"], 1, [])

    assert candidate is not None
    assert candidate["target_lane_index"] == ("B", "C", 0)


def test_s8_downstream_target_lane_uses_hardcoded_exit_branch_for_3c0_right_lane():
    vehicle = _vehicle("agent0", 25.0, -3.5, 2, speed_km_h=25.0)
    env = _env_s8_hardcoded_branch(vehicle)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    target_lane = rule_maker._S8_downstream_target_lane(env, env.agents["agent0"], env.agents["agent0"].lane)

    assert target_lane is not None
    assert tuple(target_lane.index) == ("3C0_1_", "4G1_0_", 0)


def test_debug_compute_s8_action_one_uses_downstream_ramp_lane():
    vehicle = _vehicle("agent0", 25.0, -3.5, 2, speed_km_h=25.0)
    env = _env_s8_immediate_ramp(vehicle)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    decisions = rule_maker.debug_compute(env, ["agent0"], planner_batch={}, manual_actions={"agent0": 1})
    debug = rule_maker.get_last_debug()

    assert decisions["agent0"]["action"] == 1
    assert debug is not None
    assert debug["best_actions"] == {"agent0": 1}
    selected_candidates = [
        candidate
        for candidate in debug["candidates_by_agent"]["agent0"]
        if candidate["selected"]
    ]
    assert len(selected_candidates) == 1
    assert selected_candidates[0]["target_lane_index"] == ("B", "C", 0)


def test_non_s8_action_one_still_requires_current_road_neighbor_lane():
    vehicle = _vehicle("agent0", 25.0, -3.5, 2, speed_km_h=25.0)
    env = _env_s8_immediate_ramp(vehicle, scenario_id="S7_ego_merge_from_ramp", local_route="R7_merge_core")
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    candidate = rule_maker._build_coarse_trajectory(env, env.agents["agent0"], 1, [])

    assert candidate is None


def test_debug_compute_records_invalid_manual_action_without_fallback():
    vehicle = _vehicle("agent0", 25.0, -3.5, 2, speed_km_h=25.0)
    env = _env_s8_immediate_ramp(vehicle, scenario_id="S7_ego_merge_from_ramp", local_route="R7_merge_core")
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    decisions = rule_maker.debug_compute(env, ["agent0"], planner_batch={}, manual_actions={"agent0": 1})
    debug = rule_maker.get_last_debug()

    assert "agent0" not in decisions
    assert debug is not None
    assert debug["manual_mode"] is True
    assert debug["requested_actions"] == {"agent0": 1}
    assert debug["best_actions"] == {}
    assert debug["invalid_actions"] == {"agent0": 1}


def test_s8_candidates_reward_right_lane_change_with_force_lane_score():
    vehicle = _vehicle("agent0", 25.0, 0.0, 1, speed_km_h=25.0)
    env = _env_s8_exit(vehicle)
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        num_waypoints=8,
        force_lane_change=800.0,
    )

    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], [])

    assert candidates
    for candidate in candidates:
        assert "forced_lane_mean_lateral_distance_m" not in candidate
        assert "force_lane_score" in candidate
        expected_score = 800.0 if int(candidate["action"]) == 1 else 0.0
        assert candidate["force_lane_score"] == pytest.approx(expected_score)


def test_s8_force_lane_score_prefers_right_lane_change_action():
    vehicle = _vehicle("agent0", 25.0, 0.0, 1, speed_km_h=25.0)
    env = _env_s8_exit(vehicle)
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        num_waypoints=8,
        force_lane_change=800.0,
    )
    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], [])
    scores_by_action = {int(candidate["action"]): float(candidate["force_lane_score"]) for candidate in candidates}

    assert scores_by_action[1] > scores_by_action[0]
    assert scores_by_action[1] > scores_by_action[-1]


def test_non_s8_candidates_do_not_record_force_lane_score():
    vehicle = _vehicle("agent0", 25.0, 0.0, 1, speed_km_h=25.0)
    env = _env_s8_exit(vehicle, scenario_id="S7_ego_merge_from_ramp", local_route="R7_merge_core")
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        num_waypoints=8,
        force_lane_change=800.0,
    )

    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], [])

    assert candidates
    assert all("forced_lane_mean_lateral_distance_m" not in candidate for candidate in candidates)
    assert all("force_lane_score" not in candidate for candidate in candidates)


def test_rule_maker_joint_agent_safety_score_uses_soft_reward_not_hard_constraint():
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0)
    close_combo = (
        {"trajectory_world": np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float32)},
        {"trajectory_world": np.asarray([[0.5, 0.0], [5.5, 0.0], [10.5, 0.0]], dtype=np.float32)},
    )
    far_combo = (
        {"trajectory_world": np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float32)},
        {"trajectory_world": np.asarray([[0.0, 10.0], [5.0, 10.0], [10.0, 10.0]], dtype=np.float32)},
    )

    close_score = rule_maker._joint_agent_safety_score(close_combo)
    far_score = rule_maker._joint_agent_safety_score(far_combo)

    assert np.isfinite(close_score)
    assert close_score > -10.0
    assert far_score > close_score


def test_rule_maker_joint_agent_safety_score_is_more_tolerant_for_adjacent_lane_parallel_motion():
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0)
    longitudinal_close_combo = (
        {"trajectory_world": np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float32)},
        {"trajectory_world": np.asarray([[2.0, 0.0], [7.0, 0.0], [12.0, 0.0]], dtype=np.float32)},
    )
    lateral_parallel_combo = (
        {"trajectory_world": np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float32)},
        {"trajectory_world": np.asarray([[0.0, 2.0], [5.0, 2.0], [10.0, 2.0]], dtype=np.float32)},
    )

    longitudinal_score = rule_maker._joint_agent_safety_score(longitudinal_close_combo)
    lateral_score = rule_maker._joint_agent_safety_score(lateral_parallel_combo)

    assert lateral_score > longitudinal_score


def test_rule_maker_starts_locked_with_shared_action_and_fixed_roles():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 2.0, 0.0, 1),
        },
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        lane_change_preference=2.0,
        locked_on_reset=True,
        risk_front_gap_trigger_m=0.0,
        risk_side_gap_trigger_m=0.0,
    )
    rule_maker.reset(env, ["agent0", "agent1"])

    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert debug is not None
    assert debug["formation_locked"] is True
    assert debug["best_actions"]["agent0"] == debug["best_actions"]["agent1"]
    assert debug["dynamic_roles"] == {"agent0": "leader", "agent1": "follower"}


def test_rule_maker_unlocks_after_risk_trigger_and_recovers_dynamic_roles():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 2.0, -3.5, 2),
        },
        traffic=[_vehicle("front_risk", 16.0, 0.0, 1)],
    )
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        locked_on_reset=True,
        risk_front_gap_trigger_m=100.0,
        risk_side_gap_trigger_m=100.0,
    )
    rule_maker.reset(env, ["agent0", "agent1"])

    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert debug is not None
    assert debug["formation_locked"] is False
    assert debug["risk_triggered"] is True
    assert debug["dynamic_roles"]["agent0"] == "leader"
    assert debug["dynamic_roles"]["agent1"] == "leader"
