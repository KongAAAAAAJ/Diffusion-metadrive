from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from envs.diffusion_envs.idm_merge_policy import IDMMergePolicy, StartEdgeNodeNavigation
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.engine.base_engine import BaseEngine
from metadrive.policy.idm_policy import IDMPolicy


class _Lane:
    def __init__(self, start: str, end: str, lane_id: int = 0, length: float = 100.0) -> None:
        self.index = (start, end, lane_id)
        self.length = length

    def local_coordinates(self, position):
        return float(position[0]), float(position[1])


class _RoadNetwork:
    def __init__(self) -> None:
        self.ramp = _Lane("A", "B")
        self.connector = _Lane("B", "C")
        self.mainline = [_Lane("A", "C", 0), _Lane("A", "C", 1)]
        self.after_merge = _Lane("C", "D")
        self.unrelated_shortcut = _Lane("B", "D")
        self.graph = {
            "A": {"B": [self.ramp], "C": self.mainline},
            "B": {"C": [self.connector], "D": [self.unrelated_shortcut]},
            "C": {"D": [self.after_merge]},
        }

    def shortest_path(self, start, destination):
        assert start[0] == "B"
        assert destination == "D"
        return ["B", "C", "D"]


def test_start_edge_navigation_prepends_actual_spawn_edge() -> None:
    road_network = _RoadNetwork()

    checkpoints = StartEdgeNodeNavigation._build_start_edge_checkpoints(
        road_network,
        road_network.ramp.index,
        "D",
    )

    assert checkpoints == ["A", "B", "C", "D"]


def test_start_edge_navigation_records_force_merge_road(monkeypatch) -> None:
    road_network = _RoadNetwork()
    navigation = object.__new__(StartEdgeNodeNavigation)
    monkeypatch.setattr(
        BaseEngine,
        "singleton",
        SimpleNamespace(current_map=SimpleNamespace(road_network=road_network)),
    )
    navigation.checkpoints = []
    monkeypatch.setattr(NodeNetworkNavigation, "set_route", lambda self, lane_index, destination: None)

    navigation.set_route(road_network.ramp.index, "D")

    assert navigation.merge_force_road == ("B", "C")


def _make_policy(road_network: _RoadNetwork, vehicle_lane, objects):
    policy = object.__new__(IDMMergePolicy)
    policy.control_object = SimpleNamespace(
        lane=vehicle_lane,
        position=np.asarray([0.0, 0.0]),
        navigation=SimpleNamespace(
            checkpoints=["A", "B", "C", "D"],
            map=SimpleNamespace(road_network=road_network),
            merge_branch_roads={("A", "B"), ("B", "C")},
            merge_force_road=("B", "C"),
            merge_target_lane=road_network.mainline[-1],
        ),
    )
    policy.merge_front_gap_m = 25.0
    policy.merge_rear_gap_m = 15.0
    policy.merge_creep_speed_kmh = 5.0
    policy.merge_cruise_speed_kmh = 24.0
    policy.MAX_LONG_DIST = 30.0
    policy.target_speed = 24.0
    policy.merge_completed = False
    policy.action_info = {}
    return policy


def test_merge_policy_creeps_when_front_or_rear_gap_is_unsafe(monkeypatch) -> None:
    road_network = _RoadNetwork()
    target_lane = road_network.mainline[-1]
    front = SimpleNamespace(lane=target_lane, position=np.asarray([20.0, 0.0]))
    rear = SimpleNamespace(lane=target_lane, position=np.asarray([-4.0, 0.0]))
    policy = _make_policy(road_network, road_network.ramp, [front, rear])
    monkeypatch.setattr(IDMPolicy, "lane_change_policy", lambda self, objects: (None, 30.0, self.control_object.lane))

    policy.lane_change_policy([front, rear])

    assert policy.target_speed == 5.0
    assert policy.action_info["merge_active"] is True
    assert policy.action_info["merge_front_gap"] == 20.0
    assert policy.action_info["merge_rear_gap"] == 4.0
    assert policy.action_info["merge_gap_accepted"] is False
    assert policy.action_info["merge_force_active"] is False


def test_merge_policy_does_not_force_target_lane_before_connector(monkeypatch) -> None:
    road_network = _RoadNetwork()
    policy = _make_policy(road_network, road_network.ramp, [])
    sentinel = (object(), 30.0, road_network.ramp)
    monkeypatch.setattr(IDMPolicy, "lane_change_policy", lambda self, objects: sentinel)

    result = policy.lane_change_policy([])

    assert result is sentinel
    assert policy.action_info["merge_force_active"] is False


def test_merge_policy_restores_cruise_speed_when_gap_is_safe(monkeypatch) -> None:
    road_network = _RoadNetwork()
    target_lane = road_network.mainline[-1]
    front = SimpleNamespace(lane=target_lane, position=np.asarray([30.0, 0.0]))
    rear = SimpleNamespace(lane=target_lane, position=np.asarray([-20.0, 0.0]))
    policy = _make_policy(road_network, road_network.connector, [front, rear])
    monkeypatch.setattr(IDMPolicy, "lane_change_policy", lambda self, objects: (None, 30.0, self.control_object.lane))

    result = policy.lane_change_policy([front, rear])

    assert policy.target_speed == 24.0
    assert policy.action_info["merge_gap_accepted"] is True
    assert policy.action_info["merge_force_active"] is True
    assert policy.action_info["merge_completed"] is False
    assert result[2] is target_lane


def test_merge_policy_holds_connector_lane_when_gap_is_unsafe(monkeypatch) -> None:
    road_network = _RoadNetwork()
    target_lane = road_network.mainline[-1]
    rear = SimpleNamespace(lane=target_lane, position=np.asarray([-4.0, 0.0]))
    policy = _make_policy(road_network, road_network.connector, [rear])
    sentinel_front = object()
    monkeypatch.setattr(
        IDMPolicy,
        "lane_change_policy",
        lambda self, objects: (sentinel_front, 12.0, self.control_object.lane),
    )

    result = policy.lane_change_policy([rear])

    assert policy.target_speed == 5.0
    assert policy.action_info["merge_gap_accepted"] is False
    assert policy.action_info["merge_force_active"] is True
    assert result == (sentinel_front, 12.0, road_network.connector)


def test_merge_policy_latches_completed_after_entering_mainline(monkeypatch) -> None:
    road_network = _RoadNetwork()
    policy = _make_policy(road_network, road_network.mainline[-1], [])
    policy.target_speed = 17.0
    sentinel = (object(), 12.0, road_network.mainline[-1])
    monkeypatch.setattr(IDMPolicy, "lane_change_policy", lambda self, objects: sentinel)

    result = policy.lane_change_policy([])

    assert result == sentinel
    assert policy.target_speed == 17.0
    assert policy.merge_completed is True
    assert policy.action_info["merge_active"] is False
    assert policy.action_info["merge_force_active"] is False
    assert policy.action_info["merge_completed"] is True


def test_merge_policy_locks_current_lane_after_completed_latch(monkeypatch) -> None:
    road_network = _RoadNetwork()
    policy = _make_policy(road_network, road_network.mainline[-1], [])
    parent_front = object()
    parent_lane_change_target = road_network.after_merge
    monkeypatch.setattr(
        IDMPolicy,
        "lane_change_policy",
        lambda self, objects: (parent_front, 12.0, parent_lane_change_target),
    )

    result = policy.lane_change_policy([])

    assert result == (parent_front, 12.0, road_network.mainline[-1])
    assert policy.merge_completed is True
    assert policy.action_info["merge_completed"] is True


def test_merge_policy_does_not_reenter_force_after_completed_latch(monkeypatch) -> None:
    road_network = _RoadNetwork()
    policy = _make_policy(road_network, road_network.mainline[-1], [])
    first_sentinel = (object(), 12.0, road_network.mainline[-1])
    monkeypatch.setattr(IDMPolicy, "lane_change_policy", lambda self, objects: first_sentinel)

    first_result = policy.lane_change_policy([])

    assert first_result == first_sentinel
    assert policy.merge_completed is True

    policy.control_object.lane = road_network.connector
    second_sentinel = (object(), 8.0, road_network.after_merge)
    monkeypatch.setattr(IDMPolicy, "lane_change_policy", lambda self, objects: second_sentinel)

    second_result = policy.lane_change_policy([])

    assert second_result == (second_sentinel[0], second_sentinel[1], road_network.connector)
    assert policy.merge_completed is True
    assert policy.action_info["merge_active"] is False
    assert policy.action_info["merge_force_active"] is False
    assert policy.action_info["merge_completed"] is True


def test_merge_policy_reset_clears_completed_latch(monkeypatch) -> None:
    road_network = _RoadNetwork()
    policy = _make_policy(road_network, road_network.mainline[-1], [])
    policy.merge_completed = True
    monkeypatch.setattr(IDMPolicy, "reset", lambda self: None)

    policy.reset()

    assert policy.merge_completed is False
