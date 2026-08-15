from __future__ import annotations

from math import inf
from types import SimpleNamespace

import numpy as np
import pytest

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
        speed_km_h=24.0,
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
    policy.merge_activation_step = 11
    policy.merge_rear_ttc_min_s = 4.0
    policy.MAX_LONG_DIST = 30.0
    policy.target_speed = 24.0
    policy.merge_completed = False
    policy.merge_sweep_committed = False
    policy.merge_policy_step = 0
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
    assert policy.merge_policy_step == 1


def test_merge_policy_forces_after_policy_step_two(monkeypatch) -> None:
    road_network = _RoadNetwork()
    target_lane = road_network.mainline[-1]
    policy = _make_policy(road_network, road_network.ramp, [])
    monkeypatch.setattr(IDMPolicy, "lane_change_policy", lambda self, objects: (None, 30.0, self.control_object.lane))

    for _ in range(11):
        result = policy.lane_change_policy([])
        assert result[2] is road_network.ramp
        assert policy.action_info["merge_force_active"] is False

    result = policy.lane_change_policy([])

    assert result[2] is target_lane
    assert policy.action_info["merge_force_active"] is True
    assert policy.merge_policy_step == 12


def test_merge_policy_restores_cruise_speed_when_gap_is_safe(monkeypatch) -> None:
    road_network = _RoadNetwork()
    target_lane = road_network.mainline[-1]
    front = SimpleNamespace(lane=target_lane, position=np.asarray([30.0, 0.0]))
    rear = SimpleNamespace(lane=target_lane, position=np.asarray([-20.0, 0.0]), speed_km_h=24.0)
    policy = _make_policy(road_network, road_network.connector, [front, rear])
    policy.merge_policy_step = 11
    monkeypatch.setattr(IDMPolicy, "lane_change_policy", lambda self, objects: (None, 30.0, self.control_object.lane))

    result = policy.lane_change_policy([front, rear])

    assert policy.target_speed == 24.0
    assert policy.action_info["merge_gap_accepted"] is True
    assert policy.action_info["merge_rear_ttc_s"] == inf
    assert policy.action_info["merge_force_active"] is True
    assert policy.action_info["merge_completed"] is False
    assert result[2] is target_lane


def test_merge_policy_uses_recipe_bound_gap_on_overlapping_connector(monkeypatch) -> None:
    road_network = _RoadNetwork()
    target_lane = road_network.mainline[-1]
    generic_front = SimpleNamespace(
        lane=target_lane, position=np.asarray([4.0, 0.0])
    )
    generic_rear = SimpleNamespace(
        lane=target_lane, position=np.asarray([-4.0, 0.0]), speed_km_h=24.0
    )
    designated_front = SimpleNamespace(
        lane=target_lane, position=np.asarray([30.0, 0.0])
    )
    designated_rear = SimpleNamespace(
        lane=target_lane, position=np.asarray([-20.0, 0.0]), speed_km_h=24.0
    )
    policy = _make_policy(
        road_network, road_network.connector, [generic_front, generic_rear]
    )
    policy.control_object.scenario_target_front_vehicle = designated_front
    policy.control_object.scenario_target_rear_vehicle = designated_rear
    policy.merge_policy_step = 11
    monkeypatch.setattr(
        IDMPolicy,
        "lane_change_policy",
        lambda self, objects: (None, 30.0, self.control_object.lane),
    )

    result = policy.lane_change_policy([generic_front, generic_rear])

    assert policy.action_info["merge_gap_accepted"] is True
    assert policy.action_info["merge_front_gap"] == pytest.approx(30.0)
    assert policy.action_info["merge_rear_gap"] == pytest.approx(20.0)
    assert result == (designated_front, pytest.approx(30.0), target_lane)


def test_merge_target_lane_rebinds_across_multilane_graph_seam() -> None:
    road_network = _RoadNetwork()
    downstream_lanes = [_Lane("C", "D", lane_id) for lane_id in range(3)]
    road_network.graph["C"]["D"] = downstream_lanes
    policy = _make_policy(road_network, downstream_lanes[2], [])

    target = policy._find_merge_target_lane()

    assert target is downstream_lanes[1]
    assert policy.control_object.navigation.merge_target_lane is downstream_lanes[1]


def test_merge_sweep_remains_committed_after_gap_window_closes(monkeypatch) -> None:
    road_network = _RoadNetwork()
    target_lane = road_network.mainline[-1]
    designated_front = SimpleNamespace(
        lane=target_lane, position=np.asarray([30.0, 0.0])
    )
    designated_rear = SimpleNamespace(
        lane=target_lane, position=np.asarray([-20.0, 0.0]), speed_km_h=24.0
    )
    policy = _make_policy(road_network, road_network.connector, [])
    policy.control_object.scenario_target_front_vehicle = designated_front
    policy.control_object.scenario_target_rear_vehicle = designated_rear
    policy.merge_policy_step = 11
    monkeypatch.setattr(
        IDMPolicy,
        "lane_change_policy",
        lambda self, objects: (None, 30.0, self.control_object.lane),
    )

    first = policy.lane_change_policy([])
    designated_rear.position = np.asarray([-4.0, 0.0])
    second = policy.lane_change_policy([])

    assert first[2] is target_lane
    assert policy.action_info["merge_gap_accepted"] is False
    assert policy.action_info["merge_sweep_committed"] is True
    assert policy.target_speed == policy.merge_cruise_speed_kmh
    assert second[2] is target_lane


def test_premerge_desired_gap_uses_mps_units() -> None:
    road_network = _RoadNetwork()
    policy = _make_policy(road_network, road_network.connector, [])
    front = SimpleNamespace(speed_km_h=24.0)

    gap_m = policy.desired_gap(policy.control_object, front)

    assert gap_m == pytest.approx(7.0 + 0.8 * (24.0 / 3.6))


def test_bound_gap_requires_designated_front_and_rear_order(monkeypatch) -> None:
    road_network = _RoadNetwork()
    target_lane = road_network.mainline[-1]
    policy = _make_policy(road_network, road_network.connector, [])
    policy.control_object.scenario_target_front_vehicle = SimpleNamespace(
        position=np.asarray([30.0, 0.0])
    )
    policy.control_object.scenario_target_rear_vehicle = SimpleNamespace(
        position=np.asarray([10.0, 0.0]), speed_km_h=24.0
    )
    policy.merge_policy_step = 11
    monkeypatch.setattr(
        IDMPolicy,
        "lane_change_policy",
        lambda self, objects: (None, 30.0, self.control_object.lane),
    )

    result = policy.lane_change_policy([])

    assert policy.action_info["merge_gap_accepted"] is False
    assert policy.action_info["merge_sweep_committed"] is False
    assert result[2] is road_network.connector


def test_merge_policy_uses_rear_speed_when_safe_rear_vehicle_is_faster(monkeypatch) -> None:
    road_network = _RoadNetwork()
    target_lane = road_network.mainline[-1]
    rear = SimpleNamespace(lane=target_lane, position=np.asarray([-20.0, 0.0]), speed_km_h=30.0)
    policy = _make_policy(road_network, road_network.connector, [rear])
    policy.merge_policy_step = 11
    monkeypatch.setattr(IDMPolicy, "lane_change_policy", lambda self, objects: (None, 30.0, self.control_object.lane))

    result = policy.lane_change_policy([rear])

    assert policy.action_info["merge_gap_accepted"] is True
    assert policy.action_info["merge_rear_ttc_s"] == pytest.approx(12.0)
    assert policy.target_speed == pytest.approx(24.0)
    assert result[2] is target_lane


def test_merge_policy_holds_connector_lane_when_gap_is_unsafe(monkeypatch) -> None:
    road_network = _RoadNetwork()
    target_lane = road_network.mainline[-1]
    rear = SimpleNamespace(lane=target_lane, position=np.asarray([-4.0, 0.0]), speed_km_h=24.0)
    policy = _make_policy(road_network, road_network.connector, [rear])
    policy.merge_policy_step = 11
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


def test_merge_policy_rejects_gap_when_rear_ttc_is_unsafe(monkeypatch) -> None:
    road_network = _RoadNetwork()
    target_lane = road_network.mainline[-1]
    rear = SimpleNamespace(lane=target_lane, position=np.asarray([-12.0, 0.0]), speed_km_h=60.0)
    policy = _make_policy(road_network, road_network.connector, [rear])
    policy.merge_policy_step = 11
    monkeypatch.setattr(IDMPolicy, "lane_change_policy", lambda self, objects: (None, 30.0, self.control_object.lane))

    result = policy.lane_change_policy([rear])

    assert policy.action_info["merge_rear_gap"] == 12.0
    assert policy.action_info["merge_rear_ttc_s"] == pytest.approx(1.2)
    assert policy.action_info["merge_gap_accepted"] is False
    assert result[2] is road_network.connector


def test_merge_policy_latches_completed_after_entering_mainline(monkeypatch) -> None:
    road_network = _RoadNetwork()
    policy = _make_policy(road_network, road_network.mainline[-1], [])
    policy.target_speed = 17.0
    sentinel = (object(), 12.0, road_network.mainline[-1])
    monkeypatch.setattr(IDMPolicy, "lane_change_policy", lambda self, objects: sentinel)

    result = policy.lane_change_policy([])

    assert result == sentinel
    assert policy.target_speed == 20.0
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
    policy.merge_policy_step = 7
    monkeypatch.setattr(IDMPolicy, "reset", lambda self: None)

    policy.reset()

    assert policy.merge_completed is False
    assert policy.merge_policy_step == 0
