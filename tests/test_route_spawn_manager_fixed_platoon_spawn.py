from __future__ import annotations

from types import SimpleNamespace

import pytest

from envs.diffusion_envs import route_spawn_manager as route_spawn_module
from envs.diffusion_envs.route_spawn_manager import RouteAwareSpawnManager


class _FakeRoad:
    def __init__(self, start_node="A", end_node="B"):
        self.start_node = start_node
        self.end_node = end_node

    def lane_index(self, lane_idx: int):
        return (self.start_node, self.end_node, lane_idx)


class _FakeLane:
    def __init__(self, index=("A", "B", 0), length=80.0):
        self.index = index
        self.length = length

    def heading_theta_at(self, longitude):
        return 0.0


class _FakeRoadNetwork:
    def __init__(self, lanes_by_road):
        self.graph = {}
        for road, lanes in lanes_by_road.items():
            self.graph.setdefault(road.start_node, {})[road.end_node] = list(lanes)


class _FakeBlockNetwork:
    def __init__(self, lane_groups):
        self._lane_groups = lane_groups

    def get_positive_lanes(self):
        return self._lane_groups


class _FakeBlock:
    def __init__(self, graph_block_id, lane_groups):
        self.graph_block_id = graph_block_id
        self.block_network = _FakeBlockNetwork(lane_groups)


class _FixedUniformRng:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def uniform(self, low, high):
        return self.value


def _manager_with_fixed_spawn(
    monkeypatch,
    *,
    lanes=None,
    scenario_id=None,
    scenario=None,
    route_roads=None,
    route_block_ids=None,
    blocks=None,
    extra_lanes_by_road=None,
):
    lanes = list(lanes or [_FakeLane()])
    route_roads = list(route_roads or [_FakeRoad()])
    lanes_by_road = {route_roads[0]: lanes}
    for road in route_roads[1:]:
        lanes_by_road.setdefault(road, lanes)
    lanes_by_road.update(extra_lanes_by_road or {})
    manager = object.__new__(RouteAwareSpawnManager)
    manager._test_engine = SimpleNamespace(
        current_map=SimpleNamespace(
            road_network=_FakeRoadNetwork(lanes_by_road),
            blocks=list(blocks or []),
        ),
        global_config={
            "num_agents": 3,
            "platoon_fixed_route_spawn": True,
            "platoon_spawn_gap_m": 10.0,
            "platoon_spawn_tail_buffer_m": 6.0,
            "platoon_spawn_front_buffer_m": 8.0,
            "initial_speed_km_h": 25.0,
            "agent_configs": {
                "agent0": {},
                "agent1": {},
                "agent2": {},
            },
            "vehicle_config": {},
        },
    )
    if route_block_ids is not None:
        manager._test_engine.global_config["ego_main_route_block_ids"] = tuple(route_block_ids)
    if scenario_id is not None:
        manager._test_engine.global_config["scenario_id"] = scenario_id
    if scenario_id is not None and scenario is not None:
        monkeypatch.setitem(route_spawn_module.SCENARIO_BY_ID, scenario_id, scenario)
    monkeypatch.setattr(RouteAwareSpawnManager, "engine", property(lambda self: self._test_engine))
    manager.get_main_route_spawn_roads = lambda current_map: route_roads
    manager.update_destination_for = lambda agent_id, config: {**config, "destination": "B"}
    manager.ego_spawn_zones = []
    return manager


def test_fixed_route_spawn_configs_are_deterministic_and_near_route_start(monkeypatch):
    manager = _manager_with_fixed_spawn(monkeypatch)

    assert manager._apply_fixed_route_spawn_configs() is True

    configs = manager.engine.global_config["agent_configs"]
    assert configs["agent0"]["spawn_lane_index"] == ("A", "B", 0)
    assert configs["agent0"]["spawn_longitude"] == pytest.approx(26.0)
    assert configs["agent1"]["spawn_longitude"] == pytest.approx(16.0)
    assert configs["agent2"]["spawn_longitude"] == pytest.approx(6.0)
    assert configs["agent0"]["spawn_velocity_car_frame"] is True
    assert configs["agent0"]["destination"] == "B"


def test_fixed_route_spawn_zones_match_final_agent_configs(monkeypatch):
    manager = _manager_with_fixed_spawn(monkeypatch)

    assert manager._apply_fixed_route_spawn_configs() is True
    manager._cache_ego_spawn_zones()

    configs = manager.engine.global_config["agent_configs"]
    zones = manager.ego_spawn_zones
    assert len(zones) == 3
    for agent_idx, zone in enumerate(zones):
        config = configs[f"agent{agent_idx}"]
        assert zone["spawn_lane_index"] == config["spawn_lane_index"]
        assert zone["spawn_longitude"] == pytest.approx(config["spawn_longitude"])


def test_fixed_route_spawn_can_use_scenario_distance_on_first_route_road(monkeypatch):
    lanes = [
        _FakeLane(index=("A", "B", 0), length=200.0),
        _FakeLane(index=("A", "B", 1), length=200.0),
        _FakeLane(index=("A", "B", 2), length=200.0),
    ]
    scenario = SimpleNamespace(
        ego_spawn_lane_preference="rightmost",
        ego_spawn_lane_probabilities=None,
        ego_spawn_distance_to_route_end_m=100.0,
        ego_spawn_reference_block_id=None,
    )
    manager = _manager_with_fixed_spawn(
        monkeypatch,
        lanes=lanes,
        scenario_id="test_distance_on_first_road",
        scenario=scenario,
    )

    assert manager._apply_fixed_route_spawn_configs() is True

    configs = manager.engine.global_config["agent_configs"]
    assert configs["agent0"]["spawn_lane_index"] == ("A", "B", 2)
    assert configs["agent0"]["spawn_longitude"] == pytest.approx(100.0)
    assert configs["agent1"]["spawn_longitude"] == pytest.approx(90.0)
    assert configs["agent2"]["spawn_longitude"] == pytest.approx(80.0)


def test_fixed_route_spawn_can_use_middle_lane_preference(monkeypatch):
    lanes = [
        _FakeLane(index=("A", "B", 0), length=200.0),
        _FakeLane(index=("A", "B", 1), length=200.0),
        _FakeLane(index=("A", "B", 2), length=200.0),
    ]
    scenario = SimpleNamespace(
        ego_spawn_lane_preference="middle",
        ego_spawn_lane_probabilities=None,
        ego_spawn_distance_to_route_end_m=100.0,
        ego_spawn_reference_block_id=None,
    )
    manager = _manager_with_fixed_spawn(
        monkeypatch,
        lanes=lanes,
        scenario_id="test_middle_lane_preference",
        scenario=scenario,
    )

    assert manager._apply_fixed_route_spawn_configs() is True

    configs = manager.engine.global_config["agent_configs"]
    assert configs["agent0"]["spawn_lane_index"] == ("A", "B", 1)
    assert configs["agent1"]["spawn_lane_index"] == ("A", "B", 1)
    assert configs["agent2"]["spawn_lane_index"] == ("A", "B", 1)


def test_fixed_route_spawn_lane_id_overrides_preferences(monkeypatch):
    lanes = [
        _FakeLane(index=("A", "B", 0), length=200.0),
        _FakeLane(index=("A", "B", 1), length=200.0),
        _FakeLane(index=("A", "B", 2), length=200.0),
    ]
    scenario = SimpleNamespace(
        ego_spawn_lane_id=0,
        ego_spawn_lane_preference="rightmost",
        ego_spawn_lane_probabilities={
            "rightmost": 1.0,
        },
        ego_spawn_distance_to_route_end_m=100.0,
        ego_spawn_reference_block_id=None,
    )
    manager = _manager_with_fixed_spawn(
        monkeypatch,
        lanes=lanes,
        scenario_id="test_lane_id_override",
        scenario=scenario,
    )

    assert manager._apply_fixed_route_spawn_configs() is True

    configs = manager.engine.global_config["agent_configs"]
    assert configs["agent0"]["spawn_lane_index"] == ("A", "B", 0)
    assert configs["agent1"]["spawn_lane_index"] == ("A", "B", 0)
    assert configs["agent2"]["spawn_lane_index"] == ("A", "B", 0)


def test_fixed_route_spawn_lane_id_is_clamped_to_available_lanes(monkeypatch):
    lanes = [
        _FakeLane(index=("A", "B", 0), length=200.0),
        _FakeLane(index=("A", "B", 1), length=200.0),
    ]
    scenario = SimpleNamespace(
        ego_spawn_lane_id=99,
        ego_spawn_lane_preference="leftmost",
        ego_spawn_lane_probabilities=None,
        ego_spawn_distance_to_route_end_m=100.0,
        ego_spawn_reference_block_id=None,
    )
    manager = _manager_with_fixed_spawn(
        monkeypatch,
        lanes=lanes,
        scenario_id="test_lane_id_clamp",
        scenario=scenario,
    )

    assert manager._apply_fixed_route_spawn_configs() is True

    configs = manager.engine.global_config["agent_configs"]
    assert configs["agent0"]["spawn_lane_index"] == ("A", "B", 1)


def test_s6_fixed_route_spawn_can_reference_g1_route_road(monkeypatch):
    c2_road = _FakeRoad("C2_START", "C2_END")
    g1_road = _FakeRoad("G1_START", "G1_END")
    lanes = [
        _FakeLane(index=("G1_START", "G1_END", 0), length=200.0),
        _FakeLane(index=("G1_START", "G1_END", 1), length=200.0),
        _FakeLane(index=("G1_START", "G1_END", 2), length=200.0),
    ]
    scenario = SimpleNamespace(
        ego_spawn_lane_preference="rightmost",
        ego_spawn_lane_probabilities=None,
        ego_spawn_distance_to_route_end_m=50.0,
        ego_spawn_reference_block_id="g1",
    )
    manager = _manager_with_fixed_spawn(
        monkeypatch,
        lanes=lanes,
        scenario_id="test_reference_g1",
        scenario=scenario,
        route_roads=[c2_road, g1_road],
        route_block_ids=("c2", "g1", "c3"),
    )

    assert manager._apply_fixed_route_spawn_configs() is True

    configs = manager.engine.global_config["agent_configs"]
    assert configs["agent0"]["spawn_lane_index"] == ("G1_START", "G1_END", 2)
    assert configs["agent0"]["spawn_longitude"] == pytest.approx(150.0)
    assert configs["agent1"]["spawn_longitude"] == pytest.approx(140.0)
    assert configs["agent2"]["spawn_longitude"] == pytest.approx(130.0)


def test_fixed_route_spawn_uses_scenario_longitude_from_road_start(monkeypatch):
    lanes = [
        _FakeLane(index=("A", "B", 0), length=200.0),
        _FakeLane(index=("A", "B", 1), length=200.0),
    ]
    scenario = SimpleNamespace(
        ego_spawn_lane_preference="rightmost",
        ego_spawn_lane_probabilities=None,
        ego_spawn_longitude_m=42.0,
        ego_spawn_distance_to_route_end_m=100.0,
        ego_spawn_reference_block_id=None,
    )
    manager = _manager_with_fixed_spawn(
        monkeypatch,
        lanes=lanes,
        scenario_id="test_longitude_from_start",
        scenario=scenario,
    )

    assert manager._apply_fixed_route_spawn_configs() is True

    configs = manager.engine.global_config["agent_configs"]
    assert configs["agent0"]["spawn_lane_index"] == ("A", "B", 1)
    assert configs["agent0"]["spawn_longitude"] == pytest.approx(42.0)
    assert configs["agent1"]["spawn_longitude"] == pytest.approx(32.0)
    assert configs["agent2"]["spawn_longitude"] == pytest.approx(22.0)


def test_fixed_route_spawn_samples_scenario_longitude_range(monkeypatch):
    lanes = [
        _FakeLane(index=("A", "B", 0), length=200.0),
        _FakeLane(index=("A", "B", 1), length=200.0),
    ]
    scenario = SimpleNamespace(
        ego_spawn_lane_preference="rightmost",
        ego_spawn_lane_probabilities=None,
        ego_spawn_longitude_m=(25.0, 90.0),
        ego_spawn_distance_to_route_end_m=None,
        ego_spawn_reference_block_id=None,
    )
    manager = _manager_with_fixed_spawn(
        monkeypatch,
        lanes=lanes,
        scenario_id="test_longitude_range",
        scenario=scenario,
    )
    manager.np_random = _FixedUniformRng(72.0)

    assert manager._apply_fixed_route_spawn_configs() is True

    configs = manager.engine.global_config["agent_configs"]
    assert configs["agent0"]["spawn_longitude"] == pytest.approx(72.0)
    assert configs["agent1"]["spawn_longitude"] == pytest.approx(62.0)
    assert configs["agent2"]["spawn_longitude"] == pytest.approx(52.0)


def test_fixed_route_spawn_clips_scenario_longitude_to_tail_buffer(monkeypatch):
    scenario = SimpleNamespace(
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        ego_spawn_longitude_m=5.0,
        ego_spawn_distance_to_route_end_m=None,
        ego_spawn_reference_block_id=None,
    )
    manager = _manager_with_fixed_spawn(
        monkeypatch,
        scenario_id="test_longitude_clip",
        scenario=scenario,
    )

    assert manager._apply_fixed_route_spawn_configs() is True

    configs = manager.engine.global_config["agent_configs"]
    assert configs["agent0"]["spawn_longitude"] == pytest.approx(26.0)
    assert configs["agent1"]["spawn_longitude"] == pytest.approx(16.0)
    assert configs["agent2"]["spawn_longitude"] == pytest.approx(6.0)


def test_fixed_route_spawn_can_reference_block_internal_road(monkeypatch):
    merge_route_road = _FakeRoad("MERGE_ROUTE_START", "MERGE_ROUTE_END")
    main_route_road = _FakeRoad("MAIN_START", "MAIN_END")
    split_route_road = _FakeRoad("SPLIT_START", "SPLIT_END")
    internal_road = _FakeRoad("MERGE_INTERNAL_START", "MERGE_INTERNAL_END")
    internal_lanes = [_FakeLane(index=("MERGE_INTERNAL_START", "MERGE_INTERNAL_END", 0), length=100.0)]
    scenario = SimpleNamespace(
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        ego_spawn_distance_to_route_end_m=None,
        ego_spawn_reference_block_id="merge0",
        ego_spawn_reference_kind="block_internal_road",
        ego_spawn_internal_road_index=3,
    )
    block = _FakeBlock(
        "merge0",
        [
            [_FakeLane(index=("MERGE_ROUTE_START", "MERGE_ROUTE_END", 0), length=20.0)],
            [_FakeLane(index=("SHORT_A", "SHORT_B", 0), length=10.0)],
            [_FakeLane(index=("SHORT_C", "SHORT_D", 0), length=10.0)],
            internal_lanes,
        ],
    )
    manager = _manager_with_fixed_spawn(
        monkeypatch,
        lanes=[_FakeLane(index=("MERGE_ROUTE_START", "MERGE_ROUTE_END", 0), length=20.0)],
        scenario_id="test_s9_internal_spawn",
        scenario=scenario,
        route_roads=[merge_route_road, main_route_road, split_route_road],
        route_block_ids=("merge0", "s_main2", "split0"),
        blocks=[block],
        extra_lanes_by_road={internal_road: internal_lanes},
    )

    assert manager._apply_fixed_route_spawn_configs() is True

    configs = manager.engine.global_config["agent_configs"]
    assert configs["agent0"]["spawn_lane_index"] == ("MERGE_INTERNAL_START", "MERGE_INTERNAL_END", 0)
    assert configs["agent0"]["spawn_longitude"] == pytest.approx(26.0)
    assert configs["agent1"]["spawn_longitude"] == pytest.approx(16.0)
    assert configs["agent2"]["spawn_longitude"] == pytest.approx(6.0)


def test_fixed_route_spawn_can_reference_map_block_outside_local_route(monkeypatch):
    merge_route_road = _FakeRoad("MERGE_ROUTE_START", "MERGE_ROUTE_END")
    main_route_road = _FakeRoad("MAIN_START", "MAIN_END")
    split_route_road = _FakeRoad("SPLIT_START", "SPLIT_END")
    c3_short_road = _FakeRoad("C3_SHORT_START", "C3_SHORT_END")
    c3_long_road = _FakeRoad("C3_LONG_START", "C3_LONG_END")
    c3_short_lanes = [
        _FakeLane(index=("C3_SHORT_START", "C3_SHORT_END", 0), length=20.0),
        _FakeLane(index=("C3_SHORT_START", "C3_SHORT_END", 1), length=20.0),
        _FakeLane(index=("C3_SHORT_START", "C3_SHORT_END", 2), length=20.0),
    ]
    c3_long_lanes = [
        _FakeLane(index=("C3_LONG_START", "C3_LONG_END", 0), length=100.0),
        _FakeLane(index=("C3_LONG_START", "C3_LONG_END", 1), length=100.0),
        _FakeLane(index=("C3_LONG_START", "C3_LONG_END", 2), length=100.0),
    ]
    scenario = SimpleNamespace(
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities={
            "rightmost": 0.4,
            "middle": 0.4,
            "leftmost": 0.2,
        },
        ego_spawn_distance_to_route_end_m=30.0,
        ego_spawn_reference_block_id="c3",
        ego_spawn_reference_kind="block_internal_road",
        ego_spawn_internal_road_index=1,
    )
    block = _FakeBlock("c3", [c3_short_lanes, c3_long_lanes])
    manager = _manager_with_fixed_spawn(
        monkeypatch,
        lanes=[_FakeLane(index=("MERGE_ROUTE_START", "MERGE_ROUTE_END", 0), length=20.0)],
        scenario_id="test_s9_spawn_from_c3",
        scenario=scenario,
        route_roads=[merge_route_road, main_route_road, split_route_road],
        route_block_ids=("merge0", "s_main2", "split0"),
        blocks=[block],
        extra_lanes_by_road={
            c3_short_road: c3_short_lanes,
            c3_long_road: c3_long_lanes,
        },
    )

    assert manager._apply_fixed_route_spawn_configs() is True

    configs = manager.engine.global_config["agent_configs"]
    assert configs["agent0"]["spawn_lane_index"] == ("C3_LONG_START", "C3_LONG_END", 1)
    assert configs["agent0"]["spawn_longitude"] == pytest.approx(70.0)
    assert configs["agent1"]["spawn_longitude"] == pytest.approx(60.0)
    assert configs["agent2"]["spawn_longitude"] == pytest.approx(50.0)
