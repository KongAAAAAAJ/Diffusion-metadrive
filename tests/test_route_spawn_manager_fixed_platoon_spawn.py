from __future__ import annotations

from types import SimpleNamespace

import pytest

from metadrive.envs.diffusion_envs.route_spawn_manager import RouteAwareSpawnManager


class _FakeRoad:
    start_node = "A"
    end_node = "B"

    def lane_index(self, lane_idx: int):
        return (self.start_node, self.end_node, lane_idx)


class _FakeLane:
    def __init__(self, index=("A", "B", 0), length=80.0):
        self.index = index
        self.length = length

    def heading_theta_at(self, longitude):
        return 0.0


class _FakeRoadNetwork:
    def __init__(self, lane):
        self.graph = {"A": {"B": [lane]}}


def _manager_with_fixed_spawn(monkeypatch):
    lane = _FakeLane()
    manager = object.__new__(RouteAwareSpawnManager)
    manager._test_engine = SimpleNamespace(
        current_map=SimpleNamespace(road_network=_FakeRoadNetwork(lane)),
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
    monkeypatch.setattr(RouteAwareSpawnManager, "engine", property(lambda self: self._test_engine))
    manager.get_main_route_spawn_roads = lambda current_map: [_FakeRoad()]
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
