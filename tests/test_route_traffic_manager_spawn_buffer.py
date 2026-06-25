from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import envs.diffusion_envs.route_traffic_manager as route_traffic_manager
from envs.diffusion_envs.route_traffic_manager import (
    ROUTE_TRAFFIC_SPAWN_LONGITUDE_BUFFER,
    RouteAwareTrafficManager,
)


@dataclass
class _FakeLane:
    length: float = 35.0
    index: tuple[str, str, int] = ("A", "B", 0)

    def position(self, longitude, lateral=0.0):
        return (float(longitude), float(lateral))

    def heading_theta_at(self, longitude):
        return 0.0


class _FakeRoadNetwork:
    def __init__(self, lane):
        self.lane = lane

    def get_lane(self, lane_index):
        assert tuple(lane_index) == tuple(self.lane.index)
        return self.lane


class _FakeMap:
    def __init__(self, lane):
        self.road_network = _FakeRoadNetwork(lane)


class _FakeRandom:
    @staticmethod
    def shuffle(values):
        return None

    @staticmethod
    def randint(low, high=None):
        return int(low)


def _manager_with_engine(monkeypatch, lane: _FakeLane):
    manager = object.__new__(RouteAwareTrafficManager)
    manager._test_engine = SimpleNamespace(
        current_map=_FakeMap(lane),
        global_config={
            "traffic_vehicle_config": {},
            "vehicle_config": {},
            "traffic_spawn_lane_relaxation": True,
            "traffic_spawn_min_gap_ahead": 12.0,
            "traffic_spawn_min_gap_behind": 8.0,
        },
        object_manager=SimpleNamespace(accident_lanes=[]),
    )
    monkeypatch.setattr(RouteAwareTrafficManager, "engine", property(lambda self: self._test_engine))
    manager._traffic_vehicles = []
    manager.block_triggered_vehicles = []
    manager.np_random = _FakeRandom()
    manager.respawn_lanes = [lane]
    manager.mode = "respawn"
    manager.VEHICLE_GAP = 10.0
    return manager


def test_route_aware_traffic_spawn_proposals_skip_lane_start_boundary():
    manager = object.__new__(RouteAwareTrafficManager)

    configs = manager._propose_vehicle_configs(_FakeLane())

    assert configs
    assert configs[0]["spawn_longitude"] == ROUTE_TRAFFIC_SPAWN_LONGITUDE_BUFFER
    assert all(config["spawn_longitude"] > 0.0 for config in configs)
    assert all(config["spawn_longitude"] < _FakeLane.length for config in configs)


def test_route_aware_traffic_spawn_proposals_are_empty_when_lane_too_short_for_buffer():
    manager = object.__new__(RouteAwareTrafficManager)

    configs = manager._propose_vehicle_configs(_FakeLane(length=ROUTE_TRAFFIC_SPAWN_LONGITUDE_BUFFER))

    assert configs == []


def test_final_spawn_guard_rejects_candidates_not_found_by_ray_localization(monkeypatch):
    lane = _FakeLane()
    manager = _manager_with_engine(monkeypatch, lane)
    monkeypatch.setattr(route_traffic_manager, "ray_localization", lambda *args, **kwargs: [])

    assert not manager._passes_final_spawn_guard(
        {"spawn_lane_index": lane.index, "spawn_longitude": ROUTE_TRAFFIC_SPAWN_LONGITUDE_BUFFER}
    )


def test_spawn_if_safe_does_not_call_spawn_object_for_unnavigable_candidate(monkeypatch):
    lane = _FakeLane()
    manager = _manager_with_engine(monkeypatch, lane)
    monkeypatch.setattr(route_traffic_manager, "ray_localization", lambda *args, **kwargs: [])

    def _fail_spawn(*args, **kwargs):
        raise AssertionError("spawn_object should not be called for unnavigable traffic candidate")

    manager.spawn_object = _fail_spawn

    result = manager._spawn_traffic_vehicle_if_safe(
        object,
        {"spawn_lane_index": lane.index, "spawn_longitude": ROUTE_TRAFFIC_SPAWN_LONGITUDE_BUFFER},
    )

    assert result is None


def test_create_basic_vehicles_uses_buffered_spawn_proposals(monkeypatch):
    lane = _FakeLane()
    manager = _manager_with_engine(monkeypatch, lane)
    seen_longitudes = []
    monkeypatch.setattr(route_traffic_manager, "ray_localization", lambda *args, **kwargs: [(lane, lane.index, 0.0)])
    manager.random_vehicle_type = lambda: object

    def _record_spawn(vehicle_type, traffic_v_config, *args, **kwargs):
        seen_longitudes.append(float(traffic_v_config["spawn_longitude"]))
        return None

    manager._spawn_traffic_vehicle_if_safe = _record_spawn

    manager._create_basic_vehicles(None, traffic_density=1.0)

    assert seen_longitudes
    assert all(longitude >= ROUTE_TRAFFIC_SPAWN_LONGITUDE_BUFFER for longitude in seen_longitudes)
    assert 0.0 not in seen_longitudes


def test_after_step_respawn_uses_buffered_spawn_proposals(monkeypatch):
    lane = _FakeLane()
    manager = _manager_with_engine(monkeypatch, lane)
    removed_vehicle = SimpleNamespace(
        id="traffic-1",
        on_lane=False,
        after_step=lambda: None,
    )
    manager._traffic_vehicles = [removed_vehicle]
    seen_configs = []
    monkeypatch.setattr(route_traffic_manager, "ray_localization", lambda *args, **kwargs: [(lane, lane.index, 0.0)])
    manager.clear_objects = lambda ids: None
    manager._get_all_route_lanes = lambda: [lane]
    manager._spawn_traffic_vehicle_if_safe = lambda vehicle_type, traffic_v_config: seen_configs.append(traffic_v_config)

    manager.after_step()

    assert seen_configs
    assert all(float(config["spawn_longitude"]) >= ROUTE_TRAFFIC_SPAWN_LONGITUDE_BUFFER for config in seen_configs)
    assert all(float(config["spawn_longitude"]) != 0.0 for config in seen_configs)
