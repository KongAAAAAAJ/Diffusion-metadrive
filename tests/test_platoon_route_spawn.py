from __future__ import annotations

import numpy as np

from envs.platoon_env import PlatoonEnv


class _FakeLane:
    def __init__(self, length: float) -> None:
        self.length = float(length)

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return np.asarray([float(longitudinal), float(lateral)], dtype=np.float32)


class _FakeTrafficVehicle:
    def __init__(self, x: float, y: float = 0.0) -> None:
        self.position = np.asarray([float(x), float(y)], dtype=np.float32)


class _FakeEnv:
    _route_spawn_min_tail_buffer_m = PlatoonEnv._route_spawn_min_tail_buffer_m
    _route_spawn_min_front_buffer_m = PlatoonEnv._route_spawn_min_front_buffer_m
    _route_spawn_reference_lead_long_m = staticmethod(PlatoonEnv._route_spawn_reference_lead_long_m)
    _route_spawn_lead_long_bounds = PlatoonEnv._route_spawn_lead_long_bounds
    _route_spawn_vehicle_longitude = PlatoonEnv._route_spawn_vehicle_longitude
    _route_spawn_clearance_score = PlatoonEnv._route_spawn_clearance_score
    _select_route_spawn_lead_long = PlatoonEnv._select_route_spawn_lead_long

    def __init__(self, num_agents: int = 3, traffic_positions: list[float] | None = None) -> None:
        self._agent_ids = [f"agent{i}" for i in range(num_agents)]
        vehicles = [_FakeTrafficVehicle(x) for x in (traffic_positions or [])]
        self.engine = type(
            "Engine",
            (),
            {"traffic_manager": type("TrafficManager", (), {"_traffic_vehicles": vehicles})()},
        )()
        self.config = {}

    def _cfg_float(self, key: str, default: float) -> float:
        return float(self.config.get(key, default))

    def _cfg_bool(self, key: str, default: bool = False) -> bool:
        return bool(self.config.get(key, default))


def _build_env(num_agents: int = 3, traffic_positions: list[float] | None = None) -> _FakeEnv:
    return _FakeEnv(num_agents=num_agents, traffic_positions=traffic_positions)


def test_route_spawn_bounds_keep_tail_away_from_lane_start() -> None:
    env = _build_env()

    min_lead, max_lead = env._route_spawn_lead_long_bounds(lane_length=80.0, gap_m=9.2)

    assert round(min_lead, 1) == 24.4
    assert round(max_lead, 1) == 72.0


def test_select_route_spawn_lead_moves_away_from_nearby_traffic() -> None:
    env = _build_env(traffic_positions=[20.0, 21.0, 22.0])
    lane = _FakeLane(length=80.0)

    lead_long = env._select_route_spawn_lead_long(lane, gap_m=9.2)

    assert lead_long > 30.0
