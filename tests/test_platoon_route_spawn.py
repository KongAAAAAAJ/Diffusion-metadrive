from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from envs.platoon_env import PlatoonEnv
from metadrive.engine import engine_utils


class _FakeLane:
    def __init__(self, length: float) -> None:
        self.length = float(length)

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return np.asarray([float(longitudinal), float(lateral)], dtype=np.float32)

    def heading_theta_at(self, longitudinal: float) -> float:  # noqa: ARG002
        return 0.0


class _FakeTrafficVehicle:
    def __init__(self, x: float, y: float = 0.0) -> None:
        self.position = np.asarray([float(x), float(y)], dtype=np.float32)


class _RepositionVehicle:
    def __init__(self) -> None:
        self.velocity = None
        self.navigation = None

    def set_position(self, position) -> None:
        self.position = np.asarray(position, dtype=np.float32)

    def set_heading_theta(self, heading: float) -> None:
        self.heading_theta = float(heading)

    def set_velocity(self, velocity, *, in_local_frame: bool) -> None:
        assert in_local_frame is False
        self.velocity = np.asarray(velocity, dtype=np.float64)


class _FixedUniformRng:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def uniform(self, low, high):
        return self.value


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


def test_route_reposition_preserves_fixed_spawn_manager_speed(
    monkeypatch,
) -> None:
    env = PlatoonEnv.__new__(PlatoonEnv)
    env.config = {
        "local_route": "R6_exit_to_ramp",
        "scenario_id": "S8_ego_exit_to_ramp",
        "platoon_fixed_route_spawn": True,
        "platoon_spawn_gap_m": 10.0,
        "platoon_route_spawn_lane_index": 0,
        "initial_speed_km_h": 22.0,
    }
    env._agent_ids = ["agent0", "agent1", "agent2"]
    active_agents = {
        agent_id: _RepositionVehicle() for agent_id in env._agent_ids
    }
    env.agent_manager = SimpleNamespace(active_agents=active_agents)
    lane = _FakeLane(length=130.0)
    road = SimpleNamespace(start_node="A", end_node="B")
    fixed_configs = {
        agent_id: {
            "spawn_lane_index": ("A", "B", 0),
            "spawn_longitude": 60.0 - index * 10.0,
            "spawn_velocity": (26.0 / 3.6, 0.0),
            "spawn_velocity_car_frame": True,
        }
        for index, agent_id in enumerate(env._agent_ids)
    }
    engine = SimpleNamespace(
        agent_manager=SimpleNamespace(active_agents=active_agents),
        spawn_manager=SimpleNamespace(
            get_main_route_spawn_roads=lambda _current_map: [road]
        ),
        current_map=SimpleNamespace(
            road_network=SimpleNamespace(graph={"A": {"B": [lane]}})
        ),
        global_config={"agent_configs": fixed_configs},
    )
    monkeypatch.setattr(engine_utils, "get_engine", lambda: engine)
    env._select_route_spawn_lead_long = lambda _lane, _gap: 60.0

    env._reposition_platoon_on_route()

    assert all(
        np.linalg.norm(vehicle.velocity) * 3.6 == pytest.approx(26.0)
        for vehicle in active_agents.values()
    )


def test_select_route_spawn_lead_moves_away_from_nearby_traffic() -> None:
    env = _build_env(traffic_positions=[20.0, 21.0, 22.0])
    lane = _FakeLane(length=80.0)

    lead_long = env._select_route_spawn_lead_long(lane, gap_m=9.2)

    assert lead_long > 30.0


def _build_runtime_route_env(monkeypatch, *, explicit_initial_speed: bool = False) -> PlatoonEnv:
    env = PlatoonEnv.__new__(PlatoonEnv)
    env._explicit_config_keys = {"initial_speed_km_h"} if explicit_initial_speed else set()
    env.platoon_config = SimpleNamespace(
        scenario_id="S5_hard_brake_lead",
        local_route="R1_entry_straight",
        initial_speed_km_h=31.0 if explicit_initial_speed else 24.0,
        vehicle_length_m=5.74,
    )
    env.config = {
        "scenario_id": "S5_hard_brake_lead",
        "local_route": "R1_entry_straight",
        "ego_main_route_block_ids": ("s0",),
        "route_preset": "mainline",
        "initial_speed_km_h": env.platoon_config.initial_speed_km_h,
    }
    engine = SimpleNamespace(
        traffic_manager=SimpleNamespace(np_random=_FixedUniformRng(24.5)),
        global_config={
            **env.config,
            "agent_configs": {
                "agent0": {"destination": "1S0_0_"},
                "agent1": {"destination": "1S0_0_"},
                "agent2": {"destination": "1S0_0_"},
            },
        }
    )
    monkeypatch.setattr(engine_utils, "get_engine", lambda: engine)
    return env


def test_runtime_scenario_route_replaces_route_and_invalidates_old_destinations(monkeypatch) -> None:
    env = _build_runtime_route_env(monkeypatch)

    env.set_runtime_scenario_route("S6_background_merge_in", "R6_mainline_merge_approach")

    assert env.platoon_config.scenario_id == "S6_background_merge_in"
    assert env.platoon_config.local_route == "R6_mainline_merge_approach"
    assert env.config["ego_main_route_block_ids"] == (
        "c2",
        "g1",
        "c3",
        "merge0",
        "s_main2",
    )
    assert env.engine.global_config["ego_main_route_block_ids"] == (
        "c2",
        "g1",
        "c3",
        "merge0",
        "s_main2",
    )
    assert env.config["route_preset"] == "mainline"
    assert env.platoon_config.initial_speed_km_h == pytest.approx(24.5)
    assert env.engine.global_config["initial_speed_km_h"] == pytest.approx(24.5)
    assert env.platoon_config.traffic_density == pytest.approx(0.0)
    assert env.config["traffic_spawn_exclusion_ahead_m"] == pytest.approx(100.0)
    assert env.config["traffic_spawn_exclusion_behind_m"] == pytest.approx(100.0)
    assert env.engine.global_config["traffic_density"] == pytest.approx(0.0)
    assert env.engine.global_config["traffic_spawn_exclusion_ahead_m"] == pytest.approx(100.0)
    assert env.engine.global_config["traffic_spawn_exclusion_behind_m"] == pytest.approx(100.0)
    assert all(
        agent_config["destination"] is None
        for agent_config in env.engine.global_config["agent_configs"].values()
    )


def test_runtime_scenario_route_preserves_explicit_initial_speed(monkeypatch) -> None:
    env = _build_runtime_route_env(monkeypatch, explicit_initial_speed=True)

    env.set_runtime_scenario_route("S6_background_merge_in", "R6_mainline_merge_approach")

    assert env.platoon_config.initial_speed_km_h == pytest.approx(31.0)
    assert env.config["initial_speed_km_h"] == pytest.approx(31.0)
    assert env.engine.global_config["initial_speed_km_h"] == pytest.approx(31.0)


def test_runtime_scenario_route_rejects_route_not_allowed_by_scenario(monkeypatch) -> None:
    env = _build_runtime_route_env(monkeypatch)

    with pytest.raises(ValueError, match="does not allow local route"):
        env.set_runtime_scenario_route("S6_background_merge_in", "R1_entry_straight")
