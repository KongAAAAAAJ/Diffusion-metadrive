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


def _build_runtime_route_env(monkeypatch, *, explicit_initial_speed: bool = False) -> PlatoonEnv:
    env = PlatoonEnv.__new__(PlatoonEnv)
    env._explicit_config_keys = {"initial_speed_km_h"} if explicit_initial_speed else set()
    env.platoon_config = SimpleNamespace(
        scenario_id="S5_hard_brake_lead",
        local_route="R1_entry_straight",
        initial_speed_km_h=31.0 if explicit_initial_speed else 24.0,
    )
    env.config = {
        "scenario_id": "S5_hard_brake_lead",
        "local_route": "R1_entry_straight",
        "ego_main_route_block_ids": ("s0",),
        "route_preset": "mainline",
        "initial_speed_km_h": env.platoon_config.initial_speed_km_h,
    }
    engine = SimpleNamespace(
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
    assert env.config["ego_main_route_block_ids"] == ("c2", "g1", "c3")
    assert env.engine.global_config["ego_main_route_block_ids"] == ("c2", "g1", "c3")
    assert env.config["route_preset"] == "mainline"
    assert env.platoon_config.initial_speed_km_h == pytest.approx(22.0)
    assert env.engine.global_config["initial_speed_km_h"] == pytest.approx(22.0)
    assert env.platoon_config.traffic_density == pytest.approx(0.03)
    assert env.config["traffic_spawn_exclusion_ahead_m"] == pytest.approx(100.0)
    assert env.config["traffic_spawn_exclusion_behind_m"] == pytest.approx(100.0)
    assert env.engine.global_config["traffic_density"] == pytest.approx(0.03)
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
