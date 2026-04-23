from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scenarios.definitions import get_scenario_definition
from scenarios.orchestrator import ScenarioOrchestrator


class _Lane:
    def __init__(self, index):
        self.index = index

    def local_coordinates(self, position):
        return float(position[0]), 0.0


class _Vehicle:
    def __init__(self, lane, position_x=0.0):
        self.lane = lane
        self.position = np.asarray([position_x, 0.0], dtype=np.float32)
        self.name = "ego"
        self.speed_km_h = 20.0
        self.navigation = SimpleNamespace(current_ref_lanes=[lane])


def test_noop_scenario_marks_triggered_and_realized():
    lane = _Lane(("n0", "n1", 0))
    ego = _Vehicle(lane, position_x=100.0)
    env = SimpleNamespace(agents={"default_agent": ego}, engine=SimpleNamespace(current_map=SimpleNamespace(blocks=[])))

    orchestrator = ScenarioOrchestrator(
        get_scenario_definition("S1_free_cruise_straight"),
        "R1_entry_straight",
    )
    orchestrator.reset(env, "default_agent")
    orchestrator._road_to_block_id = {("n0", "n1"): "s0"}

    orchestrator.before_step(env, "default_agent", 12)

    summary = orchestrator.get_episode_summary()
    assert summary["scenario_triggered"] is True
    assert summary["scenario_realized"] is True
    assert summary["scenario_trigger_step"] == 12
    assert "no-op" in summary["scenario_notes"]


def test_hard_brake_lead_installs_speed_profile(monkeypatch):
    lane = _Lane(("n0", "n1", 0))
    ego = _Vehicle(lane, position_x=120.0)
    lead = SimpleNamespace(name="lead_vehicle")
    env = SimpleNamespace(
        agents={"default_agent": ego},
        engine=SimpleNamespace(
            current_map=SimpleNamespace(blocks=[]),
            get_policy=lambda name: SimpleNamespace() if name == "lead_vehicle" else None,
        ),
    )

    orchestrator = ScenarioOrchestrator(
        get_scenario_definition("S5_hard_brake_lead"),
        "R1_entry_straight",
    )
    orchestrator.reset(env, "default_agent")
    orchestrator._road_to_block_id = {("n0", "n1"): "s0"}
    monkeypatch.setattr(orchestrator, "_find_front_vehicle_with_distance", lambda vehicle: (lead, 18.0))

    orchestrator.before_step(env, "default_agent", 7)

    summary = orchestrator.get_episode_summary()
    assert summary["scenario_triggered"] is True
    assert summary["scenario_realized"] is True
    assert orchestrator._speed_profiles["lead_vehicle"]["target_speed_kmh"] == 1.0


def test_hard_brake_lead_spawns_dedicated_lead_when_front_vehicle_is_out_of_range(monkeypatch):
    lane = _Lane(("n0", "n1", 0))
    ego = _Vehicle(lane, position_x=120.0)
    front = SimpleNamespace(name="front_vehicle", position=np.asarray([160.0, 0.0], dtype=np.float32))
    env = SimpleNamespace(
        agents={"default_agent": ego},
        engine=SimpleNamespace(current_map=SimpleNamespace(blocks=[]), get_policy=lambda name: SimpleNamespace()),
    )

    orchestrator = ScenarioOrchestrator(
        get_scenario_definition("S5_hard_brake_lead"),
        "R1_entry_straight",
    )
    orchestrator.reset(env, "default_agent")
    orchestrator._road_to_block_id = {("n0", "n1"): "s0"}

    spawned = SimpleNamespace(name="spawned_lead")
    monkeypatch.setattr(orchestrator, "_find_front_vehicle_with_distance", lambda vehicle: (front, 32.0))
    monkeypatch.setattr(orchestrator, "_spawn_lead_vehicle", lambda *args, **kwargs: spawned)

    orchestrator.before_step(env, "default_agent", 11)

    summary = orchestrator.get_episode_summary()
    assert summary["scenario_realized"] is True
    assert "lead_out_of_range" in summary["scenario_notes"]
    assert "lead_spawned" in summary["scenario_notes"]
    assert orchestrator._lead_vehicle_name == "spawned_lead"


def test_hard_brake_lead_spawns_dedicated_lead_when_existing_front_vehicle_policy_missing(monkeypatch):
    lane = _Lane(("n0", "n1", 0))
    ego = _Vehicle(lane, position_x=120.0)
    front = SimpleNamespace(name="front_vehicle", position=np.asarray([138.0, 0.0], dtype=np.float32))
    env = SimpleNamespace(
        agents={"default_agent": ego},
        engine=SimpleNamespace(current_map=SimpleNamespace(blocks=[]), get_policy=lambda name: None),
    )

    orchestrator = ScenarioOrchestrator(
        get_scenario_definition("S5_hard_brake_lead"),
        "R1_entry_straight",
    )
    orchestrator.reset(env, "default_agent")
    orchestrator._road_to_block_id = {("n0", "n1"): "s0"}

    spawned = SimpleNamespace(name="spawned_lead")
    monkeypatch.setattr(orchestrator, "_find_front_vehicle_with_distance", lambda vehicle: (front, 18.0))
    monkeypatch.setattr(orchestrator, "_spawn_lead_vehicle", lambda *args, **kwargs: spawned)

    orchestrator.before_step(env, "default_agent", 9)

    summary = orchestrator.get_episode_summary()
    assert summary["scenario_realized"] is True
    assert "lead_policy_missing" in summary["scenario_notes"]
    assert "lead_spawned" in summary["scenario_notes"]
    assert "lead_brake_profile" in summary["scenario_notes"]
    assert orchestrator._lead_vehicle_name == "spawned_lead"


def test_apply_speed_profiles_directly_brakes_vehicle_and_marks_warning(monkeypatch):
    lane = _Lane(("n0", "n1", 0))
    ego = _Vehicle(lane, position_x=120.0)
    policy = SimpleNamespace(NORMAL_SPEED=20.0, target_speed=20.0)
    vehicle = SimpleNamespace(
        name="lead_vehicle",
        speed_km_h=18.0,
        set_throttle_brake=lambda value: setattr(vehicle, "throttle_brake", value),
        set_velocity=lambda vec, *args, **kwargs: setattr(vehicle, "velocity", vec),
    )
    env = SimpleNamespace(
        agents={"default_agent": ego},
        engine=SimpleNamespace(get_policy=lambda name: policy, current_map=SimpleNamespace(blocks=[])),
    )

    orchestrator = ScenarioOrchestrator(
        get_scenario_definition("S5_hard_brake_lead"),
        "R1_entry_straight",
    )
    orchestrator.reset(env, "default_agent")
    orchestrator._speed_profiles["lead_vehicle"] = {"remaining_steps": 2.0, "target_speed_kmh": 0.0}
    monkeypatch.setattr(orchestrator, "_find_traffic_vehicle", lambda env, name: vehicle)

    orchestrator._apply_speed_profiles(env)

    assert getattr(vehicle, "throttle_brake", 0.0) < 0.0
    assert getattr(vehicle, "scenario_warning_marker", "") == "!"
    assert "lead_vehicle" in orchestrator._speed_profiles
