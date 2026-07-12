from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scenarios.definitions import RecipeSpec, ScenarioDefinition, TriggerSpec, get_scenario_definition
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
            global_config={"physics_world_step_size": 0.02, "decision_repeat": 5},
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

    assert "lead_vehicle" not in orchestrator._speed_profiles
    assert orchestrator.get_episode_summary()["scenario_triggered"] is False

    orchestrator.before_step(env, "default_agent", 30)

    summary = orchestrator.get_episode_summary()
    assert summary["scenario_triggered"] is True
    assert summary["scenario_realized"] is True
    assert summary["scenario_trigger_step"] == 30
    assert orchestrator._speed_profiles["lead_vehicle"]["target_speed_kmh"] == 1.25
    assert getattr(lead, "scenario_managed_vehicle") is True
    assert getattr(lead, "scenario_warning_marker") == "!"
    assert getattr(lead, "scenario_id") == "S5_hard_brake_lead"
    assert getattr(lead, "scenario_role") == "hard_brake_lead"


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

    orchestrator.before_step(env, "default_agent", 30)

    summary = orchestrator.get_episode_summary()
    assert summary["scenario_realized"] is True
    assert "lead_out_of_range" in summary["scenario_notes"]
    assert "lead_spawned" in summary["scenario_notes"]
    assert orchestrator._lead_vehicle_name == "spawned_lead"
    assert getattr(spawned, "scenario_managed_vehicle") is True
    assert getattr(spawned, "scenario_warning_marker") == "!"
    assert getattr(spawned, "scenario_id") == "S5_hard_brake_lead"
    assert getattr(spawned, "scenario_role") == "hard_brake_lead"


def test_s5_hard_brake_lead_randomizes_lead_spawn_params_from_env_seed(monkeypatch):
    lane = _Lane(("n0", "n1", 0))
    ego = _Vehicle(lane, position_x=120.0)
    ego.speed_km_h = 30.0
    rng = np.random.RandomState(123)
    env = SimpleNamespace(
        agents={"default_agent": ego},
        engine=SimpleNamespace(
            current_map=SimpleNamespace(blocks=[]),
            traffic_manager=SimpleNamespace(np_random=rng),
            get_policy=lambda name: SimpleNamespace(),
        ),
    )

    orchestrator = ScenarioOrchestrator(
        get_scenario_definition("S5_hard_brake_lead"),
        "R1_entry_straight",
    )
    orchestrator.reset(env, "default_agent")
    orchestrator._road_to_block_id = {("n0", "n1"): "s0"}
    monkeypatch.setattr(orchestrator, "_find_front_vehicle_with_distance", lambda vehicle: (None, None))

    captured_params = {}

    def fake_spawn(env_arg, ego_arg, params):
        captured_params.update(params)
        return SimpleNamespace(name="spawned_lead")

    monkeypatch.setattr(orchestrator, "_spawn_lead_vehicle", fake_spawn)

    orchestrator.before_step(env, "default_agent", 30)

    expected_rng = np.random.RandomState(123)
    expected_distance = float(expected_rng.uniform(45.0, 55.0))
    expected_target = float(expected_rng.uniform(19.0, 23.0))
    assert captured_params["lead_distance_m"] == expected_distance
    assert captured_params["lead_target_speed_kmh"] == expected_target
    assert 45.0 <= captured_params["lead_distance_m"] <= 55.0
    assert 19.0 <= captured_params["lead_target_speed_kmh"] <= 23.0


def test_s5_hard_brake_random_ranges_are_declared_in_definition():
    scenario = get_scenario_definition("S5_hard_brake_lead")
    params = scenario.traffic_recipes[0].params

    assert tuple(params["lead_distance_range_m"]) == (45.0, 55.0)
    assert tuple(params["lead_target_speed_range_kmh"]) == (19.0, 23.0)


def test_inject_background_vehicle_randomizes_spawn_and_speed_from_env_seed(monkeypatch):
    lane = _Lane(("n0", "n1", 0))
    ego = _Vehicle(lane, position_x=20.0)
    rng = np.random.RandomState(321)
    env = SimpleNamespace(
        agents={"default_agent": ego},
        engine=SimpleNamespace(
            current_map=SimpleNamespace(blocks=[]),
            traffic_manager=SimpleNamespace(np_random=rng),
        ),
    )
    definition = ScenarioDefinition(
        code="SX",
        scenario_id="SX_random_inject",
        allowed_local_routes=("R_test",),
        trigger_by_local_route={"R_test": TriggerSpec("g1", 0.0, 100.0)},
        traffic_recipes=(
            RecipeSpec(
                "inject_background_vehicle",
                {
                    "reference_kind": "block_socket_road",
                    "block_id": "g1",
                    "socket_index": 1,
                    "spawn_longitude_range": (8.0, 18.0),
                    "target_speed_range_kmh": (20.0, 28.0),
                },
            ),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="test",
        description="test",
    )
    orchestrator = ScenarioOrchestrator(definition, "R_test")
    orchestrator.reset(env, "default_agent")
    orchestrator._road_to_block_id = {("n0", "n1"): "g1"}
    captured = {}

    def fake_spawn_on_reference(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(name="spawned_merge_vehicle")

    monkeypatch.setattr(orchestrator, "_spawn_on_reference", fake_spawn_on_reference)

    orchestrator.before_step(env, "default_agent", 11)

    expected_rng = np.random.RandomState(321)
    assert captured["spawn_longitude"] == float(expected_rng.uniform(8.0, 18.0))
    assert captured["target_speed_kmh"] == float(expected_rng.uniform(20.0, 28.0))


def test_inject_background_vehicle_marks_spawned_vehicle_and_summary(monkeypatch):
    lane = _Lane(("n0", "n1", 0))
    ego = _Vehicle(lane, position_x=20.0)
    env = SimpleNamespace(
        agents={"default_agent": ego},
        engine=SimpleNamespace(current_map=SimpleNamespace(blocks=[])),
    )
    definition = ScenarioDefinition(
        code="SX",
        scenario_id="SX_random_inject",
        allowed_local_routes=("R_test",),
        trigger_by_local_route={"R_test": TriggerSpec("g1", 0.0, 100.0)},
        traffic_recipes=(
            RecipeSpec(
                "inject_background_vehicle",
                {
                    "reference_kind": "block_route_road",
                    "block_id": "g1",
                    "spawn_longitude": 12.0,
                    "target_speed_kmh": 24.0,
                },
            ),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="test",
        description="test",
    )
    orchestrator = ScenarioOrchestrator(definition, "R_test")
    orchestrator.reset(env, "default_agent")
    orchestrator._road_to_block_id = {("n0", "n1"): "g1"}
    spawned = SimpleNamespace(name="spawned_merge_vehicle")
    monkeypatch.setattr(orchestrator, "_spawn_on_reference", lambda *args, **kwargs: spawned)

    orchestrator.before_step(env, "default_agent", 11)

    summary = orchestrator.get_episode_summary()
    assert getattr(spawned, "scenario_managed_vehicle") is True
    assert getattr(spawned, "scenario_warning_marker") == "!"
    assert getattr(spawned, "scenario_id") == "SX_random_inject"
    assert getattr(spawned, "scenario_role") == "injected_background"
    assert summary["scenario_injected_vehicle_names"] == ["spawned_merge_vehicle"]
    assert summary["scenario_realized"] is True


def test_inject_background_vehicle_retries_when_sampled_spawn_is_blocked(monkeypatch):
    lane = _Lane(("n0", "n1", 0))
    ego = _Vehicle(lane, position_x=20.0)
    env = SimpleNamespace(
        agents={"default_agent": ego},
        engine=SimpleNamespace(current_map=SimpleNamespace(blocks=[])),
    )
    definition = ScenarioDefinition(
        code="SX",
        scenario_id="SX_random_inject",
        allowed_local_routes=("R_test",),
        trigger_by_local_route={"R_test": TriggerSpec("g1", 0.0, 100.0)},
        traffic_recipes=(
            RecipeSpec(
                "inject_background_vehicle",
                {
                    "reference_kind": "block_internal_road",
                    "block_id": "g1",
                    "internal_road_index": 1,
                    "spawn_longitude_range": (8.0, 18.0),
                    "target_speed_range_kmh": (22.0, 28.0),
                    "max_spawn_attempts": 2,
                },
            ),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="test",
        description="test",
    )
    orchestrator = ScenarioOrchestrator(definition, "R_test")
    orchestrator.reset(env, "default_agent")
    orchestrator._road_to_block_id = {("n0", "n1"): "g1"}
    spawned = SimpleNamespace(name="spawned_merge_vehicle")
    attempts = []

    def fake_spawn_on_reference(*args, **kwargs):
        attempts.append(kwargs)
        return None if len(attempts) == 1 else spawned

    monkeypatch.setattr(orchestrator, "_spawn_on_reference", fake_spawn_on_reference)

    orchestrator.before_step(env, "default_agent", 11)

    summary = orchestrator.get_episode_summary()
    assert len(attempts) == 2
    assert summary["scenario_realized"] is True
    assert summary["scenario_injected_vehicle_names"] == ["spawned_merge_vehicle"]


def test_ego_adjacent_lane_uses_left_lane_and_ego_relative_longitude(monkeypatch):
    lanes = [
        _Lane(("C3_START", "C3_END", 0)),
        _Lane(("C3_START", "C3_END", 1)),
        _Lane(("C3_START", "C3_END", 2)),
    ]
    for lane in lanes:
        lane.length = 100.0
        lane.position = lambda longitudinal, lateral: np.asarray([longitudinal, lateral], dtype=np.float32)
    road_network = SimpleNamespace(
        graph={"C3_START": {"C3_END": lanes}},
        get_lane=lambda lane_index: lanes[lane_index[2]],
    )
    current_map = SimpleNamespace(blocks=[], road_network=road_network)
    spawned = SimpleNamespace(name="spawned_left_vehicle")
    traffic_manager = SimpleNamespace(
        random_vehicle_type=lambda: "vehicle",
        _spawn_traffic_vehicle_if_safe=lambda vehicle_type, config: spawned,
    )
    ego = _Vehicle(lanes[1], position_x=70.0)
    ego.lane_index = ("C3_START", "C3_END", 1)
    env = SimpleNamespace(
        agents={"default_agent": ego},
        config={},
        engine=SimpleNamespace(current_map=current_map, traffic_manager=traffic_manager, get_policy=lambda name: None),
    )
    definition = ScenarioDefinition(
        code="SX",
        scenario_id="SX_adjacent_inject",
        allowed_local_routes=("R_test",),
        trigger_by_local_route={"R_test": TriggerSpec("c3", 0.0, 100.0)},
        traffic_recipes=tuple(),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="test",
        description="test",
    )
    orchestrator = ScenarioOrchestrator(definition, "R_test")
    captured = {}

    def fake_spawn(vehicle_type, config):
        captured.update(config)
        return spawned

    traffic_manager._spawn_traffic_vehicle_if_safe = fake_spawn

    result = orchestrator._spawn_on_reference(
        env,
        ego,
        reference_kind="ego_adjacent_lane",
        lane_offset=-1,
        spawn_longitude=3.0,
        target_speed_kmh=18.0,
        min_agent_clearance_m=0.0,
    )

    assert result is spawned
    assert captured["spawn_lane_index"] == ("C3_START", "C3_END", 0)
    assert captured["spawn_longitude"] == 73.0


def test_block_socket_road_can_use_ego_relative_longitude(monkeypatch):
    ego_lane = _Lane(("MAIN_START", "MAIN_END", 1))
    ego_lane.length = 120.0
    socket_lane = _Lane(("RAMP_START", "RAMP_END", 0))
    socket_lane.length = 120.0
    socket_lane.position = lambda longitudinal, lateral: np.asarray([longitudinal, lateral], dtype=np.float32)
    road_network = SimpleNamespace(
        graph={"RAMP_START": {"RAMP_END": [socket_lane]}},
        get_lane=lambda lane_index: socket_lane,
    )
    socket_road = SimpleNamespace(start_node="RAMP_START", end_node="RAMP_END")
    block = SimpleNamespace(
        graph_block_id="g1",
        get_socket=lambda socket_index: SimpleNamespace(positive_road=socket_road),
    )
    current_map = SimpleNamespace(blocks=[block], road_network=road_network)
    spawned = SimpleNamespace(name="spawned_ramp_vehicle")
    traffic_manager = SimpleNamespace(random_vehicle_type=lambda: "vehicle")
    ego = _Vehicle(ego_lane, position_x=70.0)
    ego.lane_index = ("MAIN_START", "MAIN_END", 1)
    env = SimpleNamespace(
        agents={"default_agent": ego},
        config={},
        engine=SimpleNamespace(current_map=current_map, traffic_manager=traffic_manager, get_policy=lambda name: None),
    )
    definition = ScenarioDefinition(
        code="SX",
        scenario_id="SX_parallel_socket",
        allowed_local_routes=("R_test",),
        trigger_by_local_route={"R_test": TriggerSpec("g1", 0.0, 100.0)},
        traffic_recipes=tuple(),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="test",
        description="test",
    )
    orchestrator = ScenarioOrchestrator(definition, "R_test")
    captured = {}

    def fake_spawn(vehicle_type, config):
        captured.update(config)
        return spawned

    traffic_manager._spawn_traffic_vehicle_if_safe = fake_spawn

    result = orchestrator._spawn_on_reference(
        env,
        ego,
        reference_kind="block_socket_road",
        block_id="g1",
        socket_index=1,
        spawn_longitude=4.0,
        spawn_longitude_relative_to_ego=True,
        target_speed_kmh=22.0,
        min_agent_clearance_m=0.0,
    )

    assert result is spawned
    assert captured["spawn_lane_index"] == ("RAMP_START", "RAMP_END", 0)
    assert captured["spawn_longitude"] == 74.0


def test_s6_injects_ramp_vehicle_parallel_to_ego_spawn():
    scenario = get_scenario_definition("S6_background_merge_in")
    recipe = scenario.traffic_recipes[0]

    assert recipe.operation == "inject_background_vehicle"
    assert recipe.params["reference_kind"] == "block_route_road"
    assert recipe.params["block_id"] == "h_ramp0"
    assert "socket_index" not in recipe.params
    assert recipe.params["spawn_longitude_relative_to_ego"] is True
    assert tuple(recipe.params["spawn_longitude_range"]) == (-12.0, 12.0)
    assert tuple(recipe.params["target_speed_range_kmh"]) == (20.0, 26.0)
    assert recipe.params["min_agent_clearance_m"] == 2.0
    assert recipe.params["max_spawn_attempts"] == 6


def test_s7_injects_mainline_rightmost_vehicle_parallel_to_ramp_ego():
    scenario = get_scenario_definition("S7_ego_merge_from_ramp")
    trigger = scenario.get_trigger_spec("R7_merge_core")
    recipe = scenario.traffic_recipes[0]

    assert trigger.block_id == "h_ramp0"
    assert (trigger.longitudinal_min, trigger.longitudinal_max) == (0.0, 60.0)
    assert recipe.operation == "inject_background_vehicle"
    assert recipe.params["reference_kind"] == "block_internal_road"
    assert recipe.params["block_id"] == "g1"
    assert recipe.params["internal_road_index"] == 0
    assert recipe.params["lane_index"] == "rightmost"
    assert recipe.params["spawn_longitude_relative_to_ego"] is True
    assert tuple(recipe.params["spawn_longitude_range"]) == (-12.0, 12.0)
    assert tuple(recipe.params["target_speed_range_kmh"]) == (20.0, 26.0)
    assert recipe.params["min_agent_clearance_m"] == 2.0
    assert recipe.params["max_spawn_attempts"] == 6


def test_s9_triggers_on_c3_and_injects_left_adjacent_background_vehicle():
    scenario = get_scenario_definition("S9_narrow_channel_negotiation")
    trigger = scenario.get_trigger_spec("R8_narrow_channel")
    recipe = scenario.traffic_recipes[0]

    assert trigger.block_id == "c3"
    assert (trigger.longitudinal_min, trigger.longitudinal_max) == (0.0, 100.0)
    assert recipe.operation == "inject_background_vehicle"
    assert recipe.params["reference_kind"] == "ego_adjacent_lane"
    assert recipe.params["lane_offset"] == -1
    assert tuple(recipe.params["spawn_longitude_range"]) == (-12.0, 12.0)
    assert tuple(recipe.params["target_speed_range_kmh"]) == (20.0, 26.0)
    assert recipe.params["max_spawn_attempts"] == 6


def test_s8_injects_right_adjacent_background_vehicle_near_ego_spawn():
    scenario = get_scenario_definition("S8_ego_exit_to_ramp")
    trigger = scenario.get_trigger_spec("R6_exit_to_ramp")
    recipe = scenario.traffic_recipes[0]

    assert trigger.block_id == "g0"
    assert (trigger.longitudinal_min, trigger.longitudinal_max) == (0.0, 130.0)
    assert recipe.operation == "inject_background_vehicle"
    assert recipe.params["reference_kind"] == "ego_adjacent_lane"
    assert recipe.params["lane_offset"] == 1
    assert tuple(recipe.params["spawn_longitude_range"]) == (-12.0, 12.0)
    assert tuple(recipe.params["target_speed_range_kmh"]) == (20.0, 26.0)
    assert recipe.params["min_agent_clearance_m"] == 2.0
    assert recipe.params["max_spawn_attempts"] == 6


def test_block_route_road_lane_index_can_select_numeric_lane(monkeypatch):
    lanes = [
        SimpleNamespace(index=("G1_START", "G1_END", 0)),
        SimpleNamespace(index=("G1_START", "G1_END", 1)),
        SimpleNamespace(index=("G1_START", "G1_END", 2)),
    ]
    road = SimpleNamespace(start_node="G1_START", end_node="G1_END")
    block = SimpleNamespace(graph_block_id="g1")
    current_map = SimpleNamespace(
        blocks=[block],
        road_network=SimpleNamespace(graph={"G1_START": {"G1_END": lanes}}),
    )
    env = SimpleNamespace(engine=SimpleNamespace(current_map=current_map))
    definition = ScenarioDefinition(
        code="SX",
        scenario_id="SX_lane_select",
        allowed_local_routes=("R_test",),
        trigger_by_local_route={"R_test": TriggerSpec("g1", 0.0, 100.0)},
        traffic_recipes=tuple(),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="test",
        description="test",
    )
    orchestrator = ScenarioOrchestrator(definition, "R_test")
    monkeypatch.setattr(orchestrator, "_get_first_positive_route_road", lambda block_arg: road)

    lane_tuple = orchestrator._resolve_lane_index(
        env,
        ego_vehicle=None,
        reference_kind="block_route_road",
        block_id="g1",
        lane_index=2,
    )

    assert lane_tuple == ("G1_START", "G1_END", 2)


def test_block_internal_road_can_select_numeric_lane_on_internal_road():
    internal_0 = [
        SimpleNamespace(index=("ENTRY_START", "ENTRY_END", 0)),
        SimpleNamespace(index=("ENTRY_START", "ENTRY_END", 1)),
        SimpleNamespace(index=("ENTRY_START", "ENTRY_END", 2)),
    ]
    internal_1 = [
        SimpleNamespace(index=("MAIN_START", "MAIN_END", 0)),
        SimpleNamespace(index=("MAIN_START", "MAIN_END", 1)),
        SimpleNamespace(index=("MAIN_START", "MAIN_END", 2)),
    ]
    block = SimpleNamespace(
        graph_block_id="g1",
        block_network=SimpleNamespace(get_positive_lanes=lambda: [internal_0, internal_1]),
    )
    current_map = SimpleNamespace(
        blocks=[block],
        road_network=SimpleNamespace(
            graph={
                "ENTRY_START": {"ENTRY_END": internal_0},
                "MAIN_START": {"MAIN_END": internal_1},
            }
        ),
    )
    env = SimpleNamespace(engine=SimpleNamespace(current_map=current_map))
    definition = ScenarioDefinition(
        code="SX",
        scenario_id="SX_lane_select",
        allowed_local_routes=("R_test",),
        trigger_by_local_route={"R_test": TriggerSpec("g1", 0.0, 100.0)},
        traffic_recipes=tuple(),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="test",
        description="test",
    )
    orchestrator = ScenarioOrchestrator(definition, "R_test")

    lane_tuple = orchestrator._resolve_lane_index(
        env,
        ego_vehicle=None,
        reference_kind="block_internal_road",
        block_id="g1",
        internal_road_index=0,
        lane_index=2,
    )

    assert lane_tuple == ("ENTRY_START", "ENTRY_END", 2)


def test_non_s5_hard_brake_lead_keeps_static_recipe_params(monkeypatch):
    lane = _Lane(("n0", "n1", 0))
    ego = _Vehicle(lane, position_x=80.0)
    ego.speed_km_h = 30.0
    env = SimpleNamespace(
        agents={"default_agent": ego},
        engine=SimpleNamespace(
            current_map=SimpleNamespace(blocks=[]),
            traffic_manager=SimpleNamespace(np_random=np.random.RandomState(123)),
            get_policy=lambda name: SimpleNamespace(),
        ),
    )
    definition = ScenarioDefinition(
        code="SX",
        scenario_id="SX_static_hard_brake",
        allowed_local_routes=("R_test",),
        trigger_by_local_route={"R_test": TriggerSpec("s0", 0.0, 100.0)},
        traffic_recipes=(
            RecipeSpec(
                "hard_brake_lead",
                {
                    "lead_distance_m": 10.0,
                    "lead_target_speed_kmh": 21.0,
                    "front_distance_min_m": 10.0,
                    "front_distance_max_m": 24.0,
                    "brake_target_speed_kmh": 1.0,
                    "brake_duration_steps": 200,
                },
            ),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="test",
        description="test",
    )
    orchestrator = ScenarioOrchestrator(definition, "R_test")
    orchestrator.reset(env, "default_agent")
    orchestrator._road_to_block_id = {("n0", "n1"): "s0"}
    monkeypatch.setattr(orchestrator, "_find_front_vehicle_with_distance", lambda vehicle: (None, None))

    captured_params = {}

    def fake_spawn(env_arg, ego_arg, params):
        captured_params.update(params)
        return SimpleNamespace(name="spawned_lead")

    monkeypatch.setattr(orchestrator, "_spawn_lead_vehicle", fake_spawn)

    orchestrator.before_step(env, "default_agent", 11)

    assert captured_params["lead_distance_m"] == 10.0
    assert captured_params["lead_target_speed_kmh"] == 21.0


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

    orchestrator.before_step(env, "default_agent", 30)

    summary = orchestrator.get_episode_summary()
    assert summary["scenario_realized"] is True
    assert "lead_policy_missing" in summary["scenario_notes"]
    assert "lead_spawned" in summary["scenario_notes"]
    assert "lead_brake_profile" in summary["scenario_notes"]
    assert orchestrator._lead_vehicle_name == "spawned_lead"
    assert getattr(spawned, "scenario_managed_vehicle") is True
    assert getattr(spawned, "scenario_warning_marker") == "!"
    assert getattr(spawned, "scenario_id") == "S5_hard_brake_lead"
    assert getattr(spawned, "scenario_role") == "hard_brake_lead"


def test_s5_adjacent_start_recipe_does_not_mark_main_trigger(monkeypatch):
    lane = _Lane(("n0", "n1", 1))
    ego = _Vehicle(lane, position_x=120.0)
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
    monkeypatch.setattr(orchestrator, "_find_front_vehicle_with_distance", lambda vehicle: (None, None))
    monkeypatch.setattr(orchestrator, "_resolve_adjacent_lane_index", lambda *args, **kwargs: ("n0", "n1", 0))
    monkeypatch.setattr(orchestrator, "_spawn_on_lane_tuple", lambda *args, **kwargs: SimpleNamespace(name="side_vehicle"))

    orchestrator.before_step(env, "default_agent", 1)

    summary = orchestrator.get_episode_summary()
    assert summary["scenario_triggered"] is False
    assert "0:hard_brake_lead" not in orchestrator._completed_recipe_keys
    assert "1:inject_adjacent_lane_vehicles" in orchestrator._completed_recipe_keys


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


def test_cut_in_adjacent_lead_spawns_then_moves_into_nearest_platoon_gap(monkeypatch):
    class Lane(_Lane):
        length = 200.0

        def position(self, longitudinal, lateral):
            return np.asarray([float(longitudinal), float(self.index[2]) * 4.0 + float(lateral)], dtype=np.float32)

        def heading_theta_at(self, longitudinal):
            return 0.0

    lanes = [Lane(("A", "B", 0)), Lane(("A", "B", 1))]
    road_network = SimpleNamespace(
        graph={"A": {"B": lanes}},
        get_lane=lambda lane_index: lanes[lane_index[2]],
    )
    class ReadOnlyLaneIndexVehicle:
        name = "cut_in_vehicle"

        def __init__(self):
            self.position = np.asarray([0.0, 0.0], dtype=np.float32)

        @property
        def lane_index(self):
            return ("A", "B", 0)

        def set_position(self, pos):
            self.position = np.asarray(pos, dtype=np.float32)

        def set_heading_theta(self, heading):
            self.heading_theta = float(heading)

        def set_velocity(self, vec, *args, **kwargs):
            self.velocity = vec

    spawned = ReadOnlyLaneIndexVehicle()
    traffic_manager = SimpleNamespace(random_vehicle_type=lambda: "vehicle", _traffic_vehicles=[spawned])
    current_map = SimpleNamespace(blocks=[], road_network=road_network)
    env = SimpleNamespace(
        agents={},
        config={},
        engine=SimpleNamespace(current_map=current_map, traffic_manager=traffic_manager, get_policy=lambda name: None),
    )
    ego = _Vehicle(lanes[1], position_x=100.0)
    ego.name = "agent0"
    ego.lane_index = ("A", "B", 1)
    ego.speed_km_h = 28.0
    env.agents["agent0"] = ego
    agent1 = _Vehicle(lanes[1], position_x=80.0)
    agent1.name = "agent1"
    agent1.lane_index = ("A", "B", 1)
    env.agents["agent1"] = agent1
    agent2 = _Vehicle(lanes[1], position_x=60.0)
    agent2.name = "agent2"
    agent2.lane_index = ("A", "B", 1)
    env.agents["agent2"] = agent2
    definition = ScenarioDefinition(
        code="SX",
        scenario_id="SX_cut_in",
        allowed_local_routes=("R_test",),
        trigger_by_local_route={"R_test": TriggerSpec("s0", 0.0, 120.0)},
        traffic_recipes=(
            RecipeSpec(
                "cut_in_adjacent_lead",
                {
                    "reference_kind": "ego_adjacent_lane",
                    "lane_offset": -1,
                    "spawn_longitude": 0.0,
                    "speed_delta_range_kmh": (3.0, 3.0),
                    "cut_in_delay_steps": 1,
                    "cut_in_distance_m": 10.0,
                    "min_agent_clearance_m": 0.0,
                },
            ),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="test",
        description="test",
    )
    orchestrator = ScenarioOrchestrator(definition, "R_test")
    orchestrator.reset(env, "agent0")
    orchestrator._road_to_block_id = {("A", "B"): "s0"}

    captured_spawn = {}

    def fake_spawn(vehicle_type, config):
        captured_spawn.update(config)
        spawn_lane = road_network.get_lane(config["spawn_lane_index"])
        spawned.position = spawn_lane.position(config["spawn_longitude"], 0.0)
        return spawned

    traffic_manager._spawn_traffic_vehicle_if_safe = fake_spawn

    orchestrator.before_step(env, "agent0", 4)
    summary_after_spawn = orchestrator.get_episode_summary()

    assert captured_spawn["spawn_lane_index"] == ("A", "B", 0)
    assert captured_spawn["spawn_longitude"] == 100.0
    assert summary_after_spawn["scenario_triggered"] is True
    assert summary_after_spawn["scenario_realized"] is False
    assert summary_after_spawn["scenario_injected_vehicle_names"] == ["cut_in_vehicle"]
    assert getattr(spawned, "scenario_warning_marker") == "!"
    assert getattr(spawned, "scenario_role") == "cut_in_adjacent_lead"

    orchestrator.before_step(env, "agent0", 5)
    summary_after_cut_in = orchestrator.get_episode_summary()

    assert spawned.position.tolist() == [90.0, 4.0]
    assert getattr(spawned, "heading_theta") == 0.0
    assert spawned.velocity == [25.0 / 3.6, 0.0]
    assert summary_after_cut_in["scenario_realized"] is True
    assert summary_after_cut_in["scenario_realized_step"] == 5
    assert "cut_in_spawned" in summary_after_cut_in["scenario_notes"]
    assert "cut_in_executed" in summary_after_cut_in["scenario_notes"]


def test_cut_in_adjacent_lead_does_not_fallback_to_agent0_front_without_a_gap(monkeypatch):
    class Lane(_Lane):
        length = 200.0

        def position(self, longitudinal, lateral):
            return np.asarray([float(longitudinal), float(self.index[2]) * 4.0 + float(lateral)], dtype=np.float32)

        def heading_theta_at(self, longitudinal):
            return 0.0

    lanes = [Lane(("A", "B", 0)), Lane(("A", "B", 1))]
    road_network = SimpleNamespace(
        graph={"A": {"B": lanes}},
        get_lane=lambda lane_index: lanes[lane_index[2]],
    )

    class VehicleWithReadOnlyLaneIndex:
        name = "cut_in_vehicle"

        def __init__(self):
            self.position = np.asarray([100.0, 0.0], dtype=np.float32)

        @property
        def lane_index(self):
            return ("A", "B", 0)

        def set_position(self, pos):
            self.position = np.asarray(pos, dtype=np.float32)

        def set_heading_theta(self, heading):
            self.heading_theta = float(heading)

        def set_velocity(self, vec, *args, **kwargs):
            self.velocity = vec

    spawned = VehicleWithReadOnlyLaneIndex()
    traffic_manager = SimpleNamespace(random_vehicle_type=lambda: "vehicle", _traffic_vehicles=[spawned])
    current_map = SimpleNamespace(blocks=[], road_network=road_network)
    env = SimpleNamespace(
        agents={},
        config={},
        engine=SimpleNamespace(current_map=current_map, traffic_manager=traffic_manager, get_policy=lambda name: None),
    )
    ego = _Vehicle(lanes[1], position_x=100.0)
    ego.name = "agent0"
    ego.lane_index = ("A", "B", 1)
    ego.speed_km_h = 28.0
    env.agents["agent0"] = ego
    definition = ScenarioDefinition(
        code="SX",
        scenario_id="SX_cut_in",
        allowed_local_routes=("R_test",),
        trigger_by_local_route={"R_test": TriggerSpec("s0", 0.0, 120.0)},
        traffic_recipes=(
            RecipeSpec(
                "cut_in_adjacent_lead",
                {
                    "reference_kind": "ego_adjacent_lane",
                    "lane_offset": -1,
                    "spawn_longitude": 0.0,
                    "speed_delta_range_kmh": (3.0, 3.0),
                    "cut_in_delay_steps": 1,
                    "cut_in_distance_m": 10.0,
                    "min_agent_clearance_m": 0.0,
                },
            ),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="test",
        description="test",
    )
    orchestrator = ScenarioOrchestrator(definition, "R_test")
    orchestrator.reset(env, "agent0")
    orchestrator._road_to_block_id = {("A", "B"): "s0"}

    def fake_spawn(vehicle_type, config):
        spawn_lane = road_network.get_lane(config["spawn_lane_index"])
        spawned.position = spawn_lane.position(config["spawn_longitude"], 0.0)
        return spawned

    traffic_manager._spawn_traffic_vehicle_if_safe = fake_spawn

    orchestrator.before_step(env, "agent0", 4)
    orchestrator.before_step(env, "agent0", 5)
    summary = orchestrator.get_episode_summary()

    assert spawned.position.tolist() == [100.0, 0.0]
    assert summary["scenario_realized"] is False
    assert "cut_in_gap_unavailable" in summary["scenario_notes"]
