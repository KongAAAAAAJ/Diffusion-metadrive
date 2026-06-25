from __future__ import annotations

from types import SimpleNamespace

from scenarios.definitions import RecipeSpec, ScenarioDefinition, TriggerSpec
from scenarios.orchestrator import ScenarioOrchestrator


class FakeLane:
    def __init__(self, lane_index: int, length: float = 200.0) -> None:
        self.index = ("road_a", "road_b", lane_index)
        self.length = length

    def local_coordinates(self, position):
        return float(position[0]), 0.0

    def position(self, longitudinal: float, lateral: float):
        return (float(longitudinal), float(self.index[2]) * 4.0 + float(lateral))


class FakeRoadNetwork:
    def __init__(self, lanes) -> None:
        self._lanes = {lane.index: lane for lane in lanes}
        self.graph = {"road_a": {"road_b": list(lanes)}}

    def get_lane(self, lane_index):
        return self._lanes[tuple(lane_index)]


class FakeTrafficManager:
    def __init__(self) -> None:
        self._traffic_vehicles = []
        self.spawn_calls = []

    def random_vehicle_type(self):
        return "vehicle"

    def _spawn_traffic_vehicle_if_safe(self, vehicle_type, config):
        name = f"traffic_{len(self._traffic_vehicles)}"
        vehicle = SimpleNamespace(name=name, spawn_config=config)
        self._traffic_vehicles.append(vehicle)
        self.spawn_calls.append((vehicle_type, config))
        return vehicle


def make_env_and_ego(lane_count: int = 3, ego_lane_index: int = 1):
    lanes = [FakeLane(index) for index in range(lane_count)]
    road_network = FakeRoadNetwork(lanes)
    block = SimpleNamespace(
        graph_block_id="s0",
        block_network=SimpleNamespace(get_positive_lanes=lambda: [lanes]),
        get_socket_list=lambda: [],
        get_respawn_roads=lambda: [],
    )
    traffic_manager = FakeTrafficManager()
    current_map = SimpleNamespace(road_network=road_network, blocks=[block])
    engine = SimpleNamespace(
        current_map=current_map,
        traffic_manager=traffic_manager,
        get_policy=lambda _name: None,
    )
    ego_lane = lanes[ego_lane_index]
    ego = SimpleNamespace(
        name="agent0",
        lane=ego_lane,
        lane_index=ego_lane.index,
        position=(90.0, float(ego_lane_index) * 4.0),
        speed_km_h=22.0,
    )
    env = SimpleNamespace(
        agents={"agent0": ego},
        engine=engine,
        config={"scenario_spawn_min_agent_clearance_m": 0.0},
    )
    return env, ego, traffic_manager


def make_adjacent_definition(recipe_params: dict | None = None) -> ScenarioDefinition:
    return ScenarioDefinition(
        code="T",
        scenario_id="test_adjacent",
        allowed_local_routes=("R1_entry_straight",),
        trigger_by_local_route={"R1_entry_straight": TriggerSpec("s0", 80.0, 120.0)},
        traffic_recipes=(
            RecipeSpec(
                "inject_adjacent_lane_vehicles",
                recipe_params
                or {
                    "vehicles": (
                        {"name": "left_side", "lane_side": "left", "spawn_longitude_offset_m": 12.0, "target_speed_kmh": 24.0},
                        {"name": "right_side", "lane_side": "right", "spawn_longitude_offset_m": 12.0, "target_speed_kmh": 24.0},
                    )
                },
            ),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="test",
        description="test",
    )


def test_adjacent_lane_recipe_spawns_left_and_right_once() -> None:
    env, _ego, traffic_manager = make_env_and_ego()
    orchestrator = ScenarioOrchestrator(make_adjacent_definition(), "R1_entry_straight")
    orchestrator.reset(env, "agent0")

    orchestrator.before_step(env, "agent0", 5)
    orchestrator.before_step(env, "agent0", 6)

    spawned_lanes = [call[1]["spawn_lane_index"] for call in traffic_manager.spawn_calls]
    spawned_longs = [call[1]["spawn_longitude"] for call in traffic_manager.spawn_calls]

    assert spawned_lanes == [("road_a", "road_b", 0), ("road_a", "road_b", 2)]
    assert spawned_longs == [102.0, 102.0]
    assert orchestrator.summary.scenario_triggered is True
    assert orchestrator.summary.scenario_realized is True
    assert orchestrator.summary.notes.count("adjacent_spawned:left_side") == 1
    assert orchestrator.summary.notes.count("adjacent_spawned:right_side") == 1


def test_adjacent_lane_recipe_skips_missing_side_lane() -> None:
    env, _ego, traffic_manager = make_env_and_ego(lane_count=2, ego_lane_index=0)
    orchestrator = ScenarioOrchestrator(make_adjacent_definition(), "R1_entry_straight")
    orchestrator.reset(env, "agent0")

    orchestrator.before_step(env, "agent0", 5)

    spawned_lanes = [call[1]["spawn_lane_index"] for call in traffic_manager.spawn_calls]

    assert spawned_lanes == [("road_a", "road_b", 1)]
    assert "adjacent_lane_missing:left_side" in orchestrator.summary.notes
    assert "adjacent_spawned:right_side" in orchestrator.summary.notes


def test_adjacent_lane_recipe_can_use_recipe_level_trigger() -> None:
    env, ego, traffic_manager = make_env_and_ego()
    definition = make_adjacent_definition(
        {
            "trigger_by_local_route": {
                "R1_entry_straight": {
                    "block_id": "s0",
                    "longitudinal_min": 95.0,
                    "longitudinal_max": 120.0,
                }
            },
            "vehicles": (
                {"name": "left_side", "lane_side": "left", "spawn_longitude_offset_m": 12.0, "target_speed_kmh": 24.0},
            ),
        }
    )
    orchestrator = ScenarioOrchestrator(definition, "R1_entry_straight")
    orchestrator.reset(env, "agent0")

    orchestrator.before_step(env, "agent0", 5)
    ego.position = (100.0, ego.position[1])
    orchestrator.before_step(env, "agent0", 6)

    assert [call[1]["spawn_lane_index"] for call in traffic_manager.spawn_calls] == [("road_a", "road_b", 0)]
    assert [call[1]["spawn_longitude"] for call in traffic_manager.spawn_calls] == [112.0]


def test_adjacent_lane_recipe_can_trigger_on_episode_start_before_scenario_window() -> None:
    env, ego, traffic_manager = make_env_and_ego()
    ego.position = (20.0, ego.position[1])
    definition = make_adjacent_definition(
        {
            "trigger_on_start": True,
            "vehicles": (
                {"name": "left_side", "lane_side": "left", "spawn_longitude_offset_m": 12.0, "target_speed_kmh": 24.0},
                {"name": "right_side", "lane_side": "right", "spawn_longitude_offset_m": 6.0, "target_speed_kmh": 24.0},
            ),
        }
    )
    orchestrator = ScenarioOrchestrator(definition, "R1_entry_straight")
    orchestrator.reset(env, "agent0")

    orchestrator.before_step(env, "agent0", 1)
    orchestrator.before_step(env, "agent0", 2)

    assert [call[1]["spawn_lane_index"] for call in traffic_manager.spawn_calls] == [
        ("road_a", "road_b", 0),
        ("road_a", "road_b", 2),
    ]
    assert [call[1]["spawn_longitude"] for call in traffic_manager.spawn_calls] == [32.0, 26.0]
    assert orchestrator.summary.scenario_triggered is True
    assert orchestrator.summary.trigger_step == 1
    assert orchestrator.summary.notes.count("adjacent_spawned:left_side") == 1
    assert orchestrator.summary.notes.count("adjacent_spawned:right_side") == 1


def test_adjacent_lane_recipe_same_lane_clearance_ignores_other_lanes() -> None:
    env, ego, traffic_manager = make_env_and_ego()
    env.config["scenario_spawn_min_agent_clearance_m"] = 10.0
    ego.position = (20.0, ego.position[1])
    definition = make_adjacent_definition(
        {
            "trigger_on_start": True,
            "clearance_scope": "same_lane",
            "vehicles": (
                {
                    "name": "right_side",
                    "lane_side": "right",
                    "spawn_longitude_offset_m": 6.0,
                    "target_speed_kmh": 24.0,
                },
            ),
        }
    )
    orchestrator = ScenarioOrchestrator(definition, "R1_entry_straight")
    orchestrator.reset(env, "agent0")

    orchestrator.before_step(env, "agent0", 1)

    assert [call[1]["spawn_lane_index"] for call in traffic_manager.spawn_calls] == [("road_a", "road_b", 2)]
    assert [call[1]["spawn_longitude"] for call in traffic_manager.spawn_calls] == [26.0]
    assert "adjacent_spawned:right_side" in orchestrator.summary.notes


def test_adjacent_lane_recipe_same_lane_clearance_blocks_same_lane_agent() -> None:
    env, ego, traffic_manager = make_env_and_ego()
    env.config["scenario_spawn_min_agent_clearance_m"] = 10.0
    ego.position = (20.0, ego.position[1])
    same_lane_agent = SimpleNamespace(
        name="agent_side",
        lane_index=("road_a", "road_b", 2),
        position=(26.0, 8.0),
    )
    env.agents["agent_side"] = same_lane_agent
    definition = make_adjacent_definition(
        {
            "trigger_on_start": True,
            "clearance_scope": "same_lane",
            "vehicles": (
                {
                    "name": "right_side",
                    "lane_side": "right",
                    "spawn_longitude_offset_m": 6.0,
                    "target_speed_kmh": 24.0,
                },
            ),
        }
    )
    orchestrator = ScenarioOrchestrator(definition, "R1_entry_straight")
    orchestrator.reset(env, "agent0")

    orchestrator.before_step(env, "agent0", 1)

    assert traffic_manager.spawn_calls == []
    assert "spawn_blocked:agent_clearance" in orchestrator.summary.notes
    assert "adjacent_spawn_failed:right_side" in orchestrator.summary.notes


def test_injected_background_vehicle_gets_topdown_marker() -> None:
    env, _ego, traffic_manager = make_env_and_ego()
    definition = ScenarioDefinition(
        code="T",
        scenario_id="test_injected_background",
        allowed_local_routes=("R1_entry_straight",),
        trigger_by_local_route={"R1_entry_straight": TriggerSpec("s0", 80.0, 120.0)},
        traffic_recipes=(
            RecipeSpec(
                "inject_background_vehicle",
                {
                    "reference_kind": "ego_lane",
                    "spawn_longitude": 15.0,
                    "target_speed_kmh": 18.0,
                },
            ),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="test",
        description="test",
    )
    orchestrator = ScenarioOrchestrator(definition, "R1_entry_straight")
    orchestrator.reset(env, "agent0")

    orchestrator.before_step(env, "agent0", 5)

    assert len(traffic_manager._traffic_vehicles) == 1
    spawned = traffic_manager._traffic_vehicles[0]
    assert spawned.scenario_managed_vehicle is True
    assert spawned.scenario_warning_marker == "BG"
    assert spawned.scenario_vehicle_role == "injected_background"
