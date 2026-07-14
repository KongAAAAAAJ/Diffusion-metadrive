from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scenarios.definitions import SCENARIO_BY_ID
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
    def __init__(self, rng=None) -> None:
        self._traffic_vehicles = []
        self.spawn_calls = []
        self.np_random = rng
        self.policies = {}

    def random_vehicle_type(self):
        return "vehicle"

    def _spawn_traffic_vehicle_if_safe(
        self,
        vehicle_type,
        config,
        *,
        policy_class=None,
        policy_kwargs=None,
        vehicle_config_overrides=None,
    ):
        name = f"traffic_{len(self._traffic_vehicles)}"
        vehicle = SimpleNamespace(name=name, spawn_config=config)
        self._traffic_vehicles.append(vehicle)
        self.spawn_calls.append(
            (vehicle_type, config, policy_class, policy_kwargs or {}, vehicle_config_overrides or {})
        )
        self.policies[name] = SimpleNamespace(target_speed=0.0)
        return vehicle


class _FixedUniformRng:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def uniform(self, low, high):
        return self.value


def make_env_and_ego(lane_count: int = 3, ego_lane_index: int = 1, rng=None):
    lanes = [FakeLane(index) for index in range(lane_count)]
    road_network = FakeRoadNetwork(lanes)
    block = SimpleNamespace(
        graph_block_id="s0",
        block_network=SimpleNamespace(get_positive_lanes=lambda: [lanes]),
        get_socket_list=lambda: [],
        get_respawn_roads=lambda: [],
    )
    traffic_manager = FakeTrafficManager(rng=rng)
    current_map = SimpleNamespace(road_network=road_network, blocks=[block])
    engine = SimpleNamespace(
        current_map=current_map,
        traffic_manager=traffic_manager,
        get_policy=lambda name: traffic_manager.policies.get(name),
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
    spawned_velocities = [call[1]["spawn_velocity"] for call in traffic_manager.spawn_calls]

    assert spawned_lanes == [("road_a", "road_b", 0), ("road_a", "road_b", 2)]
    assert spawned_longs == [102.0, 102.0]
    assert spawned_velocities == [(24.0 / 3.6, 0.0), (24.0 / 3.6, 0.0)]
    assert all(call[1]["spawn_velocity_car_frame"] is True for call in traffic_manager.spawn_calls)
    assert orchestrator.summary.scenario_triggered is True
    assert orchestrator.summary.scenario_realized is True
    assert orchestrator.summary.notes.count("adjacent_spawned:left_side") == 1
    assert orchestrator.summary.notes.count("adjacent_spawned:right_side") == 1


def test_s6_fixed_traffic_spawns_at_head_relative_positions_only_once() -> None:
    env, ego, traffic_manager = make_env_and_ego()
    orchestrator = ScenarioOrchestrator(
        SCENARIO_BY_ID["S6_background_merge_in"],
        "R6_mainline_merge_approach",
    )
    orchestrator.reset(env, "agent0")
    orchestrator._road_to_block_id = {("road_a", "road_b"): "g1"}

    orchestrator.before_step(env, "agent0", 1)
    orchestrator.before_step(env, "agent0", 2)

    ordinary_calls = [call for call in traffic_manager.spawn_calls if call[2] is None]
    assert len(ordinary_calls) == 15
    assert [call[1]["spawn_longitude"] for call in ordinary_calls[:7]] == [
        145.0,
        170.0,
        198.0,
        70.0,
        45.0,
        10.0,
        2.0,
    ]
    assert [call[1]["spawn_lane_index"][-1] for call in ordinary_calls[7:]] == [
        0, 0, 0, 0, 2, 2, 2, 2
    ]
    assert [call[1]["spawn_longitude"] for call in ordinary_calls[7:11]] == [
        ego.position[0] + offset for offset in (-40.0, -10.0, 15.0, 35.0)
    ]


def test_s6_fixed_traffic_skips_missing_adjacent_lane() -> None:
    env, _ego, traffic_manager = make_env_and_ego(lane_count=2, ego_lane_index=1)
    orchestrator = ScenarioOrchestrator(
        SCENARIO_BY_ID["S6_background_merge_in"],
        "R6_mainline_merge_approach",
    )
    orchestrator.reset(env, "agent0")
    orchestrator._road_to_block_id = {("road_a", "road_b"): "g1"}

    orchestrator.before_step(env, "agent0", 1)

    ordinary_calls = [call for call in traffic_manager.spawn_calls if call[2] is None]
    assert len(ordinary_calls) == 11
    assert [call[1]["spawn_lane_index"][-1] for call in ordinary_calls[7:]] == [0, 0, 0, 0]
    assert sum(note.startswith("adjacent_lane_missing:right_") for note in orchestrator.summary.notes) == 4


def test_s5_hard_brake_recipe_randomizes_from_traffic_manager_rng(monkeypatch) -> None:
    rng = np.random.RandomState(123)
    env, ego, _traffic_manager = make_env_and_ego(rng=rng)
    ego.speed_km_h = 30.0
    orchestrator = ScenarioOrchestrator(SCENARIO_BY_ID["S5_hard_brake_lead"], "R3_mainline_straight")
    orchestrator.reset(env, "agent0")
    orchestrator._road_to_block_id = {("road_a", "road_b"): "s_main0"}
    monkeypatch.setattr(orchestrator, "_find_front_vehicle_with_distance", lambda vehicle: (None, None))

    captured_params = {}

    def fake_spawn(env_arg, ego_arg, params):
        captured_params.update(params)
        return SimpleNamespace(name="spawned_lead")

    monkeypatch.setattr(orchestrator, "_spawn_lead_vehicle", fake_spawn)

    orchestrator.before_step(env, "agent0", 31)

    expected_rng = np.random.RandomState(123)
    assert captured_params["lead_distance_m"] == float(expected_rng.uniform(10.0, 15.0))
    assert captured_params["lead_target_speed_kmh"] == float(expected_rng.uniform(19.0, 23.0))
    assert captured_params["brake_target_speed_kmh"] == float(expected_rng.uniform(0.5, 2.0))
    assert captured_params["brake_duration_steps"] == int(expected_rng.randint(450, 551))
    assert SCENARIO_BY_ID["S5_hard_brake_lead"].traffic_recipes[0].params["lead_distance_m"] == 10.0


def test_s5_adjacent_recipe_randomizes_each_vehicle_from_traffic_manager_rng() -> None:
    rng = np.random.RandomState(321)
    env, ego, traffic_manager = make_env_and_ego(rng=rng)
    ego.position = (20.0, ego.position[1])
    orchestrator = ScenarioOrchestrator(SCENARIO_BY_ID["S5_hard_brake_lead"], "R3_mainline_straight")
    orchestrator.reset(env, "agent0")
    orchestrator._road_to_block_id = {("road_a", "road_b"): "s_main0"}

    orchestrator.before_step(env, "agent0", 1)

    expected_rng = np.random.RandomState(321)
    left_offset = float(expected_rng.uniform(-10.0, -6.0))
    left_speed = float(expected_rng.uniform(16.0, 20.0))
    right_offset = float(expected_rng.uniform(4.0, 8.0))
    right_speed = float(expected_rng.uniform(17.0, 21.0))

    assert [call[1]["spawn_lane_index"] for call in traffic_manager.spawn_calls] == [
        ("road_a", "road_b", 0),
        ("road_a", "road_b", 2),
    ]
    assert [call[1]["spawn_longitude"] for call in traffic_manager.spawn_calls] == [
        20.0 + left_offset,
        20.0 + right_offset,
    ]
    assert [traffic_manager.policies[vehicle.name].target_speed for vehicle in traffic_manager._traffic_vehicles] == [
        left_speed,
        right_speed,
    ]


def test_hard_brake_recipe_without_ranges_keeps_static_params(monkeypatch) -> None:
    rng = np.random.RandomState(123)
    env, _ego, _traffic_manager = make_env_and_ego(rng=rng)
    definition = ScenarioDefinition(
        code="T",
        scenario_id="test_static_hard_brake",
        allowed_local_routes=("R1_entry_straight",),
        trigger_by_local_route={"R1_entry_straight": TriggerSpec("s0", 80.0, 120.0)},
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
    orchestrator = ScenarioOrchestrator(definition, "R1_entry_straight")
    orchestrator.reset(env, "agent0")
    monkeypatch.setattr(orchestrator, "_find_front_vehicle_with_distance", lambda vehicle: (None, None))

    captured_params = {}

    def fake_spawn(env_arg, ego_arg, params):
        captured_params.update(params)
        return SimpleNamespace(name="spawned_lead")

    monkeypatch.setattr(orchestrator, "_spawn_lead_vehicle", fake_spawn)

    orchestrator.before_step(env, "agent0", 1)

    assert captured_params["lead_distance_m"] == 10.0
    assert captured_params["lead_target_speed_kmh"] == 21.0
    assert captured_params["brake_target_speed_kmh"] == 1.0
    assert captured_params["brake_duration_steps"] == 200


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
    assert spawned.scenario_warning_marker == "!"
    assert spawned.scenario_vehicle_role == "injected_background"
    assert spawned.spawn_config["spawn_velocity"] == (18.0 / 3.6, 0.0)
    assert spawned.spawn_config["spawn_velocity_car_frame"] is True


def test_injected_background_vehicle_samples_spawn_longitude_range() -> None:
    env, _ego, traffic_manager = make_env_and_ego(rng=_FixedUniformRng(7.5))
    definition = ScenarioDefinition(
        code="T",
        scenario_id="test_injected_background_range",
        allowed_local_routes=("R1_entry_straight",),
        trigger_by_local_route={"R1_entry_straight": TriggerSpec("s0", 80.0, 120.0)},
        traffic_recipes=(
            RecipeSpec(
                "inject_background_vehicle",
                {
                    "reference_kind": "ego_lane",
                    "spawn_longitude": (0.0, 20.0),
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

    assert traffic_manager.spawn_calls[0][1]["spawn_longitude"] == pytest.approx(97.5)


def test_hard_brake_spawned_lead_uses_lead_target_speed_as_initial_velocity(monkeypatch) -> None:
    env, _ego, traffic_manager = make_env_and_ego()
    definition = ScenarioDefinition(
        code="T",
        scenario_id="test_hard_brake_initial_velocity",
        allowed_local_routes=("R1_entry_straight",),
        trigger_by_local_route={"R1_entry_straight": TriggerSpec("s0", 80.0, 120.0)},
        traffic_recipes=(
            RecipeSpec(
                "hard_brake_lead",
                {
                    "lead_distance_m": 10.0,
                    "lead_target_speed_kmh": 21.0,
                    "front_distance_min_m": 10.0,
                    "front_distance_max_m": 15.0,
                    "brake_target_speed_kmh": 1.0,
                    "brake_duration_steps": 30,
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
    monkeypatch.setattr(orchestrator, "_find_front_vehicle_with_distance", lambda vehicle: (None, None))

    orchestrator.before_step(env, "agent0", 1)

    assert len(traffic_manager.spawn_calls) == 1
    spawn_config = traffic_manager.spawn_calls[0][1]
    assert spawn_config["spawn_velocity"] == (21.0 / 3.6, 0.0)
    assert spawn_config["spawn_velocity_car_frame"] is True


def test_s6_injected_background_uses_merge_policy_and_start_edge_navigation() -> None:
    from envs.diffusion_envs.idm_merge_policy import IDMMergePolicy, StartEdgeNodeNavigation

    env, _ego, traffic_manager = make_env_and_ego()
    definition = ScenarioDefinition(
        code="T",
        scenario_id="test_merge_background",
        allowed_local_routes=("R1_entry_straight",),
        trigger_by_local_route={"R1_entry_straight": TriggerSpec("s0", 80.0, 120.0)},
        traffic_recipes=(
            RecipeSpec(
                "inject_background_vehicle",
                {
                    "reference_kind": "ego_lane",
                    "spawn_longitude": 15.0,
                    "target_speed_kmh": 24.0,
                    "policy": "idm_merge",
                    "merge_front_gap_m": 25.0,
                    "merge_rear_gap_m": 15.0,
                    "merge_creep_speed_kmh": 5.0,
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

    call = traffic_manager.spawn_calls[0]
    assert call[2] is IDMMergePolicy
    assert call[3] == {
        "merge_front_gap_m": 25.0,
        "merge_rear_gap_m": 15.0,
        "merge_creep_speed_kmh": 5.0,
        "merge_cruise_speed_kmh": 24.0,
    }
    assert call[4]["navigation_module"] is StartEdgeNodeNavigation
    assert call[1]["spawn_velocity"] == (24.0 / 3.6, 0.0)
    assert call[1]["spawn_velocity_car_frame"] is True


def test_unknown_injected_background_policy_is_rejected() -> None:
    env, _ego, _traffic_manager = make_env_and_ego()
    definition = ScenarioDefinition(
        code="T",
        scenario_id="test_unknown_policy",
        allowed_local_routes=("R1_entry_straight",),
        trigger_by_local_route={"R1_entry_straight": TriggerSpec("s0", 80.0, 120.0)},
        traffic_recipes=(
            RecipeSpec(
                "inject_background_vehicle",
                {"reference_kind": "ego_lane", "policy": "does_not_exist"},
            ),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="test",
        description="test",
    )
    orchestrator = ScenarioOrchestrator(definition, "R1_entry_straight")
    orchestrator.reset(env, "agent0")

    with pytest.raises(ValueError, match="Unknown injected background policy"):
        orchestrator.before_step(env, "agent0", 5)
