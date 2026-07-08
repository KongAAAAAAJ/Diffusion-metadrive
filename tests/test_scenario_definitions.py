from __future__ import annotations

import pytest

from scenarios.definitions import SCENARIO_BY_ID


def test_s6_declares_ego_spawn_reference_block_and_start_longitude() -> None:
    scenario = SCENARIO_BY_ID["S6_background_merge_in"]

    assert scenario.ego_spawn_reference_block_id == "g1"
    assert scenario.ego_spawn_longitude_m == pytest.approx(12.0)


def test_s5_declares_adjacent_lane_side_vehicle_recipe() -> None:
    scenario = SCENARIO_BY_ID["S5_hard_brake_lead"]

    assert scenario.ego_spawn_lane_preference == "middle"
    assert scenario.ego_spawn_lane_probabilities is None

    recipes = [recipe for recipe in scenario.traffic_recipes if recipe.operation == "inject_adjacent_lane_vehicles"]

    assert len(recipes) == 1
    assert recipes[0].params["trigger_on_start"] is True
    assert recipes[0].params["clearance_scope"] == "same_lane"
    vehicles = recipes[0].params["vehicles"]
    assert vehicles == (
        {
            "name": "left_side",
            "lane_side": "left",
            "spawn_longitude_offset_m": -8.0,
            "spawn_longitude_offset_range_m": (-10.0, -6.0),
            "target_speed_kmh": 18.0,
            "target_speed_range_kmh": (16.0, 20.0),
        },
        {
            "name": "right_side",
            "lane_side": "right",
            "spawn_longitude_offset_m": 6.0,
            "spawn_longitude_offset_range_m": (4.0, 8.0),
            "target_speed_kmh": 19.0,
            "target_speed_range_kmh": (17.0, 21.0),
        },
    )


def test_s5_declares_hard_brake_random_ranges() -> None:
    scenario = SCENARIO_BY_ID["S5_hard_brake_lead"]
    recipes = [recipe for recipe in scenario.traffic_recipes if recipe.operation == "hard_brake_lead"]

    assert len(recipes) == 1
    params = recipes[0].params
    assert params["lead_distance_range_m"] == (45.0, 55.0)
    assert params["lead_target_speed_range_kmh"] == (19.0, 23.0)
    assert params["brake_target_speed_range_kmh"] == (0.5, 2.0)
    assert params["brake_duration_steps_range"] == (450, 550)


def test_s9_declares_safe_internal_spawn_road() -> None:
    scenario = SCENARIO_BY_ID["S9_narrow_channel_negotiation"]

    assert scenario.ego_spawn_reference_block_id == "merge0"
    assert scenario.ego_spawn_reference_kind == "block_internal_road"
    assert scenario.ego_spawn_internal_road_index == 3
    assert scenario.ego_spawn_longitude_m == pytest.approx(10.0)


def test_s8_declares_explicit_ego_spawn_controls() -> None:
    scenario = SCENARIO_BY_ID["S8_ego_exit_to_ramp"]

    assert scenario.ego_spawn_reference_block_id == "g0"
    assert scenario.ego_spawn_longitude_m == pytest.approx(30.0)
    assert scenario.ego_spawn_lane_id == 1


def test_s8_injected_background_recipes_declare_lane_index() -> None:
    scenario = SCENARIO_BY_ID["S8_ego_exit_to_ramp"]
    recipes = [recipe for recipe in scenario.traffic_recipes if recipe.operation == "inject_background_vehicle"]

    assert recipes
    for recipe in recipes:
        assert recipe.params["reference_kind"] == "block_internal_road"
        assert recipe.params["block_id"] == "g0"
        assert "socket_index" not in recipe.params
        assert recipe.params["internal_road_index"] == 0
        assert recipe.params["lane_index"] == 2
