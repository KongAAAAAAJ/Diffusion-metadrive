from __future__ import annotations

import pytest

from scenarios.definitions import SCENARIO_BY_ID


def test_s6_declares_ego_spawn_reference_block_and_start_longitude() -> None:
    scenario = SCENARIO_BY_ID["S6_background_merge_in"]

    assert scenario.ego_spawn_reference_block_id == "g1"
    assert scenario.ego_spawn_longitude_m == (25.0, 90.0)
    assert scenario.ego_initial_speed_km_h == (20.0, 26.0)
    assert scenario.override_traffic_density == pytest.approx(0.03)
    assert scenario.env_overrides == {
        "traffic_spawn_exclusion_ahead_m": 100.0,
        "traffic_spawn_exclusion_behind_m": 100.0,
    }


def test_s6_declares_merge_aware_background_policy() -> None:
    scenario = SCENARIO_BY_ID["S6_background_merge_in"]
    recipe = next(recipe for recipe in scenario.traffic_recipes if recipe.operation == "inject_background_vehicle")

    assert scenario.trigger_by_local_route["R6_mainline_merge_approach"].block_id == "g1"
    assert recipe.params["policy"] == "idm_merge"
    assert recipe.params["merge_front_gap_m"] == pytest.approx(10.0)
    assert recipe.params["merge_rear_gap_m"] == pytest.approx(10.0)
    assert recipe.params["merge_creep_speed_kmh"] == pytest.approx(15.0)
    assert recipe.params["target_speed_kmh"] == pytest.approx(25.0)
    assert recipe.params["spawn_longitude"] == (0.0, 20.0)


def test_s6_declares_fixed_same_lane_background_traffic() -> None:
    scenario = SCENARIO_BY_ID["S6_background_merge_in"]
    recipes = [
        recipe
        for recipe in scenario.traffic_recipes
        if recipe.operation == "inject_background_vehicle"
        and recipe.params.get("reference_kind") == "ego_lane"
    ]

    assert len(recipes) == 7
    assert [recipe.params["name"] for recipe in recipes] == [
        "ego_lane_front_2",
        "ego_lane_front_3",
        "ego_lane_front_4",
        "ego_lane_rear_1",
        "ego_lane_rear_2",
        "ego_lane_rear_3",
        "ego_lane_rear_4",
    ]
    assert [recipe.params["spawn_longitude_offset"] for recipe in recipes] == [
        55.0,
        80.0,
        115.0,
        -20.0,
        -45.0,
        -80.0,
        -115.0,
    ]
    assert all(recipe.params["spawn_longitude"] == 0.0 for recipe in recipes)
    assert all(recipe.params["trigger_on_start"] is True for recipe in recipes)
    assert all("policy" not in recipe.params for recipe in recipes)


def test_s6_declares_four_background_vehicles_per_adjacent_lane() -> None:
    scenario = SCENARIO_BY_ID["S6_background_merge_in"]
    recipes = [
        recipe
        for recipe in scenario.traffic_recipes
        if recipe.operation == "inject_adjacent_lane_vehicles"
    ]

    assert len(recipes) == 1
    assert recipes[0].params["trigger_on_start"] is True
    assert recipes[0].params["clearance_scope"] == "same_lane"
    vehicles = recipes[0].params["vehicles"]
    assert len(vehicles) == 8
    assert len({vehicle["name"] for vehicle in vehicles}) == 8
    for lane_side in ("left", "right"):
        lane_vehicles = [vehicle for vehicle in vehicles if vehicle["lane_side"] == lane_side]
        assert len(lane_vehicles) == 4
        assert [vehicle["spawn_longitude_offset_m"] for vehicle in lane_vehicles] == [
            -40.0,
            -10.0,
            15.0,
            35.0,
        ]
        assert all(20.0 <= vehicle["target_speed_kmh"] <= 27.0 for vehicle in lane_vehicles)


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
    assert params["lead_distance_range_m"] == (10.0, 15.0)
    assert params["lead_target_speed_range_kmh"] == (19.0, 23.0)
    assert params["brake_target_speed_range_kmh"] == (0.5, 2.0)
    assert params["brake_duration_steps_range"] == (450, 550)


def test_s9_declares_safe_internal_spawn_road() -> None:
    scenario = SCENARIO_BY_ID["S9_narrow_channel_negotiation"]

    assert scenario.ego_spawn_reference_block_id == "c3"
    assert scenario.ego_spawn_reference_kind == "block_internal_road"
    assert scenario.ego_spawn_internal_road_index == 1
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
