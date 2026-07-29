from __future__ import annotations

import pytest

from scenarios.definitions import SCENARIO_BY_ID


def test_s6_declares_ego_spawn_reference_block_and_start_longitude() -> None:
    scenario = SCENARIO_BY_ID["S6_background_merge_in"]

    assert scenario.ego_spawn_reference_block_id == "g1"
    assert scenario.ego_spawn_reference_kind == "block_internal_road"
    assert scenario.ego_spawn_internal_road_index == 1
    assert scenario.ego_spawn_longitude_m == (72.0, 78.0)
    assert scenario.ego_initial_speed_km_h == (23.0, 25.0)
    assert scenario.override_traffic_density == pytest.approx(0.0)
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
    assert recipe.params["merge_creep_speed_kmh"] == pytest.approx(20.0)
    assert recipe.params["target_speed_kmh"] == pytest.approx(24.0)
    assert recipe.params["merge_arrival_offset_s"] == pytest.approx(1.2)
    assert "spawn_longitude" not in recipe.params


def test_s6_declares_only_one_controlled_merge_vehicle() -> None:
    scenario = SCENARIO_BY_ID["S6_background_merge_in"]
    assert len(scenario.traffic_recipes) == 1
    recipe = scenario.traffic_recipes[0]
    assert recipe.operation == "inject_background_vehicle"
    assert recipe.params["reference_kind"] == "block_socket_road"
    assert recipe.params["trigger_on_start"] is True


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
            "spawn_longitude_offset_range_m": (-15.0, -13.0),
            "target_speed_kmh": 24.0,
        },
        {
            "name": "right_side",
            "lane_side": "right",
            "spawn_longitude_offset_range_m": (13.0, 15.0),
            "target_speed_kmh": 24.0,
        },
    )
    assert scenario.allowed_local_routes == ("R1_entry_straight",)
    assert tuple(scenario.trigger_by_local_route) == ("R1_entry_straight",)
    assert scenario.override_traffic_density == 0.0


def test_s7_declares_mainline_background_traffic_on_three_lanes() -> None:
    scenario = SCENARIO_BY_ID["S7_ego_merge_from_ramp"]
    recipes = [
        recipe
        for recipe in scenario.traffic_recipes
        if recipe.operation == "inject_background_vehicle"
    ]

    assert len(recipes) == 10
    assert len({recipe.params["name"] for recipe in recipes}) == 10
    assert all(recipe.params["reference_kind"] == "block_route_road" for recipe in recipes)
    assert all(recipe.params["block_id"] == "g1" for recipe in recipes)
    assert all(recipe.params["trigger_on_start"] is True for recipe in recipes)

    lane_ids = [recipe.params["lane_id"] for recipe in recipes]
    assert sorted(set(lane_ids)) == [0, 1, 2]
    assert {lane_id: lane_ids.count(lane_id) for lane_id in sorted(set(lane_ids))} == {
        0: 3,
        1: 4,
        2: 3,
    }


def test_s5_declares_hard_brake_random_ranges() -> None:
    scenario = SCENARIO_BY_ID["S5_hard_brake_lead"]
    recipes = [recipe for recipe in scenario.traffic_recipes if recipe.operation == "hard_brake_lead"]

    assert len(recipes) == 1
    params = recipes[0].params
    assert params["trigger_after_s"] == 3.0
    assert params["lead_bumper_gap_range_m"] == (9.0, 13.0)
    assert params["lead_target_speed_range_kmh"] == (22.0, 26.0)
    assert params["brake_target_speed_range_kmh"] == (0.5, 2.0)
    assert params["brake_deceleration_range_mps2"] == (5.0, 7.0)


def test_s9_declares_safe_internal_spawn_road() -> None:
    scenario = SCENARIO_BY_ID["S9_narrow_channel_negotiation"]

    assert scenario.ego_spawn_reference_block_id == "c3"
    assert scenario.ego_spawn_reference_kind == "block_internal_road"
    assert scenario.ego_spawn_internal_road_index == 1
    assert scenario.ego_spawn_longitude_m == pytest.approx(10.0)
    trigger = scenario.trigger_by_local_route["R8_narrow_channel"]
    assert trigger.block_id == "c3"
    assert trigger.longitudinal_min == pytest.approx(35.0)


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
