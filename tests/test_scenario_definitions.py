from __future__ import annotations

import pytest

from scenarios.definitions import SCENARIO_BY_ID


def test_s6_declares_ego_spawn_reference_block_and_distance_to_merge() -> None:
    scenario = SCENARIO_BY_ID["S6_background_merge_in"]

    assert scenario.ego_spawn_reference_block_id == "g1"
    assert scenario.ego_spawn_reference_kind == "block_internal_road"
    assert scenario.ego_spawn_internal_road_index == 1
    assert scenario.ego_spawn_longitude_m is None
    assert scenario.ego_spawn_distance_to_route_end_m == (25.0, 50.0)
    assert scenario.ego_initial_speed_km_h == (22.0, 27.0)
    # S6 starts just above the unchanged 7 m hard gate and creates the actor corridor
    # dynamically before the ego-only platoon changes lane and reassembles.
    assert scenario.ego_initial_bumper_gap_m == pytest.approx(8.0)
    assert scenario.override_traffic_density == pytest.approx(0.0)
    assert scenario.env_overrides["traffic_spawn_exclusion_ahead_m"] == 100.0
    assert scenario.env_overrides["traffic_spawn_exclusion_behind_m"] == 100.0
    assert scenario.env_overrides["platoon_route_spawn_lane_index"] == 1


def test_s6_declares_merge_aware_background_policy() -> None:
    scenario = SCENARIO_BY_ID["S6_background_merge_in"]
    recipe = next(recipe for recipe in scenario.traffic_recipes if recipe.operation == "inject_background_vehicle")

    assert scenario.trigger_by_local_route["R6_mainline_merge_approach"].block_id == "g1"
    assert recipe.params["policy"] == "idm_merge"
    assert recipe.params["target_speed_range_kmh"] == (18.0, 27.0)
    assert recipe.params["target_gap_choices"] == ("agent0-agent1", "agent1-agent2")
    assert recipe.params["merge_arrival_delta_range_s"] == (-0.5, 0.5)
    assert recipe.params["scenario_vehicle_role"] == "s6_gap_intruder"
    assert "spawn_longitude" not in recipe.params


def test_s6_declares_one_controlled_merge_vehicle_plus_incidental_traffic() -> None:
    scenario = SCENARIO_BY_ID["S6_background_merge_in"]
    assert len(scenario.traffic_recipes) == 2
    recipe = next(
        row
        for row in scenario.traffic_recipes
        if row.operation == "inject_background_vehicle"
    )
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
    assert {row["lane_side"] for row in vehicles} == {"left", "right"}
    assert all(row["relation_choices"] == ("ahead", "behind") for row in vehicles)
    assert all(row["behind_offset_range_m"] == (-20.0, -10.0) for row in vehicles)
    assert all(row["ahead_offset_range_m"] == (10.0, 22.0) for row in vehicles)
    assert all(row["target_speed_range_kmh"] == (18.0, 32.0) for row in vehicles)
    assert scenario.allowed_local_routes == ("R1_entry_straight",)
    assert tuple(scenario.trigger_by_local_route) == ("R1_entry_straight",)
    assert scenario.override_traffic_density == 0.0


def test_s7_declares_atomic_gap_conditioned_mainline_traffic() -> None:
    scenario = SCENARIO_BY_ID["S7_ego_merge_from_ramp"]
    recipes = [
        recipe
        for recipe in scenario.traffic_recipes
        if recipe.operation == "inject_s7_merge_traffic"
    ]

    assert len(recipes) == 1
    assert recipes[0].params["required_roles"] == (
        "critical_gap_front", "critical_gap_rear", "next_gap_front", "next_gap_rear"
    )
    assert recipes[0].params["optional_adjacent_actor_range"] == (0, 2)


def test_s5_declares_hard_brake_random_ranges() -> None:
    scenario = SCENARIO_BY_ID["S5_hard_brake_lead"]
    recipes = [recipe for recipe in scenario.traffic_recipes if recipe.operation == "hard_brake_lead"]

    assert len(recipes) == 1
    params = recipes[0].params
    assert params["trigger_after_range_s"] == (2.5, 4.0)
    assert params["lead_bumper_gap_range_m"] == (9.0, 15.0)
    assert params["brake_target_speed_range_kmh"] == (0.0, 3.0)
    assert params["brake_deceleration_range_mps2"] == (4.5, 7.0)


def test_s9_declares_safe_internal_spawn_road() -> None:
    scenario = SCENARIO_BY_ID["S9_narrow_channel_negotiation"]

    assert scenario.ego_spawn_reference_block_id == "c3"
    assert scenario.ego_spawn_reference_kind == "block_internal_road"
    assert scenario.ego_spawn_internal_road_index == 1
    assert scenario.ego_spawn_longitude_m == (50.0, 75.0)
    assert scenario.ego_spawn_lane_id == 1
    assert scenario.ego_spawn_lane_probabilities is None
    trigger = scenario.trigger_by_local_route["R8_narrow_channel"]
    assert trigger.block_id == "c3"
    assert trigger.longitudinal_min == pytest.approx(35.0)


def test_s8_declares_explicit_ego_spawn_controls() -> None:
    scenario = SCENARIO_BY_ID["S8_ego_exit_to_ramp"]

    assert scenario.ego_spawn_reference_block_id == "g0"
    assert scenario.ego_spawn_longitude_m == (68.0, 70.0)
    assert scenario.ego_spawn_lane_id == 1
    assert scenario.ego_initial_bumper_gap_m == pytest.approx(18.0)
    assert scenario.override_traffic_density == pytest.approx(0.0)


def test_s8_declares_atomic_exit_gap_recipe() -> None:
    scenario = SCENARIO_BY_ID["S8_ego_exit_to_ramp"]
    recipes = [recipe for recipe in scenario.traffic_recipes if recipe.operation == "inject_s8_exit_gap"]

    assert recipes
    for recipe in recipes:
        assert recipe.params["block_id"] == "g0"
        assert "socket_index" not in recipe.params
        assert recipe.params["internal_road_index"] == 0
        assert recipe.params["lane_index"] == 2
    assert len(recipes) == 1
    assert recipes[0].params["trigger_on_start"] is True


def test_every_s5_s9_scenario_declares_three_to_six_incidental_actors() -> None:
    for number in range(5, 10):
        scenario = next(
            value for value in SCENARIO_BY_ID.values() if value.code == f"S{number}"
        )
        recipes = [
            recipe
            for recipe in scenario.traffic_recipes
            if recipe.operation == "inject_incidental_background_traffic"
        ]
        assert len(recipes) == 1
        assert recipes[0].params["actor_count_range"] == (3, 6)
        assert recipes[0].params["trigger_on_start"] is True
