from __future__ import annotations

from routes.route_definitions import get_required_preset, get_route_blocks
from scenarios.definitions import (
    DEFAULT_SCENARIO_WEIGHTS,
    ROUTE_TO_SCENARIOS,
    SCENARIO_BY_ID,
    SCENARIO_DEFINITIONS,
    SCENARIO_EXPERT_OVERRIDES,
    SCENARIO_TO_ROUTES,
)


def test_default_scenario_weights_cover_all_definitions():
    assert {scenario.scenario_id for scenario in SCENARIO_DEFINITIONS} == set(DEFAULT_SCENARIO_WEIGHTS)
    assert all(weight > 0.0 for weight in DEFAULT_SCENARIO_WEIGHTS.values())


def test_route_and_scenario_mappings_are_consistent():
    for scenario in SCENARIO_DEFINITIONS:
        assert SCENARIO_TO_ROUTES[scenario.scenario_id] == scenario.allowed_local_routes
        for route_name in scenario.allowed_local_routes:
            assert scenario.scenario_id in ROUTE_TO_SCENARIOS[route_name]
            assert route_name in scenario.trigger_by_local_route


def test_allowed_route_presets_are_derived_from_local_routes():
    scenario = SCENARIO_BY_ID["S2_free_cruise_curve"]

    assert set(scenario.allowed_route_presets) == {
        get_required_preset("R5_ramp_curve"),
    }


def test_get_trigger_spec_is_route_specific():
    scenario = SCENARIO_BY_ID["S2_free_cruise_curve"]

    assert scenario.get_trigger_spec("R2_entry_curve").block_id == "c0"
    assert scenario.get_trigger_spec("R5_ramp_curve").block_id == "c0_ramp0"


def test_mainline_straight_and_transition_routes_are_split_cleanly():
    assert get_route_blocks("R3_mainline_straight") == ("s_main0",)
    assert get_route_blocks("R4_mainline_transition") == ("x0",)
    assert get_route_blocks("R3_post_transition_straight") == ("s_main1",)


def test_merge_and_exit_scenarios_use_routes_that_start_before_key_interaction_area():
    assert get_route_blocks("R6_exit_to_ramp") == ("g0", "s_ramp0", "c0_ramp0")
    assert get_route_blocks("R6_mainline_merge_approach") == ("c2", "g1", "c3")

    s6 = SCENARIO_BY_ID["S6_background_merge_in"]
    assert s6.allowed_local_routes == ("R6_mainline_merge_approach",)
    assert s6.get_trigger_spec("R6_mainline_merge_approach").block_id == "g1"
    assert s6.ego_spawn_lane_preference == "rightmost"
    assert len(s6.traffic_recipes) == 1
    assert s6.traffic_recipes[0].params["reference_kind"] == "block_route_road"
    assert s6.traffic_recipes[0].params["block_id"] == "h_ramp0"
    assert "socket_index" not in s6.traffic_recipes[0].params
    assert s6.traffic_recipes[0].params["spawn_longitude_relative_to_ego"] is True
    assert tuple(s6.traffic_recipes[0].params["spawn_longitude_range"]) == (-12.0, 12.0)
    assert tuple(s6.traffic_recipes[0].params["target_speed_range_kmh"]) == (20.0, 26.0)
    assert s6.traffic_recipes[0].params["min_agent_clearance_m"] == 2.0
    assert s6.traffic_recipes[0].params["max_spawn_attempts"] == 6
    assert SCENARIO_EXPERT_OVERRIDES["S6_background_merge_in"]["enable_lane_change"] is True

    s7 = SCENARIO_BY_ID["S7_ego_merge_from_ramp"]
    s7_trigger = s7.get_trigger_spec("R7_merge_core")
    assert s7_trigger.block_id == "h_ramp0"
    assert (s7_trigger.longitudinal_min, s7_trigger.longitudinal_max) == (0.0, 60.0)
    assert len(s7.traffic_recipes) == 1
    assert s7.traffic_recipes[0].params["reference_kind"] == "block_internal_road"
    assert s7.traffic_recipes[0].params["block_id"] == "g1"
    assert s7.traffic_recipes[0].params["internal_road_index"] == 0
    assert s7.traffic_recipes[0].params["lane_index"] == "rightmost"
    assert s7.traffic_recipes[0].params["spawn_longitude_relative_to_ego"] is True
    assert tuple(s7.traffic_recipes[0].params["spawn_longitude_range"]) == (-12.0, 12.0)
    assert tuple(s7.traffic_recipes[0].params["target_speed_range_kmh"]) == (20.0, 26.0)
    assert s7.traffic_recipes[0].params["min_agent_clearance_m"] == 2.0
    assert s7.traffic_recipes[0].params["max_spawn_attempts"] == 6

    s8 = SCENARIO_BY_ID["S8_ego_exit_to_ramp"]
    assert s8.allowed_local_routes == ("R6_exit_to_ramp",)
    trigger = s8.get_trigger_spec("R6_exit_to_ramp")
    assert trigger.block_id == "g0"
    assert (trigger.longitudinal_min, trigger.longitudinal_max) == (0.0, 130.0)
    assert s8.ego_spawn_lane_preference == "middle"
    assert len(s8.traffic_recipes) == 1
    assert s8.traffic_recipes[0].params["reference_kind"] == "ego_adjacent_lane"
    assert s8.traffic_recipes[0].params["lane_offset"] == 1
    assert tuple(s8.traffic_recipes[0].params["spawn_longitude_range"]) == (-12.0, 12.0)
    assert tuple(s8.traffic_recipes[0].params["target_speed_range_kmh"]) == (20.0, 26.0)
    assert s8.traffic_recipes[0].params["min_agent_clearance_m"] == 2.0
    assert s8.traffic_recipes[0].params["max_spawn_attempts"] == 6

def test_scenario_codes_cover_s1_to_s12():
    scenario_codes = {scenario.code for scenario in SCENARIO_DEFINITIONS}
    assert scenario_codes == {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"}


def test_s12_cut_in_from_adjacent_lane_definition():
    s12 = SCENARIO_BY_ID["S12_cut_in_from_adjacent_lane"]
    assert s12.allowed_local_routes == ("R1_entry_straight", "R3_mainline_straight", "R3_post_transition_straight")
    assert s12.ego_spawn_lane_preference == "middle"
    assert s12.get_trigger_spec("R1_entry_straight").block_id == "s0"
    assert s12.get_trigger_spec("R3_mainline_straight").block_id == "s_main0"
    assert s12.get_trigger_spec("R3_post_transition_straight").block_id == "s_main1"
    assert len(s12.traffic_recipes) == 1
    recipe = s12.traffic_recipes[0]
    assert recipe.operation == "cut_in_adjacent_lead"
    assert recipe.params["cut_in_target"] == "nearest_platoon_gap"
    assert recipe.params["reference_kind"] == "ego_adjacent_lane"
    assert recipe.params["lane_offset"] == -1
    assert tuple(recipe.params["spawn_longitude_range"]) == (-12.0, 12.0)
    assert tuple(recipe.params["cut_in_delay_steps_range"]) == (30, 60)
    assert tuple(recipe.params["speed_delta_range_kmh"]) == (2.0, 5.0)
    assert recipe.params["max_spawn_attempts"] == 6
    assert SCENARIO_EXPERT_OVERRIDES["S12_cut_in_from_adjacent_lane"]["enable_lane_change"] is True


def test_s9_uses_probabilistic_spawn_lane_and_keeps_lane_change_enabled():
    s9 = SCENARIO_BY_ID["S9_narrow_channel_negotiation"]
    assert get_route_blocks("R8_narrow_channel") == ("merge0", "s_main2", "split0")
    assert s9.ego_spawn_lane_preference is None
    assert s9.ego_spawn_lane_probabilities == {
        "rightmost": 0.4,
        "middle": 0.4,
        "leftmost": 0.2,
    }
    assert s9.ego_spawn_reference_block_id == "c3"
    assert s9.ego_spawn_reference_kind == "block_internal_road"
    assert s9.ego_spawn_internal_road_index == 1
    assert s9.ego_spawn_distance_to_route_end_m == 30.0
    trigger = s9.get_trigger_spec("R8_narrow_channel")
    assert trigger.block_id == "c3"
    assert (trigger.longitudinal_min, trigger.longitudinal_max) == (0.0, 100.0)
    assert len(s9.traffic_recipes) == 1
    recipe = s9.traffic_recipes[0]
    assert recipe.params["reference_kind"] == "ego_adjacent_lane"
    assert recipe.params["lane_offset"] == -1
    assert tuple(recipe.params["spawn_longitude_range"]) == (-12.0, 12.0)
    assert tuple(recipe.params["target_speed_range_kmh"]) == (20.0, 26.0)
    assert recipe.params["min_agent_clearance_m"] == 2.0
    assert recipe.params["max_spawn_attempts"] == 6
    assert SCENARIO_EXPERT_OVERRIDES["S9_narrow_channel_negotiation"]["enable_lane_change"] is True
