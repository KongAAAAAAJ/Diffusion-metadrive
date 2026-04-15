from __future__ import annotations

from metadrive.exp_dataset.route_definitions import get_required_preset
from metadrive.exp_dataset.route_definitions import get_route_blocks
from metadrive.exp_dataset.scenario_definitions import (
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
        get_required_preset("R2_entry_curve"),
        get_required_preset("R5_ramp_curve"),
        get_required_preset("R9_post_split_curve"),
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
    assert s6.get_trigger_spec("R6_mainline_merge_approach").block_id == "c2"
    assert s6.ego_spawn_lane_preference == "rightmost"
    assert SCENARIO_EXPERT_OVERRIDES["S6_background_merge_in"]["enable_lane_change"] is True

    s8 = SCENARIO_BY_ID["S8_ego_exit_to_ramp"]
    assert s8.allowed_local_routes == ("R6_exit_to_ramp",)
    trigger = s8.get_trigger_spec("R6_exit_to_ramp")
    assert trigger.block_id == "g0"
    assert (trigger.longitudinal_min, trigger.longitudinal_max) == (5.0, 45.0)
    assert s8.ego_spawn_lane_preference == "rightmost"
    assert len(s8.traffic_recipes) == 1
    assert s8.traffic_recipes[0].params["reference_kind"] == "block_socket_road"
    assert s8.traffic_recipes[0].params["block_id"] == "g0"
    assert s8.traffic_recipes[0].params["socket_index"] == 1

def test_only_s1_to_s9_scenarios_remain():
    scenario_codes = {scenario.code for scenario in SCENARIO_DEFINITIONS}
    assert scenario_codes == {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"}
    assert "S10_right_biased_transition" not in SCENARIO_BY_ID


def test_s9_uses_probabilistic_spawn_lane_and_keeps_lane_change_enabled():
    s9 = SCENARIO_BY_ID["S9_narrow_channel_negotiation"]
    assert get_route_blocks("R8_narrow_channel") == ("merge0", "s_main2", "split0")
    assert s9.ego_spawn_lane_preference is None
    assert s9.ego_spawn_lane_probabilities == {
        "rightmost": 0.4,
        "middle": 0.4,
        "leftmost": 0.2,
    }
    assert SCENARIO_EXPERT_OVERRIDES["S9_narrow_channel_negotiation"]["enable_lane_change"] is True
