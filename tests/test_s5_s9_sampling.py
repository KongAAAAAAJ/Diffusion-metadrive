from __future__ import annotations

import pytest

from scenarios.s5_s9_sampling import resolve_s5_s9_parameters


SEEDS = (17, 23, 31, 47, 59)
ROUTES = {
    "S5_hard_brake_lead": "R1_entry_straight",
    "S6_background_merge_in": "R6_mainline_merge_approach",
    "S7_ego_merge_from_ramp": "R7_merge_core",
    "S8_ego_exit_to_ramp": "R6_exit_to_ramp",
    "S9_narrow_channel_negotiation": "R8_narrow_channel",
}


def _resolve(scenario_id: str, seed: int) -> dict:
    return resolve_s5_s9_parameters(
        spawn_seed=seed, scenario_id=scenario_id, local_route=ROUTES[scenario_id],
        ego_initial_speed_km_h=24.0,
    )


def _resolve_v4(seed: int) -> dict:
    return resolve_s5_s9_parameters(
        spawn_seed=seed,
        scenario_id="S5_hard_brake_lead",
        local_route=ROUTES["S5_hard_brake_lead"],
        ego_initial_speed_km_h=24.0,
        scenario_contract_id="candidate_v4",
    )


def test_sampler_is_deterministic_and_finite() -> None:
    for scenario_id in ROUTES:
        for seed in SEEDS:
            assert _resolve(scenario_id, seed) == _resolve(scenario_id, seed)
            assert _resolve(scenario_id, seed)["severity_bucket"] in {"low", "medium", "high"}


def test_every_s5_s9_sample_has_three_to_six_incidental_background_actors() -> None:
    realized_counts = set()
    for scenario_id in ROUTES:
        for seed in SEEDS:
            value = _resolve(scenario_id, seed)
            count = value["incidental_background_actor_count"]
            realized_counts.add(count)
            assert 3 <= count <= 6
            assert len(value["incidental_background_speeds_km_h"]) == count
    assert len(realized_counts) >= 2


def test_severity_bucket_distribution_tracks_declared_weights() -> None:
    counts = {"low": 0, "medium": 0, "high": 0}
    for seed in range(2000):
        counts[_resolve("S5_hard_brake_lead", seed)["severity_bucket"]] += 1
    ratios = {key: value / 2000.0 for key, value in counts.items()}
    assert abs(ratios["low"] - 0.3) < 0.04
    assert abs(ratios["medium"] - 0.4) < 0.04
    assert abs(ratios["high"] - 0.3) < 0.04


def test_s5_has_two_real_neighbors_without_fixed_relations() -> None:
    values = [_resolve("S5_hard_brake_lead", seed) for seed in SEEDS]
    assert len({(v["left_relation"], v["right_relation"]) for v in values}) >= 2
    for value in values:
        for side in ("left", "right"):
            offset = value[f"{side}_offset_m"]
            assert (-20.0 <= offset <= -10.0) or (10.0 <= offset <= 22.0)
            assert 18.0 <= value[f"{side}_speed_km_h"] <= 32.0
        assert 9.0 <= value["lead_trigger_bumper_gap_m"] <= 15.0
        assert 2.5 <= value["brake_trigger_time_s"] <= 4.0


def test_s5_fixed_seeds_couple_lead_pressure_to_adjacent_windows() -> None:
    values = {seed: _resolve("S5_hard_brake_lead", seed) for seed in SEEDS}
    assert {value["lead_pressure_bucket"] for value in values.values()} == {
        "moderate",
        "high",
    }
    assert values[17]["lead_pressure_bucket"] == "high"
    assert values[17]["left_relation"] == "ahead"
    assert values[23]["lead_pressure_bucket"] == "high"
    assert values[23]["right_relation"] == "ahead"
    for value in values.values():
        assert -3.0 <= value["lead_speed_delta_from_ego_km_h"] <= 3.0
        assert 4.5 <= value["lead_brake_deceleration_mps2"] <= 7.0
        assert 0.0 <= value["lead_target_speed_km_h"] <= 3.0


def test_candidate_v3_explicit_contract_keeps_legacy_samples_identical() -> None:
    for seed in SEEDS:
        assert _resolve("S5_hard_brake_lead", seed) == resolve_s5_s9_parameters(
            spawn_seed=seed,
            scenario_id="S5_hard_brake_lead",
            local_route=ROUTES["S5_hard_brake_lead"],
            ego_initial_speed_km_h=24.0,
            scenario_contract_id="candidate_v3",
        )


def test_candidate_v4_s5_has_exact_deterministic_eighty_percent_target_layer() -> None:
    assert sum(
        bool(_resolve_v4(seed)["target_background_condition_sampled"])
        for seed in range(10000)
    ) == 8000
    for start in range(100):
        assert sum(
            bool(_resolve_v4(seed)["target_background_condition_sampled"])
            for seed in range(start, start + 10)
        ) == 8


def test_candidate_v4_target_and_control_geometry_are_disjoint() -> None:
    target_patterns = set()
    control_patterns = set()
    for seed in range(200):
        value = _resolve_v4(seed)
        pattern = (value["left_relation"], value["right_relation"])
        if value["target_background_condition_sampled"]:
            target_patterns.add(pattern)
            assert set(pattern) == {"ahead", "behind"}
            for side, relation in zip(("left", "right"), pattern):
                offset = value[f"{side}_offset_m"]
                if relation == "ahead":
                    assert 19.0 <= offset <= 22.0
                else:
                    assert -20.0 <= offset <= -12.0
            assert value["lead_pressure_bucket"] == "high"
        else:
            control_patterns.add(pattern)
            assert pattern in {("ahead", "ahead"), ("behind", "behind")}
            assert value["lead_pressure_bucket"] == "moderate"
    assert target_patterns == {("ahead", "behind"), ("behind", "ahead")}
    assert control_patterns == {("ahead", "ahead"), ("behind", "behind")}


def test_s6_fixed_seeds_cover_both_internal_platoon_gaps() -> None:
    values = {seed: _resolve("S6_background_merge_in", seed) for seed in SEEDS}
    gaps = {value["target_gap_id"] for value in values.values()}
    assert gaps == {"agent0-agent1", "agent1-agent2"}
    assert values[23]["merge_actor_speed_km_h"] == pytest.approx(22.6)
    assert values[47]["merge_actor_speed_km_h"] == pytest.approx(21.5)
    assert -0.8 <= values[47]["response_timing_compensation_s"] <= -0.5


def test_s7_severity_controls_parallel_constraint_actor_count() -> None:
    expected_constraints = {"low": 1, "medium": 2, "high": 3}
    for seed in SEEDS:
        value = _resolve("S7_ego_merge_from_ramp", seed)
        expected = expected_constraints[value["severity_bucket"]]
        assert value["timed_stream_actor_count"] == 4
        assert value["parallel_constraint_actor_count"] == expected
        assert value["actor_count"] == 4 + expected
        assert len(value["mainline_actor_speeds_km_h"]) == 4
        assert len(value["parallel_constraint_actor_speeds_km_h"]) == expected
        speed_bounds = {
            "low": (28.0, 31.0),
            "medium": (24.0, 28.0),
            "high": (24.0, 28.0),
        }[value["severity_bucket"]]
        assert all(
            speed_bounds[0] <= speed <= speed_bounds[1]
            for speed in value["parallel_constraint_actor_speeds_km_h"]
        )
        distance_bounds = (18.0, 21.0)
        assert distance_bounds[0] <= value["ego_distance_to_merge_point_m"] <= distance_bounds[1]
        assert 45.0 <= value["usable_mainline_gap_m"] <= 70.0
        actor_boundary_margin_m = (
            3.0 if value["severity_bucket"] == "high" else 20.0
        )
        physical_gap_floor_m = (
            3.0 * 5.74
            + 2.0 * value["initial_platoon_bumper_gap_m"]
            + actor_boundary_margin_m
        )
        assert physical_gap_floor_m <= value["usable_mainline_gap_m"]


def test_s8_and_s9_ranges_match_functional_contract() -> None:
    s8_constraint_counts = set()
    s8_target_gaps = set()
    for seed in SEEDS:
        s8 = _resolve("S8_ego_exit_to_ramp", seed)
        s8_constraint_counts.add(s8["exit_constraint_actor_count"])
        s8_target_gaps.add(s8["exit_constraint_target_gap_id"])
        assert 1 <= s8["exit_constraint_actor_count"] <= 3
        assert len(s8["exit_constraint_actor_speeds_km_h"]) == s8[
            "exit_constraint_actor_count"
        ]
        assert s8["exit_constraint_actor_speeds_km_h"][0] == s8[
            "ego_initial_speed_km_h"
        ]
        assert all(
            18.0 <= speed <= 23.0
            for speed in s8["exit_constraint_actor_speeds_km_h"][1:]
        )
        assert s8["ego_initial_speed_km_h"] == 25.0
        assert 55.0 <= s8["ego_distance_to_diverge_m"] <= 57.0
        assert 58.0 <= s8["mandatory_lane_change_remaining_distance_m"] <= 60.0
        assert 20.0 <= s8["exit_lane_rear_actor_speed_km_h"] <= 28.0
        assert 16.0 <= s8["exit_lane_front_actor_speed_km_h"] <= 23.0
        assert (
            s8["exit_lane_rear_actor_speed_km_h"]
            < s8["exit_lane_front_actor_speed_km_h"]
        )
        assert 45.0 <= s8["usable_exit_lane_gap_m"] <= 75.0
        assert s8["usable_exit_lane_gap_m"] >= 74.0
        s9 = _resolve("S9_narrow_channel_negotiation", seed)
        assert s9["source_lane_id"] == 1 and s9["bypass_lane_id"] == 0
        assert s9["initial_platoon_bumper_gap_m"] == 16.5
        assert s9["narrow_section_block_id"] == "c3"
        assert s9["post_narrow_return_lane_id"] == 1
        assert s9["left_bypass_trigger_actor_count"] == 1
        assert s9["designated_split_actor_role"] == "left_bypass_split_actor"
        assert s9["designated_split_actor_target_gap_id"] in {
            "agent0-agent1",
            "agent1-agent2",
        }
        assert 12.0 <= s9["designated_split_actor_speed_km_h"] <= 22.0
        assert s9["additional_left_bypass_trigger_actor_count"] == 0
        assert 21.5 <= s9["ego_initial_speed_km_h"] <= 22.0
        assert 15.0 <= s9["agent0_to_blocker_bumper_gap_m"] <= 28.0
        assert 0.0 <= s9["blocker_speed_km_h"] <= 0.5
        assert 3.9 <= s9["predicted_blocker_ttc_s"] <= 4.0
        assert 45.0 <= s9["usable_bypass_gap_m"] <= 70.0
        assert 40.0 <= s9["ego_distance_to_narrow_entry_m"] <= 50.0
        assert 8.0 <= s9["latest_lane_change_completion_before_blocker_m"] <= 10.0
    assert len(s8_constraint_counts) >= 2
    assert s8_target_gaps == {"agent0-agent1", "agent1-agent2"}
