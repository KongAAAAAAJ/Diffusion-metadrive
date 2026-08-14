from evaluation.run_s5_s9_candidate_revision import (
    SCENARIO_HORIZONS,
    _incidental_background_gate,
    _s5_physical_behavior_gate,
    _s7_parallel_constraint_gate,
    _s8_physical_behavior_gate,
    _s9_physical_behavior_gate,
)


def test_s6_and_s9_runner_horizons_match_candidate_memory() -> None:
    assert SCENARIO_HORIZONS == {
        "S6_background_merge_in": 260,
        "S9_narrow_channel_negotiation": 800,
    }


def _row(seed, behavior, *, directions=None, steps=None):
    return {
        "seed": seed,
        "conflict_evidence": {
            "observed_behavior_class": behavior,
            "real_lane_change_completed": directions is not None,
            "lane_change_direction_by_agent": directions or {},
            "lane_change_completion_steps": steps or {},
        },
    }


def test_s5_physical_gate_accepts_keep_plus_asynchronous_mixed_lane_change():
    rows = [
        _row(
            17,
            "temporary_formation_release_and_recovery",
            directions={"agent0": "left", "agent1": "left", "agent2": "right"},
            steps={"agent0": 54, "agent1": 61, "agent2": 64},
        ),
        _row(23, "keep_emergency_braking"),
        _row(31, "keep_emergency_braking"),
        _row(47, "keep_emergency_braking"),
        _row(59, "keep_emergency_braking"),
    ]

    result = _s5_physical_behavior_gate(rows)

    assert result["passed"] is True
    assert result["physical_lane_change_seeds"] == [17]
    assert result["asynchronous_response_seeds"] == [17]


def test_s5_physical_gate_rejects_all_keep_or_all_synchronous_lane_change():
    all_keep = [
        _row(seed, "keep_emergency_braking")
        for seed in (17, 23, 31, 47, 59)
    ]
    assert _s5_physical_behavior_gate(all_keep)["passed"] is False

    all_synchronous = [
        _row(
            seed,
            "coordinated_lane_change",
            directions={"agent0": "left", "agent1": "left", "agent2": "left"},
            steps={"agent0": 50, "agent1": 50, "agent2": 50},
        )
        for seed in (17, 23, 31, 47, 59)
    ]
    assert _s5_physical_behavior_gate(all_synchronous)["passed"] is False


def test_s8_gate_requires_front_behind_straddling_and_diverse_async_order() -> None:
    orders = (
        ("agent0", "agent1", "agent2"),
        ("agent0", "agent2", "agent1"),
        ("agent0", "agent1", "agent2"),
        ("agent0", "agent2", "agent1"),
        ("agent0", "agent1", "agent2"),
    )
    rows = []
    for seed, order in zip((17, 23, 31, 47, 59), orders):
        rows.append(
            {
                "seed": seed,
                "conflict_evidence": {
                    "constraint_straddled_by_lane_changes": True,
                    "non_simultaneous_right_lane_changes": True,
                    "constraint_relation_at_lane_change_by_agent": {
                        "agent0": "ahead",
                        "agent1": "behind",
                        "agent2": "behind",
                    },
                    "observed_lane_change_behavior_class": "gap:" + "-".join(order),
                },
            }
        )

    result = _s8_physical_behavior_gate(rows)

    assert result["passed"] is True
    rows[0]["conflict_evidence"]["constraint_straddled_by_lane_changes"] = False
    assert _s8_physical_behavior_gate(rows)["passed"] is False


def test_s7_gate_requires_parallel_constraints_and_async_reassembly() -> None:
    rows = [
        {
            "seed": seed,
            "conflict_evidence": {
                "parallel_constraint_declared_count": count,
                "parallel_constraint_realized_count": count,
                "parallel_constraint_initial_region_valid": True,
                "parallel_constraint_roles_present": True,
                "physical_split_observed": True,
                "non_simultaneous_mainline_entries": True,
                "formation_recovered_after_merge": True,
            },
        }
        for seed, count in zip((17, 23, 31, 47, 59), (3, 1, 1, 2, 3))
    ]

    assert _s7_parallel_constraint_gate(rows)["passed"] is True
    rows[0]["conflict_evidence"]["parallel_constraint_initial_region_valid"] = False
    assert _s7_parallel_constraint_gate(rows)["passed"] is False


def test_s9_gate_requires_same_actor_split_post_channel_return_and_recovery() -> None:
    rows = []
    for seed in (17, 23, 31, 47, 59):
        rows.append(
            {
                "seed": seed,
                "conflict_evidence": {
                    "left_bypass_trigger_actor_declared_count": 1,
                    "left_bypass_trigger_actor_realized_count": 1,
                    "designated_split_actor_initial_region_valid": True,
                    "s9_actor_roles_present": True,
                    "non_simultaneous_left_lane_changes": True,
                    "split_actor_straddled_by_left_completions": True,
                    "split_actor_relation_at_left_completion_by_agent": {
                        "agent0": "ahead",
                        "agent1": "behind",
                        "agent2": "behind",
                    },
                    "causal_split_actor_interaction_observed": True,
                    "left_completion_clearance_satisfied": True,
                    "return_before_narrow_section_clear_observed": False,
                    "formation_recovered_after_return": True,
                },
                "route_completion": {
                    "all_agents_passed_blocker": True,
                    "all_agents_traversed_narrow_section": True,
                    "all_agents_returned_to_original_lane": True,
                },
            }
        )

    assert _s9_physical_behavior_gate(rows)["passed"] is True
    rows[0]["conflict_evidence"][
        "return_before_narrow_section_clear_observed"
    ] = True
    assert _s9_physical_behavior_gate(rows)["passed"] is False


def test_incidental_background_gate_requires_exact_three_to_six_realizations() -> None:
    rows = [
        {
            "scenario_id": "S8_ego_exit_to_ramp",
            "seed": seed,
            "conflict_evidence": {
                "incidental_background_declared_count": count,
                "incidental_background_realized_count": count,
            },
        }
        for seed, count in zip((17, 23, 31, 47, 59), (3, 4, 5, 6, 3))
    ]
    assert _incidental_background_gate(rows)["passed"] is True
    rows[0]["conflict_evidence"]["incidental_background_realized_count"] = 2
    assert _incidental_background_gate(rows)["passed"] is False
