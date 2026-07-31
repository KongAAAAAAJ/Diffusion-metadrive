from __future__ import annotations

from evaluation.audit_bev_expert_chain import (
    AUDIT_SCENARIOS,
    _crash_reason,
    _episode_spec,
)


def test_audit_scope_excludes_only_s8_from_primary_scenarios():
    assert [value[0][:2] for value in AUDIT_SCENARIOS] == [
        "S5",
        "S6",
        "S7",
        "S9",
    ]


def test_crash_reasons_preserve_physical_failure_type():
    assert (
        _crash_reason(
            {"crash": True, "crash_vehicle": True},
            "agent0",
        )
        == "crash_vehicle:agent0"
    )
    assert (
        _crash_reason(
            {"crash": True, "crash_sidewalk": True},
            "agent1",
        )
        == "crash_sidewalk:agent1"
    )
    assert (
        _crash_reason({"out_of_road": True}, "agent2")
        == "out_of_road:agent2"
    )


def test_episode_spec_uses_explicit_audit_seed_and_frozen_route():
    first = _episode_spec("S6_background_merge_in", "R6_mainline_merge_approach", 17)
    second = _episode_spec("S6_background_merge_in", "R6_mainline_merge_approach", 17)

    assert first == second
    assert first.spawn_seed == 17
    assert first.local_route == "R6_mainline_merge_approach"
    assert first.traffic_density == 0.0
