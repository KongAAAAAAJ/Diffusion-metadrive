from evaluation.run_s5_s9_candidate_revision import _s5_physical_behavior_gate


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
