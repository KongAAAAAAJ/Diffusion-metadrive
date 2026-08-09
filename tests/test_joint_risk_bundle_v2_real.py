from __future__ import annotations

from collections import Counter

import numpy as np

from expert_dataset.joint_risk_bundle_v2_real import (
    CONTROL_ANCHOR_STEP,
    PARTITIONS,
    SCENARIO_FAMILIES,
    SEVERITIES,
    RealV2EpisodeSpec,
    _episode_specs,
    _online_observation_arrays,
    _scenario_contract,
    _topology,
)
from scenarios.definitions import get_scenario_definition


def test_real_smoke_matrix_is_balanced_and_seeds_do_not_leak() -> None:
    seeds: set[int] = set()
    for partition in PARTITIONS:
        specs = _episode_specs(partition)
        assert len(specs) == 8
        assert Counter((row.scenario_family, row.severity) for row in specs) == Counter(
            (family, severity)
            for family in SCENARIO_FAMILIES
            for severity in SEVERITIES
        )
        assert sum(row.traffic_density > 0.0 for row in specs) == 4
        assert all(row.spawn_seed not in seeds for row in specs)
        seeds.update(row.spawn_seed for row in specs)
        if partition != "id":
            assert {row.split for row in specs} == {"test"}


def test_scenario_contract_is_diagnostic_and_current_state_only() -> None:
    contract = _scenario_contract("id")
    assert contract["diagnostic_only"] is True
    assert contract["eligible_for_formal_training"] is False
    assert contract["online_observation_policy"]["uses_current_state_only"] is True
    assert contract["online_observation_policy"]["uses_future_or_offline_fields"] is False


def test_real_risk_scenarios_use_measured_activation_at_six_seconds() -> None:
    cut_in = get_scenario_definition("RV2_adjacent_lane_cut_in_near_critical")
    cut_vehicle = cut_in.traffic_recipes[0].params["vehicles"][0]
    assert cut_vehicle["policy"] == "forced_cut_in"
    assert cut_vehicle["activation_step"] == CONTROL_ANCHOR_STEP == 60
    assert cut_vehicle["scenario_vehicle_role"] == "risk_v2_entry_source"

    brake = get_scenario_definition("RV2_external_lead_hard_brake_near_critical")
    assert brake.traffic_recipes[0].params["trigger_after_s"] == 6.0
    assert brake.traffic_recipes[0].params["brake_deceleration_range_mps2"] == (
        3.0,
        4.0,
    )

    # Existing S5 remains physically unchanged by the new diagnostic scenarios.
    s5 = get_scenario_definition("S5_hard_brake_lead")
    sides = [row["lane_side"] for row in s5.traffic_recipes[1].params["vehicles"]]
    assert sides == ["left", "right"]


def test_online_observation_is_range_masked_and_zero_filled() -> None:
    spec = _episode_specs("id")[0]
    state = np.zeros((2, 5, 8), dtype=np.float32)
    state[:, :3, 0] = np.asarray((0.0, -10.0, -20.0), dtype=np.float32)
    state[:, 3, 0] = 50.0
    state[:, 4, 0] = 200.0
    arrays = {
        "actor_state": state,
        "actor_valid_mask": np.ones((2, 5), dtype=np.bool_),
        "actor_state_valid_mask": np.ones((2, 5, 8), dtype=np.bool_),
    }
    result = _online_observation_arrays(arrays, spec)
    assert result["actor_observation_mask"][:, 3].all()
    assert not result["actor_observation_mask"][:, 4].any()
    assert np.all(result["observed_actor_state"][:, 4] == 0.0)
    assert not result["observed_actor_state_valid_mask"][:, 4].any()


def test_topology_relations_resolve_and_work_zone_changes_hash() -> None:
    lane_rows = [
        {"lane_id": "L000", "source_lane_id": '["a","b",0]', "lane_type": "unknown"},
        {"lane_id": "L001", "source_lane_id": '["a","b",1]', "lane_type": "unknown"},
        {"lane_id": "L002", "source_lane_id": '["b","c",0]', "lane_type": "unknown"},
    ]
    base = _episode_specs("id")[0]
    construction = RealV2EpisodeSpec(
        **{
            **base.__dict__,
            "scenario_family": "construction_zone_forced_merge",
            "severity": "near_critical",
        }
    )
    normal = _topology(base, lane_rows, 161, 65)
    work_zone = _topology(construction, lane_rows, 161, 65)
    lane_ids = {row["lane_id"] for row in lane_rows}
    assert all(
        edge["source_lane_id"] in lane_ids and edge["target_lane_id"] in lane_ids
        for edge in normal["lane_relations"]
    )
    assert normal["canonical_hash"] != work_zone["canonical_hash"]
    assert any(
        row["state"] == "work_zone"
        for row in work_zone["time_varying_lane_states"]
    )
