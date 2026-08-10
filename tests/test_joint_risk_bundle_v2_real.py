from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

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
    eligible_anchor_violation,
)
from expert_dataset.riskentry_sidecar_adapter import SidecarActorRecord
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

    merge = get_scenario_definition("RV2_on_ramp_external_merge_near_critical")
    assert merge.ego_spawn_longitude_m == (112.0, 116.0)
    assert merge.traffic_recipes[0].params["merge_activation_step"] == 30

    # Existing S5 remains physically unchanged by the new diagnostic scenarios.
    s5 = get_scenario_definition("S5_hard_brake_lead")
    sides = [row["lane_side"] for row in s5.traffic_recipes[1].params["vehicles"]]
    assert sides == ["left", "right"]


def test_topology_hard_brake_pair_pre_spawns_same_route_actor() -> None:
    near = get_scenario_definition(
        "RV2_external_lead_hard_brake_near_critical_topology_ood"
    )
    control = get_scenario_definition(
        "RV2_external_lead_hard_brake_control_topology_ood"
    )

    assert near.allowed_local_routes == control.allowed_local_routes == (
        "R2_entry_curve",
    )
    assert near.traffic_recipes[0].operation == "inject_background_vehicle"
    assert control.traffic_recipes[0].operation == "inject_background_vehicle"
    for scenario in (near, control):
        params = scenario.traffic_recipes[0].params
        assert params["trigger_on_start"] is True
        assert params["reference_kind"] == "ego_lane"
        assert params["spawn_longitude"] == 22.0
        assert params["target_speed_kmh"] == 24.0
    assert near.traffic_recipes[0].params["scenario_vehicle_role"] == (
        "risk_v2_entry_source"
    )
    assert control.traffic_recipes[0].params["scenario_vehicle_role"] == (
        "risk_v2_control_source"
    )
    assert near.traffic_recipes[1].operation == "hard_brake_lead"
    assert near.traffic_recipes[1].params["trigger_after_s"] == 6.0
    assert len(control.traffic_recipes) == 1


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


def test_topology_id_is_physical_and_partition_independent() -> None:
    lane_rows = [
        {"lane_id": "L000", "source_lane_id": '["a","b",0]', "lane_type": "unknown"},
        {"lane_id": "L001", "source_lane_id": '["b","c",0]', "lane_type": "unknown"},
    ]
    id_spec = _episode_specs("id")[0]
    same_physics = RealV2EpisodeSpec(
        **{**id_spec.__dict__, "partition": "compositional_ood"}
    )
    topology_spec = _episode_specs("topology_ood")[0]
    id_topology = _topology(id_spec, lane_rows, 121, 60)
    same_topology = _topology(same_physics, lane_rows, 121, 60)
    held_out_topology = _topology(topology_spec, lane_rows, 121, 60)
    assert id_topology["topology_id"] == same_topology["topology_id"]
    assert not id_topology["topology_id"].startswith("id_")
    assert id_topology["topology_id"] != held_out_topology["topology_id"]


def test_eligible_anchor_requires_truth_at_all_three_horizons() -> None:
    spec = _episode_specs("id")[1]
    records = tuple(
        SidecarActorRecord(
            actor_index=index,
            actor_id=actor_id,
            source_object_id=actor_id,
            actor_type="platoon" if index < 3 else "external",
            platoon_role=("leader", "middle", "rear")[index] if index < 3 else None,
            first_seen_step=0,
            length_m=5.0,
            width_m=2.0,
        )
        for index, actor_id in enumerate(("P0", "P1", "P2", "V000"))
    )
    rollout = SimpleNamespace(
        sidecar=SimpleNamespace(
            actor_records=records,
            key_actor_ids={"entry_source": "V000"},
        )
    )
    arrays = {
        "step_index": np.arange(121, dtype=np.int64),
        "actor_valid_mask": np.ones((121, 4), dtype=np.bool_),
        "actor_state_valid_mask": np.ones((121, 4, 8), dtype=np.bool_),
        "actor_observation_mask": np.ones((121, 4), dtype=np.bool_),
        "actor_state": np.zeros((121, 4, 8), dtype=np.float32),
    }
    assert eligible_anchor_violation(spec, rollout, arrays, 70) is None
    arrays["actor_valid_mask"][120, 0] = False
    assert (
        eligible_anchor_violation(spec, rollout, arrays, 70)
        == "platoon_truth_invalid_at_120"
    )
