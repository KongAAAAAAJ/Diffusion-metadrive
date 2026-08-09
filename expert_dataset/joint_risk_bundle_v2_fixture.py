"""Generate and verify a diagnostic bundle-v2 cross-project fixture.

The fixture exercises the complete physical hand-off contract without
pretending to be formal research data.  It is intentionally independent from
the bundle-v1 writer so existing pilot/formal-v1 roots remain immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from expert_dataset.collect_joint_bev import JOINT_SAMPLE_DTYPES, JOINT_SAMPLE_SHAPES
from expert_dataset.joint_bev_storage import (
    PACKED_BEV_FIELD,
    STORAGE_FORMAT,
    STORAGE_SCHEMA_VERSION,
    STORED_FIELD_DTYPES,
    STORED_FIELD_SHAPES,
)
from expert_dataset.semantic_bev_codec import (
    pack_semantic_bev,
    packed_bev_contract,
    unpack_semantic_bev,
)


BUNDLE_FORMAT = "metadrive-joint-planning-risk-bundle"
BUNDLE_SCHEMA_VERSION = "2.0.0"
SIDECAR_FORMAT = "riskentry-metadrive-actor-sidecar"
SIDECAR_SCHEMA_VERSION = "2.0.0"
PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "metadrive_joint_risk_bundle_v2.json"
)
FIXTURE_FORMAT = "metadrive-joint-risk-bundle-v2-diagnostic-fixture"
FIXTURE_SCHEMA_VERSION = 1
PARTITIONS = ("id", "compositional_ood", "topology_ood")
SCENARIO_FAMILIES = (
    "adjacent_lane_cut_in",
    "on_ramp_external_merge",
    "external_lead_hard_brake",
    "construction_zone_forced_merge",
)
SEVERITIES = ("control", "near_critical")
TARGET_WINDOWS = {"id": 50_000, "compositional_ood": 10_000, "topology_ood": 10_000}
SIDECAR_ARRAY_DTYPES = {
    "step_index": np.dtype(np.int64),
    "timestamp_s": np.dtype(np.float64),
    "actor_state": np.dtype(np.float32),
    "actor_state_valid_mask": np.dtype(np.bool_),
    "actor_valid_mask": np.dtype(np.bool_),
    "lane_index": np.dtype(np.int32),
    "lane_state": np.dtype(np.float32),
    "lane_valid_mask": np.dtype(np.bool_),
    "base_sample_step_index": np.dtype(np.int64),
    "actor_observation_mask": np.dtype(np.bool_),
    "observed_actor_state": np.dtype(np.float32),
    "observed_actor_state_valid_mask": np.dtype(np.bool_),
    "platoon_comm_available": np.dtype(np.bool_),
    "platoon_comm_delay_s": np.dtype(np.float32),
    "platoon_comm_valid_mask": np.dtype(np.bool_),
}
COMMUNICATION_EDGE_ORDER = ("P0->P1", "P1->P2")
ANCHOR_STEP = 80
TIMELINE_LENGTH = 131


class BundleV2FixtureError(RuntimeError):
    """Raised when the diagnostic fixture violates bundle-v2."""


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, np.ascontiguousarray(array), allow_pickle=False)


def _fixture_scenario_contract(partition: str) -> dict[str, object]:
    topology_id = (
        "topology_ood_curved_work_zone_v1"
        if partition == "topology_ood"
        else "id_straight_three_lane_v1"
    )
    return {
        "format": "metadrive-riskentry-fixture-scenario-contract-v2",
        "schema_version": 1,
        "diagnostic_fixture": True,
        "eligible_for_formal_training": False,
        "benchmark_partition": partition,
        "scenario_families": list(SCENARIO_FAMILIES),
        "severities": list(SEVERITIES),
        "decision_dt_s": 0.1,
        "history_seconds": 2.0,
        "future_seconds": 5.0,
        "near_critical_onset_step": ANCHOR_STEP,
        "online_observation_policy": {
            "id": "ideal_range_80m_no_occlusion_v1",
            "range_m": 80.0,
            "uses_current_state_only": True,
        },
        "communication_policy": {
            "id": "always_available_50ms_fixture_v1",
            "edge_order": list(COMMUNICATION_EDGE_ORDER),
            "delay_s": 0.05,
        },
        "topology_id": topology_id,
    }


def _split_for_fixture(partition: str, family_index: int) -> str:
    if partition != "id":
        return "test"
    return ("train", "train", "val", "test")[family_index]


def _topology(partition: str) -> dict[str, object]:
    topology_id = (
        "topology_ood_curved_work_zone_v1"
        if partition == "topology_ood"
        else "id_straight_three_lane_v1"
    )
    relations: list[dict[str, object]] = []
    lane_states = [
        {"lane_id": "L000", "start_step": 0, "end_step": TIMELINE_LENGTH - 1, "state": "open"}
    ]
    if partition == "topology_ood":
        lane_states.append(
            {"lane_id": "L001", "start_step": 0, "end_step": TIMELINE_LENGTH - 1, "state": "work_zone"}
        )
        relations.extend(
            [
                {"source_lane_id": "L000", "target_lane_id": "L001", "relation": "right"},
                {"source_lane_id": "L001", "target_lane_id": "L000", "relation": "left"},
            ]
        )
    graph = {
        "topology_id": topology_id,
        "lane_relations": relations,
        "time_varying_lane_states": lane_states,
    }
    return {**graph, "canonical_hash": _sha256_payload(graph)}


def _parameter_values(
    partition: str, family_index: int, severity: str
) -> dict[str, float]:
    nonzero_density = (family_index + (severity == "near_critical")) % 2 == 1
    if partition == "id":
        road_friction = 0.85
        sensing_noise = 0.01 if family_index % 2 else 0.0
        aggressiveness = 0.7 if severity == "near_critical" else 0.3
    elif partition == "compositional_ood":
        # Values remain inside ID marginal support, but the tuple is not used in ID.
        road_friction = 0.85
        sensing_noise = 0.01 if family_index % 2 == 0 else 0.0
        aggressiveness = 0.3 if severity == "near_critical" else 0.7
    else:
        road_friction = 0.75
        sensing_noise = 0.02
        aggressiveness = 0.8 if severity == "near_critical" else 0.2
    return {
        "road_friction": road_friction,
        "sensing_noise_std": sensing_noise,
        "traffic_density": 0.02 if nonzero_density else 0.0,
        "external_driver_aggressiveness": aggressiveness,
    }


def _actor_table() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (actor_id, role) in enumerate(
        (("P0", "leader"), ("P1", "middle"), ("P2", "rear"))
    ):
        rows.append(
            {
                "actor_id": actor_id,
                "actor_index": index,
                "actor_type": "platoon",
                "first_seen_step": 0,
                "length_m": 5.74,
                "width_m": 2.3,
                "platoon_role": role,
                "source_object_id": f"agent{index}",
                "mass_kg": 24_000.0,
                "max_brake_deceleration_mps2": 6.5,
                "actuation_delay_s": 0.15,
                "vehicle_parameter_set_id": f"heavy_truck_{actor_id}_fixture_v1",
            }
        )
    rows.append(
        {
            "actor_id": "V000",
            "actor_index": 3,
            "actor_type": "external",
            "first_seen_step": 0,
            "length_m": 4.8,
            "width_m": 1.9,
            "platoon_role": None,
            "source_object_id": "fixture_external_000",
        }
    )
    return rows


def _sidecar_arrays(family_index: int) -> dict[str, np.ndarray]:
    timeline = np.arange(TIMELINE_LENGTH, dtype=np.int64)
    timestamps = timeline.astype(np.float64) * 0.1
    actor_state = np.zeros((TIMELINE_LENGTH, 4, 8), dtype=np.float32)
    starts = np.asarray((0.0, -12.0, -24.0), dtype=np.float32)
    for actor_index, start_x in enumerate(starts):
        actor_state[:, actor_index, 0] = start_x + 4.0 * timestamps
        actor_state[:, actor_index, 3] = 4.0
    actor_state[:, 3, 0] = 10.0 + 3.5 * timestamps
    actor_state[:, 3, 1] = (1.0 + family_index) * np.ones(TIMELINE_LENGTH)
    actor_state[:, 3, 3] = 3.5
    actor_valid = np.ones((TIMELINE_LENGTH, 4), dtype=np.bool_)
    actor_state_valid = np.ones((TIMELINE_LENGTH, 4, 8), dtype=np.bool_)
    lane_index = np.zeros((TIMELINE_LENGTH, 4), dtype=np.int32)
    lane_state = np.zeros((TIMELINE_LENGTH, 4, 4), dtype=np.float32)
    lane_state[..., 0] = actor_state[..., 0]
    lane_state[..., 1] = actor_state[..., 1]
    lane_state[..., 3] = 4.0
    lane_valid = np.ones((TIMELINE_LENGTH, 4), dtype=np.bool_)

    distance = np.linalg.norm(
        actor_state[:, :, None, :2] - actor_state[:, None, :3, :2], axis=-1
    )
    observation_mask = np.min(distance, axis=-1) <= 80.0
    observation_mask[:, :3] = True
    observed_state = np.where(
        observation_mask[..., None], actor_state, np.float32(0.0)
    ).astype(np.float32)
    observed_valid = actor_state_valid & observation_mask[..., None]
    comm_available = np.ones((TIMELINE_LENGTH, 2), dtype=np.bool_)
    comm_delay = np.full((TIMELINE_LENGTH, 2), 0.05, dtype=np.float32)
    comm_valid = np.ones((TIMELINE_LENGTH, 2), dtype=np.bool_)
    return {
        "step_index": timeline,
        "timestamp_s": timestamps,
        "actor_state": actor_state,
        "actor_state_valid_mask": actor_state_valid,
        "actor_valid_mask": actor_valid,
        "lane_index": lane_index,
        "lane_state": lane_state,
        "lane_valid_mask": lane_valid,
        "base_sample_step_index": np.asarray([ANCHOR_STEP], dtype=np.int64),
        "actor_observation_mask": observation_mask.astype(np.bool_),
        "observed_actor_state": observed_state,
        "observed_actor_state_valid_mask": observed_valid.astype(np.bool_),
        "platoon_comm_available": comm_available,
        "platoon_comm_delay_s": comm_delay,
        "platoon_comm_valid_mask": comm_valid,
    }


def _base_arrays() -> dict[str, np.ndarray]:
    bev = np.zeros((1, *JOINT_SAMPLE_SHAPES["bev"]), dtype=np.uint8)
    bev[:, :, 0] = np.uint8(255)
    bev[:, :, 1, 127:129] = np.uint8(128)
    bev[:, :, 2, 127:129] = np.uint8(255)
    arrays: dict[str, np.ndarray] = {PACKED_BEV_FIELD: pack_semantic_bev(bev)}
    for name, shape in JOINT_SAMPLE_SHAPES.items():
        if name == "bev":
            continue
        arrays[name] = np.zeros((1, *shape), dtype=JOINT_SAMPLE_DTYPES[name])
    arrays["ego_pose_global"][0, :, 0] = np.asarray((0.0, -12.0, -24.0))
    arrays["agent_role"][0] = np.asarray((0, 1, 2), dtype=np.int64)
    arrays["mode_valid_mask"][0, :, 9] = True
    arrays["gt_mode"][0] = np.asarray((9, 9, 9), dtype=np.int64)
    return arrays


def _base_contract(partition: str, fingerprint: str) -> dict[str, object]:
    split_assignment = (
        {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1, "seed": 17}
        if partition == "id"
        else {"train_ratio": 0.0, "val_ratio": 0.0, "test_ratio": 1.0, "seed": 17}
    )
    sample_contract = {
        name: {"shape": list(shape), "dtype": str(JOINT_SAMPLE_DTYPES[name])}
        for name, shape in JOINT_SAMPLE_SHAPES.items()
    }
    physical = {
        name: {"shape": list(shape), "dtype": str(STORED_FIELD_DTYPES[name])}
        for name, shape in STORED_FIELD_SHAPES.items()
    }
    return {
        "schema_version": STORAGE_SCHEMA_VERSION,
        "format": STORAGE_FORMAT,
        "joint_first": True,
        "sample_contract": sample_contract,
        "physical_storage_contract": physical,
        "packed_semantic_bev": packed_bev_contract(),
        "split_assignment": split_assignment,
        "dataset_fingerprint": fingerprint,
    }


def _sidecar_contract(base_fingerprint: str) -> dict[str, object]:
    return {
        "format": SIDECAR_FORMAT,
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "base_format": STORAGE_FORMAT,
        "base_schema_version": STORAGE_SCHEMA_VERSION,
        "base_dataset_fingerprint": base_fingerprint,
        "split_policy": "episode_frozen_partition_policy",
        "timeline_policy": "all_decision_boundaries_including_terminal_post_step",
        "missing_value_policy": "zero_plus_explicit_mask",
        "actor_state_channels": [
            "world_x_m",
            "world_y_m",
            "heading_rad",
            "velocity_x_mps",
            "velocity_y_mps",
            "acceleration_x_mps2",
            "acceleration_y_mps2",
            "yaw_rate_radps",
        ],
        "lane_state_channels": [
            "lane_s_m",
            "lane_lateral_m",
            "lane_heading_error_rad",
            "lane_width_m",
        ],
        "communication_edge_order": list(COMMUNICATION_EDGE_ORDER),
        "array_contract": {
            name: {"dtype": str(dtype)} for name, dtype in SIDECAR_ARRAY_DTYPES.items()
        },
        "ground_truth_and_online_observation_separated": True,
    }


def _entry_event(severity: str) -> dict[str, object]:
    if severity == "control":
        return {
            "edge_id": "none",
            "source_actor_id": None,
            "target_actor_id": None,
            "onset_step": None,
            "resolution_step": None,
            "resolution_status": "not_applicable",
        }
    return {
        "edge_id": "V000->P1",
        "source_actor_id": "V000",
        "target_actor_id": "P1",
        "onset_step": ANCHOR_STEP,
        "resolution_step": 100,
        "resolution_status": "resolved",
    }


def _episode_metadata(
    *,
    episode_index: int,
    split: str,
    partition: str,
    family: str,
    family_index: int,
    severity: str,
    base_fingerprint: str,
    scenario_contract_sha256: str,
) -> dict[str, object]:
    topology = _topology(partition)
    parameters = _parameter_values(partition, family_index, severity)
    parameter_id = f"{partition}_{family}_{severity}_{_sha256_payload(parameters)[:12]}"
    lane_rows = [
        {
            "lane_id": "L000",
            "lane_index": 0,
            "lane_type": "mainline",
            "source_lane_id": "fixture_lane_0",
        }
    ]
    if partition == "topology_ood":
        lane_rows.append(
            {
                "lane_id": "L001",
                "lane_index": 1,
                "lane_type": "merge",
                "source_lane_id": "fixture_work_zone_lane_1",
            }
        )
    return {
        "format": SIDECAR_FORMAT,
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "complete": True,
        "diagnostic_fixture": True,
        "eligible_for_formal_training": False,
        "episode_index": episode_index,
        "benchmark_partition": partition,
        "split": split,
        "base_dataset_fingerprint": base_fingerprint,
        "decision_dt_s": 0.1,
        "scenario_id": family,
        "scenario_family": family,
        "severity": severity,
        "topology_id": topology["topology_id"],
        "parameter_combination_id": parameter_id,
        "matched_pair_id": f"{partition}_{family}_pair_fixture_v1",
        "scenario_realization_status": "realized",
        "local_route": f"fixture_{topology['topology_id']}",
        "spawn_seed": 10_000 * PARTITIONS.index(partition) + episode_index + 17,
        "scenario_parameters": {
            **parameters,
            "scenario_contract_sha256": scenario_contract_sha256,
        },
        "actors": _actor_table(),
        "lanes": lane_rows,
        "topology": topology,
        "entry_event": _entry_event(severity),
        "key_actor_ids": {} if severity == "control" else {"entry_source": "V000"},
        "events": [],
        "retention": {"outcome": "success", "kept_despite_dangerous_outcome": False},
        "eligible_anchor_steps": [ANCHOR_STEP],
        "observation_policy_id": "ideal_range_80m_no_occlusion_v1",
        "communication_policy_id": "always_available_50ms_fixture_v1",
    }


def _episode_specs(partition: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for family_index, family in enumerate(SCENARIO_FAMILIES):
        split = _split_for_fixture(partition, family_index)
        for severity in SEVERITIES:
            result.append(
                {
                    "episode_index": len(result),
                    "split": split,
                    "family": family,
                    "family_index": family_index,
                    "severity": severity,
                }
            )
    return result


def _write_component_manifests(
    *,
    root: Path,
    format_name: str,
    schema_version: object,
    allowed_splits: Sequence[str],
    rows_by_split: Mapping[str, list[dict[str, object]]],
    sidecar: bool,
) -> None:
    for split in allowed_splits:
        rows = rows_by_split.get(split, [])
        payload: dict[str, object] = {
            "format": format_name,
            "schema_version": schema_version,
            "split": split,
            "episode_count": len(rows),
            "episodes": rows,
        }
        if sidecar:
            payload["raw_steps"] = sum(int(row["raw_steps"]) for row in rows)
            payload["base_samples"] = sum(int(row["base_samples"]) for row in rows)
        else:
            payload["joint_samples"] = sum(int(row["joint_samples"]) for row in rows)
        _write_json(root / split / "manifest.json", payload)
        (root / split / "episodes").mkdir(parents=True, exist_ok=True)


def generate_fixture(output_root: Path | str) -> dict[str, object]:
    """Create three new diagnostic roots; existing paths are never overwritten."""

    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise BundleV2FixtureError(f"fixture output root is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    protocol_sha = _sha256_file(PROTOCOL_PATH)
    roots: dict[str, str] = {}
    for partition in PARTITIONS:
        root = output / f"riskentry_fixture_{partition}_v2"
        root.mkdir()
        roots[partition] = str(root)
        scenario_contract = _fixture_scenario_contract(partition)
        scenario_sha = _sha256_payload(scenario_contract)
        _write_json(root / "scenario_contract.json", scenario_contract)
        base_fingerprint = _sha256_payload(
            {
                "fixture": FIXTURE_FORMAT,
                "partition": partition,
                "scenario_contract_sha256": scenario_sha,
                "component": "base",
            }
        )
        sidecar_contract = _sidecar_contract(base_fingerprint)
        sidecar_fingerprint = _sha256_payload(sidecar_contract)
        base_root = root / "platoon_joint_bev"
        sidecar_root = root / "riskentry_actor_sidecar"
        _write_json(base_root / "dataset_contract.json", _base_contract(partition, base_fingerprint))
        _write_json(sidecar_root / "dataset_contract.json", sidecar_contract)

        allowed_splits = ("train", "val", "test") if partition == "id" else ("test",)
        base_rows: dict[str, list[dict[str, object]]] = {name: [] for name in allowed_splits}
        sidecar_rows: dict[str, list[dict[str, object]]] = {name: [] for name in allowed_splits}
        index_rows: list[dict[str, object]] = []
        parameter_ids: list[str] = []
        topology_ids: list[str] = []
        for spec in _episode_specs(partition):
            episode_index = int(spec["episode_index"])
            split = str(spec["split"])
            family = str(spec["family"])
            family_index = int(spec["family_index"])
            severity = str(spec["severity"])
            metadata = _episode_metadata(
                episode_index=episode_index,
                split=split,
                partition=partition,
                family=family,
                family_index=family_index,
                severity=severity,
                base_fingerprint=base_fingerprint,
                scenario_contract_sha256=scenario_sha,
            )
            episode_name = f"episode_{episode_index:08d}"
            side_episode = sidecar_root / split / "episodes" / episode_name
            _write_json(side_episode / "episode.json", metadata)
            for name, array in _sidecar_arrays(family_index).items():
                _write_npy(side_episode / f"{name}.npy", array)

            selected_steps = [ANCHOR_STEP]
            attributes = {
                "scenario_id": family,
                "local_route": metadata["local_route"],
                "spawn_seed": metadata["spawn_seed"],
                "sidecar_dataset_fingerprint": sidecar_fingerprint,
                "selected_sample_steps": selected_steps,
                "benchmark_partition": partition,
                "scenario_family": family,
                "severity": severity,
                "diagnostic_fixture": True,
                "eligible_for_formal_training": False,
            }
            base_episode = base_root / split / "episodes" / episode_name
            _write_json(
                base_episode / "episode.json",
                {
                    "format": STORAGE_FORMAT,
                    "schema_version": STORAGE_SCHEMA_VERSION,
                    "complete": True,
                    "episode_index": episode_index,
                    "split": split,
                    "joint_samples": 1,
                    "attributes": attributes,
                },
            )
            for name, array in _base_arrays().items():
                _write_npy(base_episode / f"{name}.npy", array)

            base_rows[split].append(
                {
                    "episode_index": episode_index,
                    "directory": episode_name,
                    "joint_samples": 1,
                    "attributes": attributes,
                }
            )
            sidecar_rows[split].append(
                {
                    "episode_index": episode_index,
                    "split": split,
                    "directory": episode_name,
                    "raw_steps": TIMELINE_LENGTH,
                    "actor_count": 4,
                    "base_samples": 1,
                    "outcome": "success",
                }
            )
            index_rows.append(
                {
                    "episode_index": episode_index,
                    "split": split,
                    "scenario_id": family,
                    "local_route": metadata["local_route"],
                    "spawn_seed": metadata["spawn_seed"],
                    "base_status": "committed",
                    "base_rejection_reason": None,
                    "sidecar_status": "committed",
                    "sidecar_rejection_reason": None,
                    "raw_steps": TIMELINE_LENGTH,
                    "base_samples": 1,
                    "outcome": "success",
                }
            )
            parameter_ids.append(str(metadata["parameter_combination_id"]))
            topology_ids.append(str(metadata["topology_id"]))

        _write_component_manifests(
            root=base_root,
            format_name=STORAGE_FORMAT,
            schema_version=STORAGE_SCHEMA_VERSION,
            allowed_splits=allowed_splits,
            rows_by_split=base_rows,
            sidecar=False,
        )
        _write_component_manifests(
            root=sidecar_root,
            format_name=SIDECAR_FORMAT,
            schema_version=SIDECAR_SCHEMA_VERSION,
            allowed_splits=allowed_splits,
            rows_by_split=sidecar_rows,
            sidecar=True,
        )
        (root / "bundle_episode_index.jsonl").write_text(
            "".join(_canonical_json(row) + "\n" for row in index_rows),
            encoding="utf-8",
        )
        split_manifest = {
            split: [int(row["episode_index"]) for row in rows]
            for split, rows in base_rows.items()
        }
        manifest = {
            "format": BUNDLE_FORMAT,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "diagnostic_fixture": True,
            "eligible_for_formal_training": False,
            "dataset_instance_id": f"diagnostic_bundle_v2_{partition}_fixture_v1",
            "benchmark_partition": partition,
            "bundle_protocol_sha256": protocol_sha,
            "scenario_contract_sha256": scenario_sha,
            "base_dataset_fingerprint": base_fingerprint,
            "sidecar_dataset_fingerprint": sidecar_fingerprint,
            "parameter_space_id": "diagnostic_parameter_space_fixture_v1",
            "parameter_tuple_set_sha256": _sha256_payload(sorted(parameter_ids)),
            "topology_set_id": f"diagnostic_{partition}_topology_set_v1",
            "topology_id_set_sha256": _sha256_payload(sorted(set(topology_ids))),
            "target_eligible_anchor_windows": TARGET_WINDOWS[partition],
            "eligible_anchor_window_count": len(index_rows),
            "counting_unit": "eligible_anchor_windows_with_2s_history_and_valid_5s_future",
            "scenario_catalog": list(SCENARIO_FAMILIES),
            "episode_split_manifest": split_manifest,
            "base_directory": "platoon_joint_bev",
            "sidecar_directory": "riskentry_actor_sidecar",
        }
        _write_json(root / "dataset_bundle_manifest.json", manifest)

    summary = {
        "format": FIXTURE_FORMAT,
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "diagnostic_fixture": True,
        "eligible_for_formal_training": False,
        "bundle_protocol_path": str(PROTOCOL_PATH),
        "bundle_protocol_sha256": protocol_sha,
        "roots": roots,
        "episodes_per_partition": len(SCENARIO_FAMILIES) * len(SEVERITIES),
        "eligible_windows_per_partition": len(SCENARIO_FAMILIES) * len(SEVERITIES),
    }
    _write_json(output / "fixture_summary.json", summary)
    return verify_fixture(output)


def _load_memmap(path: Path, dtype: np.dtype) -> np.memmap:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise BundleV2FixtureError(f"unable to mmap {path}") from exc
    if not isinstance(array, np.memmap) or array.dtype != dtype:
        raise BundleV2FixtureError(f"array mmap/dtype mismatch: {path}")
    return array


def _verify_sidecar_episode(path: Path, partition: str) -> dict[str, object]:
    expected_files = {"episode.json", *(f"{name}.npy" for name in SIDECAR_ARRAY_DTYPES)}
    if {item.name for item in path.iterdir()} != expected_files:
        raise BundleV2FixtureError("sidecar physical file set mismatch")
    metadata = json.loads((path / "episode.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise BundleV2FixtureError("sidecar schema version mismatch")
    if metadata.get("benchmark_partition") != partition:
        raise BundleV2FixtureError("sidecar partition mismatch")
    arrays = {
        name: _load_memmap(path / f"{name}.npy", dtype)
        for name, dtype in SIDECAR_ARRAY_DTYPES.items()
    }
    expected_shapes = {
        "step_index": (TIMELINE_LENGTH,),
        "timestamp_s": (TIMELINE_LENGTH,),
        "actor_state": (TIMELINE_LENGTH, 4, 8),
        "actor_state_valid_mask": (TIMELINE_LENGTH, 4, 8),
        "actor_valid_mask": (TIMELINE_LENGTH, 4),
        "lane_index": (TIMELINE_LENGTH, 4),
        "lane_state": (TIMELINE_LENGTH, 4, 4),
        "lane_valid_mask": (TIMELINE_LENGTH, 4),
        "base_sample_step_index": (1,),
        "actor_observation_mask": (TIMELINE_LENGTH, 4),
        "observed_actor_state": (TIMELINE_LENGTH, 4, 8),
        "observed_actor_state_valid_mask": (TIMELINE_LENGTH, 4, 8),
        "platoon_comm_available": (TIMELINE_LENGTH, 2),
        "platoon_comm_delay_s": (TIMELINE_LENGTH, 2),
        "platoon_comm_valid_mask": (TIMELINE_LENGTH, 2),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise BundleV2FixtureError(f"{name} shape mismatch")
    if not np.array_equal(arrays["step_index"], np.arange(TIMELINE_LENGTH)):
        raise BundleV2FixtureError("sidecar timeline is not contiguous")
    if not np.allclose(arrays["timestamp_s"], np.arange(TIMELINE_LENGTH) * 0.1):
        raise BundleV2FixtureError("sidecar timestamps mismatch")
    if not np.isfinite(arrays["actor_state"]).all() or not np.isfinite(
        arrays["observed_actor_state"]
    ).all():
        raise BundleV2FixtureError("sidecar actor arrays are non-finite")
    observed_mask = np.asarray(arrays["actor_observation_mask"])
    observed = np.asarray(arrays["observed_actor_state"])
    observed_valid = np.asarray(arrays["observed_actor_state_valid_mask"])
    if np.any(observed[~observed_valid] != 0.0):
        raise BundleV2FixtureError("unobserved state must be zero")
    if np.any(observed_valid & ~observed_mask[..., None]):
        raise BundleV2FixtureError("unobserved actor has valid observed channels")
    if not np.all(arrays["platoon_comm_valid_mask"]):
        raise BundleV2FixtureError("fixture communication facts must be valid")
    if np.any(arrays["platoon_comm_delay_s"] < 0.0):
        raise BundleV2FixtureError("communication delay must be non-negative")
    anchors = np.asarray(arrays["base_sample_step_index"])
    if np.any(anchors < 20) or np.any(anchors + 50 >= TIMELINE_LENGTH):
        raise BundleV2FixtureError("fixture anchor lacks 2 s history or 5 s future")
    topology = metadata.get("topology")
    if not isinstance(topology, Mapping):
        raise BundleV2FixtureError("topology metadata is missing")
    graph = {
        "topology_id": topology.get("topology_id"),
        "lane_relations": topology.get("lane_relations"),
        "time_varying_lane_states": topology.get("time_varying_lane_states"),
    }
    if topology.get("canonical_hash") != _sha256_payload(graph):
        raise BundleV2FixtureError("topology canonical hash mismatch")
    lane_ids = {row["lane_id"] for row in metadata.get("lanes", [])}
    for relation in topology.get("lane_relations", []):
        if relation.get("source_lane_id") not in lane_ids or relation.get(
            "target_lane_id"
        ) not in lane_ids:
            raise BundleV2FixtureError("topology lane relation does not resolve")
    entry = metadata.get("entry_event", {})
    if metadata.get("severity") == "near_critical":
        if entry.get("source_actor_id") != "V000" or not observed_mask[ANCHOR_STEP, 3]:
            raise BundleV2FixtureError("near-critical entry source is not observable")
    elif entry != _entry_event("control"):
        raise BundleV2FixtureError("control entry event is not canonical")
    return metadata


def _verify_base_episode(
    path: Path,
    *,
    sidecar_metadata: Mapping[str, object],
    sidecar_fingerprint: str,
) -> None:
    expected_files = {"episode.json", *(f"{name}.npy" for name in STORED_FIELD_DTYPES)}
    if {item.name for item in path.iterdir()} != expected_files:
        raise BundleV2FixtureError("base physical file set mismatch")
    metadata = json.loads((path / "episode.json").read_text(encoding="utf-8"))
    if (
        metadata.get("format") != STORAGE_FORMAT
        or metadata.get("schema_version") != STORAGE_SCHEMA_VERSION
        or metadata.get("complete") is not True
        or metadata.get("joint_samples") != 1
    ):
        raise BundleV2FixtureError("base episode metadata mismatch")
    arrays: dict[str, np.memmap] = {}
    for name, dtype in STORED_FIELD_DTYPES.items():
        array = _load_memmap(path / f"{name}.npy", dtype)
        if array.shape != (1, *STORED_FIELD_SHAPES[name]):
            raise BundleV2FixtureError(f"base {name} shape mismatch")
        if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
            raise BundleV2FixtureError(f"base {name} contains a non-finite value")
        arrays[name] = array
    logical_bev = unpack_semantic_bev(arrays[PACKED_BEV_FIELD])
    if logical_bev.shape != (1, *JOINT_SAMPLE_SHAPES["bev"]):
        raise BundleV2FixtureError("base packed BEV round-trip mismatch")
    if not np.array_equal(arrays["agent_role"][0], np.asarray((0, 1, 2))):
        raise BundleV2FixtureError("base role order mismatch")
    gt_mode = np.asarray(arrays["gt_mode"][0])
    mode_mask = np.asarray(arrays["mode_valid_mask"][0])
    if not np.all(mode_mask[np.arange(3), gt_mode]):
        raise BundleV2FixtureError("base GT mode is not hard-valid")
    attributes = metadata.get("attributes")
    if not isinstance(attributes, Mapping):
        raise BundleV2FixtureError("base attributes are missing")
    for name, expected in {
        "scenario_id": sidecar_metadata.get("scenario_id"),
        "local_route": sidecar_metadata.get("local_route"),
        "spawn_seed": sidecar_metadata.get("spawn_seed"),
        "sidecar_dataset_fingerprint": sidecar_fingerprint,
        "selected_sample_steps": [ANCHOR_STEP],
    }.items():
        if attributes.get(name) != expected:
            raise BundleV2FixtureError(f"base/sidecar join mismatch: {name}")


def verify_fixture(output_root: Path | str) -> dict[str, object]:
    output = Path(output_root).expanduser().resolve()
    if _sha256_file(PROTOCOL_PATH) != "21f9a8a193e1c0ca9e85ac648a0a3f9663f979ad0b3ef9518cda6f5750a7b89c":
        raise BundleV2FixtureError("bundle-v2 protocol hash differs from RiskEntry")
    partition_report: dict[str, object] = {}
    parameter_sets: dict[str, set[str]] = {}
    topology_sets: dict[str, set[str]] = {}
    all_seeds: set[int] = set()
    for partition in PARTITIONS:
        root = output / f"riskentry_fixture_{partition}_v2"
        manifest = json.loads((root / "dataset_bundle_manifest.json").read_text())
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        required_root_fields = set(protocol["root_manifest"]["required_fields"])
        if not required_root_fields.issubset(manifest):
            raise BundleV2FixtureError("root manifest is missing required v2 fields")
        if manifest.get("benchmark_partition") != partition:
            raise BundleV2FixtureError("root partition mismatch")
        if manifest.get("bundle_protocol_sha256") != _sha256_file(PROTOCOL_PATH):
            raise BundleV2FixtureError("root protocol binding mismatch")
        if manifest.get("diagnostic_fixture") is not True or manifest.get(
            "eligible_for_formal_training"
        ) is not False:
            raise BundleV2FixtureError("fixture eligibility flags are unsafe")
        sidecar_root = root / "riskentry_actor_sidecar"
        base_root = root / "platoon_joint_bev"
        base_contract = json.loads((base_root / "dataset_contract.json").read_text())
        sidecar_contract = json.loads((sidecar_root / "dataset_contract.json").read_text())
        if base_contract.get("dataset_fingerprint") != manifest.get(
            "base_dataset_fingerprint"
        ):
            raise BundleV2FixtureError("base fingerprint binding mismatch")
        if _sha256_payload(sidecar_contract) != manifest.get("sidecar_dataset_fingerprint"):
            raise BundleV2FixtureError("sidecar fingerprint is not reproducible")
        allowed_splits = ("train", "val", "test") if partition == "id" else ("test",)
        episode_paths = [
            path
            for split in allowed_splits
            for path in sorted((sidecar_root / split / "episodes").glob("episode_*"))
        ]
        if len(episode_paths) != 8:
            raise BundleV2FixtureError("each fixture partition requires eight episodes")
        counts: Counter[tuple[str, str]] = Counter()
        params: set[str] = set()
        topologies: set[str] = set()
        pair_splits: dict[str, str] = {}
        for path in episode_paths:
            metadata = _verify_sidecar_episode(path, partition)
            base_path = (
                base_root
                / str(metadata["split"])
                / "episodes"
                / path.name
            )
            _verify_base_episode(
                base_path,
                sidecar_metadata=metadata,
                sidecar_fingerprint=str(manifest["sidecar_dataset_fingerprint"]),
            )
            if metadata.get("base_dataset_fingerprint") != manifest.get(
                "base_dataset_fingerprint"
            ):
                raise BundleV2FixtureError("sidecar/base fingerprint binding mismatch")
            counts[(str(metadata["scenario_family"]), str(metadata["severity"]))] += 1
            params.add(str(metadata["parameter_combination_id"]))
            topologies.add(str(metadata["topology_id"]))
            pair_id = str(metadata["matched_pair_id"])
            split = str(metadata["split"])
            previous_split = pair_splits.setdefault(pair_id, split)
            if previous_split != split:
                raise BundleV2FixtureError("matched pair crosses a split")
            seed = int(metadata["spawn_seed"])
            if seed in all_seeds:
                raise BundleV2FixtureError("spawn seed leaks across partitions")
            all_seeds.add(seed)
        expected_counts = {
            (family, severity): 1
            for family in SCENARIO_FAMILIES
            for severity in SEVERITIES
        }
        if counts != Counter(expected_counts):
            raise BundleV2FixtureError("scenario/severity fixture balance mismatch")
        if int(manifest.get("eligible_anchor_window_count", -1)) != 8:
            raise BundleV2FixtureError("eligible window aggregate mismatch")
        parameter_sets[partition] = params
        topology_sets[partition] = topologies
        partition_report[partition] = {
            "episodes": 8,
            "eligible_anchor_windows": 8,
            "scenario_severity_cells": len(counts),
            "entry_source_observable_rate": 1.0,
        }
    if parameter_sets["id"] & parameter_sets["compositional_ood"]:
        raise BundleV2FixtureError("ID/compositional-OOD parameter IDs overlap")
    if topology_sets["id"] & topology_sets["topology_ood"]:
        raise BundleV2FixtureError("ID/topology-OOD topology IDs overlap")
    report = {
        "format": f"{FIXTURE_FORMAT}-verification",
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "status": "pass",
        "diagnostic_fixture": True,
        "eligible_for_formal_training": False,
        "bundle_protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "partitions": partition_report,
        "checks": {
            "protocol_hash": "pass",
            "physical_arrays": "pass",
            "ground_truth_observation_separation": "pass",
            "communication": "pass",
            "eligible_windows": "pass",
            "scenario_severity_balance": "pass",
            "entry_events": "pass",
            "topology_graph": "pass",
            "partition_leakage": "pass",
        },
    }
    report["report_sha256"] = _sha256_payload(report)
    _write_json(output / "fixture_verification_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    report = verify_fixture(args.output_root) if args.verify_only else generate_fixture(args.output_root)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BundleV2FixtureError",
    "generate_fixture",
    "verify_fixture",
]
