"""Real-simulator bundle-v2 smoke collector and strict local verifier.

This module deliberately writes new roots and never upgrades bundle-v1 data.
It reuses the production joint expert rollout and raw actor registry adapter,
then adds the v2 online-observation, communication, topology, and entry-event
facts before committing one eligible anchor per episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from expert_dataset.collect_joint_bev import (
    JointBEVSample,
    JointBEVSampleBuilder,
    JointCollectionError,
    JointEpisodeRollout,
    JointEpisodeSidecar,
    JointStepRejected,
    RulePlannerExpert,
    SensorlessJointBEVPlatoonEnv,
    simulator_decision_dt_s,
)
from expert_dataset.riskentry_sidecar_adapter import MetaDriveRiskEntrySidecarAdapter
from expert_dataset.joint_bev_storage import (
    PACKED_BEV_FIELD,
    STORAGE_FORMAT,
    STORAGE_SCHEMA_VERSION,
)
from expert_dataset.joint_risk_bundle_v2_fixture import (
    BUNDLE_FORMAT,
    BUNDLE_SCHEMA_VERSION,
    COMMUNICATION_EDGE_ORDER,
    PARTITIONS,
    PROTOCOL_PATH,
    SCENARIO_FAMILIES,
    SEVERITIES,
    SIDECAR_ARRAY_DTYPES,
    SIDECAR_FORMAT,
    SIDECAR_SCHEMA_VERSION,
    _base_contract,
    _canonical_json,
    _sha256_file,
    _sha256_payload,
    _sidecar_contract,
    _write_component_manifests,
    _write_json,
    _write_npy,
)
from expert_dataset.riskentry_sidecar_storage import (
    SidecarEpisodeStart,
    _build_dense_arrays,
    _derive_outcome,
)
from expert_dataset.semantic_bev_codec import pack_semantic_bev
from models.platoon_planner.collision_geometry import world_trajectory_to_ego_local


REAL_SMOKE_FORMAT = "metadrive-joint-risk-bundle-v2-real-smoke"
REAL_SMOKE_SCHEMA_VERSION = 1
HISTORY_STEPS = 20
FUTURE_STEPS = 50
CONTROL_ANCHOR_STEP = 60
MAX_EPISODE_STEPS = 160
SCENARIO_IDS = {
    ("adjacent_lane_cut_in", "control"): "RV2_adjacent_lane_cut_in_control",
    ("adjacent_lane_cut_in", "near_critical"): "RV2_adjacent_lane_cut_in_near_critical",
    ("on_ramp_external_merge", "control"): "RV2_on_ramp_external_merge_control",
    ("on_ramp_external_merge", "near_critical"): "RV2_on_ramp_external_merge_near_critical",
    ("external_lead_hard_brake", "control"): "RV2_external_lead_hard_brake_control",
    ("external_lead_hard_brake", "near_critical"): "RV2_external_lead_hard_brake_near_critical",
    ("construction_zone_forced_merge", "control"): "RV2_construction_zone_forced_merge_control",
    ("construction_zone_forced_merge", "near_critical"): "RV2_construction_zone_forced_merge_near_critical",
}
ROUTES = {
    "adjacent_lane_cut_in": "R1_entry_straight",
    "on_ramp_external_merge": "R6_mainline_merge_approach",
    "external_lead_hard_brake": "R1_entry_straight",
    "construction_zone_forced_merge": "R1_entry_straight",
}


class RealBundleV2Error(RuntimeError):
    """Raised when real smoke capture or validation violates bundle-v2."""


@dataclass(frozen=True)
class RealV2EpisodeSpec:
    partition: str
    scenario_family: str
    severity: str
    episode_index: int
    split: str
    spawn_seed: int
    traffic_density: float
    road_friction: float
    sensing_noise_std: float
    external_driver_aggressiveness: float
    matched_pair_index: int = 0

    @property
    def scenario_id(self) -> str:
        return SCENARIO_IDS[(self.scenario_family, self.severity)]

    @property
    def local_route(self) -> str:
        return ROUTES[self.scenario_family]

    @property
    def matched_pair_id(self) -> str:
        return (
            f"{self.partition}_{self.scenario_family}_pair_"
            f"{int(self.matched_pair_index):06d}"
        )

    def parameters(self) -> dict[str, float]:
        return {
            "road_friction": float(self.road_friction),
            "sensing_noise_std": float(self.sensing_noise_std),
            "traffic_density": float(self.traffic_density),
            "external_driver_aggressiveness": float(
                self.external_driver_aggressiveness
            ),
        }


def _episode_specs(partition: str) -> tuple[RealV2EpisodeSpec, ...]:
    if partition not in PARTITIONS:
        raise RealBundleV2Error(f"invalid partition: {partition}")
    rows: list[RealV2EpisodeSpec] = []
    for family_index, family in enumerate(SCENARIO_FAMILIES):
        split = "test" if partition != "id" else ("train", "train", "val", "test")[family_index]
        for severity_index, severity in enumerate(SEVERITIES):
            # Half of every partition's cells exercise non-zero traffic.
            # Keep the near-critical realization deterministic and reserve the
            # required 50% non-zero stratum for matched control episodes.
            density = 0.02 if severity == "control" else 0.0
            if partition == "id":
                friction, noise = 0.70, 0.0 if family_index % 2 == 0 else 0.01
                aggression = 0.30 if severity == "control" else 0.70
            elif partition == "compositional_ood":
                # Marginals are from ID support; the tuple is held out.
                friction, noise = 0.70, 0.01 if family_index % 2 == 0 else 0.0
                aggression = 0.70 if severity == "control" else 0.30
            else:
                friction, noise = 0.62, 0.02
                aggression = 0.25 if severity == "control" else 0.80
            episode_index = len(rows)
            seed = 100_000 * PARTITIONS.index(partition) + 1_000 * family_index + 10 * severity_index + 17
            rows.append(
                RealV2EpisodeSpec(
                    partition=partition,
                    scenario_family=family,
                    severity=severity,
                    episode_index=episode_index,
                    split=split,
                    spawn_seed=seed,
                    traffic_density=density,
                    road_friction=friction,
                    sensing_noise_std=noise,
                    external_driver_aggressiveness=aggression,
                )
            )
    return tuple(rows)


def _scenario_contract(partition: str) -> dict[str, object]:
    return {
        "format": "metadrive-riskentry-real-smoke-scenario-contract-v2",
        "schema_version": 1,
        "diagnostic_only": True,
        "eligible_for_formal_training": False,
        "physical_source": "live_metadrive_object_registry",
        "benchmark_partition": partition,
        "scenario_families": list(SCENARIO_FAMILIES),
        "severities": list(SEVERITIES),
        "decision_dt_s": 0.1,
        "history_seconds": 2.0,
        "future_seconds": 5.0,
        "online_observation_policy": {
            "id": "ideal_current_state_range_80m_v1",
            "range_m": 80.0,
            "uses_current_state_only": True,
            "uses_future_or_offline_fields": False,
        },
        "communication_policy": {
            "id": "always_available_50ms_real_smoke_v1",
            "edge_order": list(COMMUNICATION_EDGE_ORDER),
            "delay_s": 0.05,
        },
        "entry_detection": {
            "hard_brake": "measured_longitudinal_acceleration_below_-1mps2",
            "lateral_entry": "measured_lane_change_or_abs_lane_lateral_above_0.15m",
            "onset_range_steps": [60, 120],
        },
        "episode_specs": [
            {
                "scenario_family": spec.scenario_family,
                "severity": spec.severity,
                "scenario_id": spec.scenario_id,
                "local_route": spec.local_route,
                "traffic_density": spec.traffic_density,
                "road_friction": spec.road_friction,
                "sensing_noise_std": spec.sensing_noise_std,
                "external_driver_aggressiveness": spec.external_driver_aggressiveness,
            }
            for spec in _episode_specs(partition)
        ],
    }


def _env_config(spec: RealV2EpisodeSpec) -> dict[str, object]:
    return {
        "num_agents": 3,
        "use_render": False,
        "image_observation": False,
        "image_on_cuda": False,
        "sensors": {},
        "observation_mode": "bev_gt",
        "scenario_id": spec.scenario_id,
        "local_route": spec.local_route,
        "start_seed": spec.spawn_seed,
        "num_scenarios": 1,
        "horizon": MAX_EPISODE_STEPS,
        "traffic_density": spec.traffic_density,
        "ground_truth_traffic_policy": True,
        "platoon_wheel_friction": spec.road_friction,
        "enable_idm_lane_change": bool(
            spec.severity == "near_critical"
            and spec.scenario_family
            in {"adjacent_lane_cut_in", "construction_zone_forced_merge"}
        ),
    }


def _select_source_actor(
    rollout: JointEpisodeRollout, arrays: Mapping[str, np.ndarray]
) -> tuple[str, int]:
    assert rollout.sidecar is not None
    records = {row.actor_id: row for row in rollout.sidecar.actor_records}
    key_ids = rollout.sidecar.key_actor_ids
    for key in ("entry_source", "lead_braker", "intruder"):
        actor_id = key_ids.get(key)
        if actor_id in records:
            return str(actor_id), int(records[str(actor_id)].actor_index)
    external = [row for row in rollout.sidecar.actor_records if row.actor_type == "external"]
    if not external:
        raise RealBundleV2Error("near-critical episode has no external entry source")
    # Deterministic fallback is only identity selection; onset still must be
    # demonstrated by measured motion below.
    external.sort(key=lambda row: (row.first_seen_step, row.actor_index))
    return external[0].actor_id, external[0].actor_index


def _measured_entry_event(
    spec: RealV2EpisodeSpec,
    rollout: JointEpisodeRollout,
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, object], int]:
    if spec.severity == "control":
        return {
            "edge_id": "none",
            "source_actor_id": None,
            "target_actor_id": None,
            "onset_step": None,
            "resolution_step": None,
            "resolution_status": "not_applicable",
        }, CONTROL_ANCHOR_STEP
    source_id, source_index = _select_source_actor(rollout, arrays)
    actor_valid = np.asarray(arrays["actor_valid_mask"])
    state = np.asarray(arrays["actor_state"])
    state_valid = np.asarray(arrays["actor_state_valid_mask"])
    lane_index = np.asarray(arrays["lane_index"])
    lane_state = np.asarray(arrays["lane_state"])
    lane_valid = np.asarray(arrays["lane_valid_mask"])
    onset: int | None = None
    for step in range(60, min(121, len(actor_valid))):
        if not actor_valid[step, source_index]:
            continue
        if spec.scenario_family == "external_lead_hard_brake":
            if not state_valid[step, source_index, 5:7].all():
                continue
            velocity = state[step, source_index, 3:5].astype(np.float64)
            acceleration = state[step, source_index, 5:7].astype(np.float64)
            speed = float(np.linalg.norm(velocity))
            longitudinal_accel = (
                float(np.dot(acceleration, velocity / speed)) if speed > 0.2 else -float(np.linalg.norm(acceleration))
            )
            realized = longitudinal_accel < -1.0
        else:
            previous = max(0, step - 1)
            lane_changed = (
                lane_valid[step, source_index]
                and lane_valid[previous, source_index]
                and lane_index[step, source_index] != lane_index[previous, source_index]
            )
            lateral_motion = lane_valid[step, source_index] and abs(
                float(lane_state[step, source_index, 1])
            ) > 0.15
            realized = bool(lane_changed or lateral_motion)
        if realized:
            onset = step
            break
    if onset is None:
        raise RealBundleV2Error(
            f"{spec.scenario_family} has no measured entry onset in [60,120]"
        )

    platoon_positions = state[onset, :3, :2].astype(np.float64)
    source_position = state[onset, source_index, :2].astype(np.float64)
    target_index = int(np.argmin(np.linalg.norm(platoon_positions - source_position, axis=1)))
    target_id = f"P{target_index}"
    resolution_step: int | None = None
    if spec.scenario_family != "external_lead_hard_brake":
        onset_lane = int(lane_index[onset, source_index])
        for step in range(onset + 1, len(actor_valid) - 5):
            window = lane_index[step : step + 5, source_index]
            if np.all(actor_valid[step : step + 5, source_index]) and np.all(window == window[0]) and int(window[0]) != onset_lane:
                resolution_step = step
                break
    return {
        "edge_id": f"{source_id}->{target_id}",
        "source_actor_id": source_id,
        "target_actor_id": target_id,
        "onset_step": onset,
        "resolution_step": resolution_step,
        "resolution_status": "resolved" if resolution_step is not None else "right_censored",
    }, CONTROL_ANCHOR_STEP


def _online_observation_arrays(
    arrays: Mapping[str, np.ndarray], spec: RealV2EpisodeSpec
) -> dict[str, np.ndarray]:
    state = np.asarray(arrays["actor_state"])
    valid = np.asarray(arrays["actor_valid_mask"])
    state_valid = np.asarray(arrays["actor_state_valid_mask"])
    distances = np.linalg.norm(
        state[:, :, None, :2] - state[:, None, :3, :2], axis=-1
    )
    observed_mask = valid & (np.min(distances, axis=-1) <= 80.0)
    observed_mask[:, :3] = valid[:, :3]
    observed_state = np.zeros_like(state, dtype=np.float32)
    observed_state[observed_mask] = state[observed_mask]
    if spec.sensing_noise_std > 0.0:
        rng = np.random.RandomState(spec.spawn_seed ^ 0x5EED5EED)
        noise = rng.normal(
            0.0, spec.sensing_noise_std, size=observed_state[..., :2].shape
        ).astype(np.float32)
        observed_state[..., :2] += noise * observed_mask[..., None]
    observed_valid = state_valid & observed_mask[..., None]
    observed_state[~observed_valid] = np.float32(0.0)
    timeline = state.shape[0]
    return {
        "actor_observation_mask": observed_mask.astype(np.bool_),
        "observed_actor_state": observed_state.astype(np.float32),
        "observed_actor_state_valid_mask": observed_valid.astype(np.bool_),
        "platoon_comm_available": np.ones((timeline, 2), dtype=np.bool_),
        "platoon_comm_delay_s": np.full((timeline, 2), 0.05, dtype=np.float32),
        "platoon_comm_valid_mask": np.ones((timeline, 2), dtype=np.bool_),
    }


def _lane_tuple(source_lane_id: str) -> tuple[str, str, int] | None:
    try:
        value = json.loads(source_lane_id)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return str(value[0]), str(value[1]), int(value[2])
    except (TypeError, ValueError):
        return None


def _topology(
    spec: RealV2EpisodeSpec,
    lane_rows: Sequence[Mapping[str, object]],
    timeline_length: int,
    onset_step: int,
) -> dict[str, object]:
    parsed = {str(row["lane_id"]): _lane_tuple(str(row["source_lane_id"])) for row in lane_rows}
    relations: set[tuple[str, str, str]] = set()
    for source_id, source in parsed.items():
        if source is None:
            continue
        for target_id, target in parsed.items():
            if target is None or source_id == target_id:
                continue
            if source[:2] == target[:2] and target[2] == source[2] - 1:
                relations.add((source_id, target_id, "left"))
            if source[:2] == target[:2] and target[2] == source[2] + 1:
                relations.add((source_id, target_id, "right"))
            if source[1] == target[0]:
                relation = "merge" if source[0] != target[0] else "successor"
                relations.add((source_id, target_id, relation))
    relation_rows = [
        {"source_lane_id": source, "target_lane_id": target, "relation": relation}
        for source, target, relation in sorted(relations)
    ]
    states: list[dict[str, object]] = []
    for row in lane_rows:
        lane_id = str(row["lane_id"])
        if spec.scenario_family == "construction_zone_forced_merge" and spec.severity == "near_critical" and lane_id == lane_rows[-1]["lane_id"]:
            states.append({"lane_id": lane_id, "start_step": 0, "end_step": onset_step - 1, "state": "open"})
            states.append({"lane_id": lane_id, "start_step": onset_step, "end_step": timeline_length - 1, "state": "work_zone"})
        elif spec.partition == "topology_ood" and lane_id == lane_rows[-1]["lane_id"]:
            states.append({"lane_id": lane_id, "start_step": 0, "end_step": timeline_length - 1, "state": "work_zone"})
        else:
            states.append({"lane_id": lane_id, "start_step": 0, "end_step": timeline_length - 1, "state": "open"})
    physical_signature = _sha256_payload(
        {"lanes": [dict(row) for row in lane_rows], "relations": relation_rows, "states": states}
    )
    topology_id = f"{spec.partition}_{spec.local_route}_{physical_signature[:12]}"
    graph = {
        "topology_id": topology_id,
        "lane_relations": relation_rows,
        "time_varying_lane_states": states,
    }
    return {**graph, "canonical_hash": _sha256_payload(graph)}


def _actor_rows(rollout: JointEpisodeRollout, env_config: Mapping[str, object]) -> list[dict[str, object]]:
    assert rollout.sidecar is not None
    result: list[dict[str, object]] = []
    mass = float(env_config.get("platoon_vehicle_mass", 1600.0))
    max_brake = float(env_config.get("brake_acceleration_scale_mps2", 8.0))
    parameter_id = f"platoon_{_sha256_payload({'mass': mass, 'max_brake': max_brake, 'delay': 0.0})[:12]}"
    for record in rollout.sidecar.actor_records:
        row = asdict(record)
        if record.actor_type == "platoon":
            row.update(
                {
                    "mass_kg": mass,
                    "max_brake_deceleration_mps2": max_brake,
                    "actuation_delay_s": 0.0,
                    "vehicle_parameter_set_id": parameter_id,
                }
            )
        result.append(row)
    return result


def _base_arrays(sample: JointBEVSample) -> dict[str, np.ndarray]:
    arrays = {PACKED_BEV_FIELD: pack_semantic_bev(sample.bev[None])}
    for name, value in sample.as_dict().items():
        if name == "bev":
            continue
        arrays[name] = np.ascontiguousarray(np.asarray(value)[None])
    return arrays


def base_arrays_from_samples(
    samples: Sequence[JointBEVSample],
) -> dict[str, np.ndarray]:
    """Materialize one or more joint samples without changing base schema v2."""

    if not samples:
        raise RealBundleV2Error("at least one joint sample is required")
    bev = np.stack([np.asarray(sample.bev) for sample in samples])
    arrays: dict[str, np.ndarray] = {
        PACKED_BEV_FIELD: pack_semantic_bev(np.ascontiguousarray(bev))
    }
    for name in samples[0].as_dict():
        if name == "bev":
            continue
        arrays[name] = np.ascontiguousarray(
            np.stack([np.asarray(sample.as_dict()[name]) for sample in samples])
        )
    return arrays


def build_real_episode_payload(
    spec: RealV2EpisodeSpec,
    rollout: JointEpisodeRollout,
    *,
    base_fingerprint: str,
    scenario_contract_sha256: str,
    env_config: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, np.ndarray], JointBEVSample, int]:
    if rollout.sidecar is None:
        raise RealBundleV2Error("real rollout did not capture a sidecar timeline")
    start = SidecarEpisodeStart(
        episode_index=spec.episode_index,
        split=spec.split,
        scenario_id=spec.scenario_id,
        local_route=spec.local_route,
        spawn_seed=spec.spawn_seed,
        decision_dt_s=0.1,
        base_dataset_fingerprint=base_fingerprint,
        scenario_parameters={"scenario_contract_sha256": scenario_contract_sha256},
    )
    arrays = _build_dense_arrays(
        metadata=start,
        frames=tuple(capture.frame for capture in rollout.sidecar.captures),
        actor_records=rollout.sidecar.actor_records,
        lane_records=rollout.sidecar.lane_records,
        base_sample_step_indices=(),
    )
    entry, anchor_step = _measured_entry_event(spec, rollout, arrays)
    if anchor_step < HISTORY_STEPS or anchor_step + FUTURE_STEPS >= len(arrays["step_index"]):
        raise RealBundleV2Error("entry anchor lacks continuous 2s history and 5s future")
    sample_by_step = dict(zip(rollout.sample_step_indices, rollout.samples))
    sample = sample_by_step.get(anchor_step)
    if sample is None:
        raise RealBundleV2Error(f"eligible anchor step {anchor_step} has no valid joint BEV sample")
    arrays["base_sample_step_index"] = np.asarray([anchor_step], dtype=np.int64)
    arrays.update(_online_observation_arrays(arrays, spec))
    lane_rows = [asdict(row) for row in rollout.sidecar.lane_records]
    topology_onset = (
        anchor_step
        if entry["onset_step"] is None
        else int(entry["onset_step"])
    )
    topology = _topology(
        spec, lane_rows, len(arrays["step_index"]), topology_onset
    )
    parameters = spec.parameters() | {"scenario_contract_sha256": scenario_contract_sha256}
    parameter_id = f"{spec.partition}_{_sha256_payload(parameters | {'family': spec.scenario_family, 'severity': spec.severity})[:16]}"
    events = [asdict(event) for capture in rollout.sidecar.captures for event in capture.events]
    outcome = _derive_outcome(events)
    realization = "realized" if spec.severity == "control" or entry["onset_step"] is not None else "not_realized"
    metadata = {
        "format": SIDECAR_FORMAT,
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "complete": True,
        "diagnostic_only": True,
        "eligible_for_formal_training": False,
        "physical_source": "live_metadrive_object_registry",
        "episode_index": spec.episode_index,
        "benchmark_partition": spec.partition,
        "split": spec.split,
        "base_dataset_fingerprint": base_fingerprint,
        "decision_dt_s": 0.1,
        "scenario_id": spec.scenario_id,
        "scenario_family": spec.scenario_family,
        "severity": spec.severity,
        "topology_id": topology["topology_id"],
        "parameter_combination_id": parameter_id,
        "matched_pair_id": spec.matched_pair_id,
        "scenario_realization_status": realization,
        "local_route": spec.local_route,
        "spawn_seed": spec.spawn_seed,
        "scenario_parameters": parameters,
        "actors": _actor_rows(rollout, env_config),
        "lanes": lane_rows,
        "topology": topology,
        "entry_event": entry,
        "key_actor_ids": dict(rollout.sidecar.key_actor_ids),
        "events": events,
        "retention": {
            "outcome": outcome,
            "kept_despite_dangerous_outcome": outcome in {"collision", "out_of_road", "out_of_route"},
        },
        "eligible_anchor_steps": [anchor_step],
        "observation_policy_id": "ideal_current_state_range_80m_v1",
        "communication_policy_id": "always_available_50ms_real_smoke_v1",
    }
    return metadata, arrays, sample, anchor_step


def _collect_spec(
    spec: RealV2EpisodeSpec,
    *,
    base_fingerprint: str,
    scenario_contract_sha256: str,
    max_attempts: int,
) -> tuple[dict[str, object], dict[str, np.ndarray], JointBEVSample, int]:
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        attempt_spec = RealV2EpisodeSpec(
            **{**asdict(spec), "spawn_seed": spec.spawn_seed + attempt}
        )
        env_config = _env_config(attempt_spec)
        env = SensorlessJointBEVPlatoonEnv(env_config)
        try:
            rollout = collect_real_smoke_episode(
                env,
                max_steps=MAX_EPISODE_STEPS,
                reset_seed=attempt_spec.spawn_seed,
                sample_step=CONTROL_ANCHOR_STEP,
            )
            payload = build_real_episode_payload(
                attempt_spec,
                rollout,
                base_fingerprint=base_fingerprint,
                scenario_contract_sha256=scenario_contract_sha256,
                env_config=env_config,
            )
            print(
                f"[INFO] v2-real partition={spec.partition} family={spec.scenario_family} "
                f"severity={spec.severity} seed={attempt_spec.spawn_seed} "
                f"raw_steps={len(payload[1]['step_index'])} anchor={payload[3]}",
                flush=True,
            )
            return payload
        except Exception as exc:
            last_error = exc
            print(
                f"[WARNING] v2-real retry partition={spec.partition} family={spec.scenario_family} "
                f"severity={spec.severity} seed={attempt_spec.spawn_seed}: {exc}",
                flush=True,
            )
        finally:
            env.close()
    raise RealBundleV2Error(
        f"real smoke cell failed after {max_attempts} attempts: "
        f"{spec.partition}/{spec.scenario_family}/{spec.severity}: {last_error}"
    )


def collect_real_smoke_episode(
    env: SensorlessJointBEVPlatoonEnv,
    *,
    max_steps: int,
    reset_seed: int,
    sample_step: int,
) -> JointEpisodeRollout:
    """Capture a real timeline with stable lane-follow execution.

    The full production expert is invoked at the requested anchor to create a
    genuine RuleMaker/Normal-planner label, but its maneuver is not executed.
    This isolates the v2 data-interface smoke from long-horizon expert policy
    quality while exercising the same offline-label path as formal v2 runs.
    """

    return collect_real_v2_episode(
        env,
        max_steps=max_steps,
        reset_seed=reset_seed,
        sample_steps=(int(sample_step),),
    )


def collect_real_v2_episode(
    env: SensorlessJointBEVPlatoonEnv,
    *,
    max_steps: int,
    reset_seed: int,
    sample_steps: Sequence[int],
) -> JointEpisodeRollout:
    """Capture a live v2 timeline and label requested states independently.

    The low-level visitation policy is deliberately independent of the label
    planner.  A fresh production RulePlannerExpert is used for every requested
    anchor, preventing unexecuted lane-change commitments from leaking between
    offline labels.
    """

    requested_steps = tuple(sorted({int(step) for step in sample_steps}))
    if not requested_steps or requested_steps[0] < 0:
        raise RealBundleV2Error("sample_steps must contain non-negative steps")
    requested = set(requested_steps)
    env.reset(seed=int(reset_seed))
    agent_ids = ("agent0", "agent1", "agent2")
    builder = JointBEVSampleBuilder(agent_ids)
    builder.reset()
    dt_s = simulator_decision_dt_s(env)
    adapter = MetaDriveRiskEntrySidecarAdapter(decision_dt_s=dt_s)
    captures = [adapter.capture_frame(env, step_index=0, timestamp_s=0.0)]
    samples: list[JointBEVSample] = []
    sample_steps: list[int] = []
    terminated_all = False
    truncated_all = False
    failure_reason: str | None = None
    rejected_joint_steps = 0
    rejection_counts: Counter[str] = Counter()
    simulator_steps = 0
    for step in range(int(max_steps)):
        builder.capture_state(env, timestamp_s=step * dt_s)
        if step in requested:
            if not builder.history_ready():
                continue
            try:
                model_inputs = builder.build_model_inputs(env)
                expert = RulePlannerExpert(env, agent_ids)
                expert_step = expert.plan(env, model_inputs=model_inputs)
                samples.append(
                    builder.build_sample(
                        env, expert_step, model_inputs=model_inputs
                    )
                )
                sample_steps.append(step)
            except (JointCollectionError, JointStepRejected) as exc:
                # Formal quota accounting only includes fully valid anchors.
                # The raw timeline remains usable by RiskEntry even when an
                # individual planner label is rejected.
                rejected_joint_steps += 1
                reason = getattr(exc, "reason_code", type(exc).__name__)
                rejection_counts[str(reason)] += 1

        controls = {
            agent_id: env.trajectory_to_control(
                agent_id,
                _lane_follow_trajectory(env.agents[agent_id], target_speed_mps=6.0),
            )
            for agent_id in agent_ids
        }
        _, _, terminated, truncated, info = env.low_level_step(controls)
        simulator_steps += 1
        captures.append(
            adapter.capture_frame(
                env,
                step_index=simulator_steps,
                timestamp_s=simulator_steps * dt_s,
                transition_info=info if isinstance(info, Mapping) else {},
                terminated=terminated,
                truncated=truncated,
            )
        )
        terminated_all = bool(terminated.get("__all__", False))
        truncated_all = bool(truncated.get("__all__", False))
        if terminated_all or truncated_all:
            failure_reason = "terminated_before_smoke_horizon"
            break
    summary: Mapping[str, object] = {}
    orchestrator = getattr(env, "_scenario_orchestrator", None)
    getter = getattr(orchestrator, "get_episode_summary", None)
    if callable(getter):
        summary = getter()
    return JointEpisodeRollout(
        samples=tuple(samples),
        simulator_steps=simulator_steps,
        rejected_joint_steps=rejected_joint_steps,
        failure_reason=failure_reason,
        terminated=terminated_all,
        truncated=truncated_all,
        joint_step_rejection_counts=dict(rejection_counts),
        sample_step_indices=tuple(sample_steps),
        scenario_summary=dict(summary),
        sidecar=JointEpisodeSidecar(
            captures=tuple(captures),
            actor_records=adapter.actor_records,
            lane_records=adapter.lane_records,
            key_actor_ids=adapter.key_actor_ids,
        ),
    )


def _lane_follow_trajectory(vehicle: Any, *, target_speed_mps: float) -> np.ndarray:
    """Build a current-lane 0.5--4.0 s reference without future-state access."""

    lane = getattr(vehicle, "lane", None)
    if lane is None:
        raise RealBundleV2Error("lane-follow smoke controller has no current lane")
    try:
        start_s, start_d = lane.local_coordinates(vehicle.position)
        lane_length = float(lane.length)
        world = []
        for step in range(1, 9):
            desired_s = float(start_s) + float(target_speed_mps) * 0.5 * step
            sample_s = min(max(desired_s, 0.0), max(lane_length - 0.5, 0.0))
            position = lane.position(sample_s, float(start_d))
            world.append(
                [
                    float(position[0]),
                    float(position[1]),
                    float(lane.heading_theta_at(sample_s)),
                ]
            )
        ego_pose = np.asarray(
            [vehicle.position[0], vehicle.position[1], vehicle.heading_theta],
            dtype=np.float64,
        )
        return world_trajectory_to_ego_local(
            np.asarray(world, dtype=np.float64), ego_pose
        ).astype(np.float32)
    except Exception as exc:
        raise RealBundleV2Error("unable to build lane-follow smoke trajectory") from exc


def collect_real_smoke(output_root: Path | str, *, max_attempts: int = 3) -> dict[str, object]:
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RealBundleV2Error(f"real smoke output root is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    protocol_sha = _sha256_file(PROTOCOL_PATH)
    roots: dict[str, str] = {}
    for partition in PARTITIONS:
        root = output / f"riskentry_real_smoke_{partition}_v2"
        root.mkdir()
        roots[partition] = str(root)
        scenario_contract = _scenario_contract(partition)
        scenario_sha = _sha256_payload(scenario_contract)
        _write_json(root / "scenario_contract.json", scenario_contract)
        base_fingerprint = _sha256_payload(
            {"collector": REAL_SMOKE_FORMAT, "partition": partition, "scenario_contract_sha256": scenario_sha}
        )
        sidecar_contract = _sidecar_contract(base_fingerprint)
        sidecar_fingerprint = _sha256_payload(sidecar_contract)
        base_root = root / "platoon_joint_bev"
        sidecar_root = root / "riskentry_actor_sidecar"
        _write_json(base_root / "dataset_contract.json", _base_contract(partition, base_fingerprint))
        _write_json(sidecar_root / "dataset_contract.json", sidecar_contract)
        allowed_splits = ("train", "val", "test") if partition == "id" else ("test",)
        base_rows: dict[str, list[dict[str, object]]] = {split: [] for split in allowed_splits}
        sidecar_rows: dict[str, list[dict[str, object]]] = {split: [] for split in allowed_splits}
        index_rows: list[dict[str, object]] = []
        parameter_ids: list[str] = []
        topology_ids: list[str] = []
        for spec in _episode_specs(partition):
            metadata, side_arrays, sample, anchor = _collect_spec(
                spec,
                base_fingerprint=base_fingerprint,
                scenario_contract_sha256=scenario_sha,
                max_attempts=max_attempts,
            )
            episode_name = f"episode_{spec.episode_index:08d}"
            side_path = sidecar_root / spec.split / "episodes" / episode_name
            _write_json(side_path / "episode.json", metadata)
            for name, array in side_arrays.items():
                _write_npy(side_path / f"{name}.npy", array)
            attributes = {
                "scenario_id": metadata["scenario_id"],
                "local_route": metadata["local_route"],
                "spawn_seed": metadata["spawn_seed"],
                "sidecar_dataset_fingerprint": sidecar_fingerprint,
                "selected_sample_steps": [anchor],
                "benchmark_partition": partition,
                "scenario_family": spec.scenario_family,
                "severity": spec.severity,
                "diagnostic_only": True,
                "eligible_for_formal_training": False,
            }
            base_path = base_root / spec.split / "episodes" / episode_name
            _write_json(
                base_path / "episode.json",
                {
                    "format": STORAGE_FORMAT,
                    "schema_version": STORAGE_SCHEMA_VERSION,
                    "complete": True,
                    "episode_index": spec.episode_index,
                    "split": spec.split,
                    "joint_samples": 1,
                    "attributes": attributes,
                },
            )
            for name, array in _base_arrays(sample).items():
                _write_npy(base_path / f"{name}.npy", array)
            base_rows[spec.split].append(
                {"episode_index": spec.episode_index, "directory": episode_name, "joint_samples": 1, "attributes": attributes}
            )
            outcome = str(metadata["retention"]["outcome"])
            sidecar_rows[spec.split].append(
                {
                    "episode_index": spec.episode_index,
                    "directory": episode_name,
                    "raw_steps": len(side_arrays["step_index"]),
                    "actor_count": side_arrays["actor_state"].shape[1],
                    "base_samples": 1,
                    "outcome": outcome,
                }
            )
            index_rows.append(
                {
                    "episode_index": spec.episode_index,
                    "split": spec.split,
                    "scenario_id": str(metadata["scenario_id"]),
                    "local_route": spec.local_route,
                    "spawn_seed": int(metadata["spawn_seed"]),
                    "base_status": "committed",
                    "base_rejection_reason": None,
                    "sidecar_status": "committed",
                    "sidecar_rejection_reason": None,
                    "raw_steps": len(side_arrays["step_index"]),
                    "base_samples": 1,
                    "outcome": outcome,
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
            "".join(_canonical_json(row) + "\n" for row in index_rows), encoding="utf-8"
        )
        manifest = {
            "format": BUNDLE_FORMAT,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "diagnostic_only": True,
            "eligible_for_formal_training": False,
            "physical_source": "live_metadrive_object_registry",
            "dataset_instance_id": f"diagnostic_real_bundle_v2_{partition}_smoke_v1",
            "benchmark_partition": partition,
            "bundle_protocol_sha256": protocol_sha,
            "scenario_contract_sha256": scenario_sha,
            "base_dataset_fingerprint": base_fingerprint,
            "sidecar_dataset_fingerprint": sidecar_fingerprint,
            "parameter_space_id": f"diagnostic_real_smoke_{partition}_parameters_v1",
            "parameter_tuple_set_sha256": _sha256_payload(sorted(parameter_ids)),
            "topology_set_id": f"diagnostic_real_smoke_{partition}_topologies_v1",
            "topology_id_set_sha256": _sha256_payload(sorted(set(topology_ids))),
            "target_eligible_anchor_windows": 8,
            "eligible_anchor_window_count": 8,
            "counting_unit": "eligible_anchor_windows_with_2s_history_and_valid_5s_future",
            "scenario_catalog": list(SCENARIO_FAMILIES),
            "episode_split_manifest": {
                split: [int(row["episode_index"]) for row in rows]
                for split, rows in base_rows.items()
            },
            "base_directory": "platoon_joint_bev",
            "sidecar_directory": "riskentry_actor_sidecar",
        }
        _write_json(root / "dataset_bundle_manifest.json", manifest)
    _write_json(
        output / "real_smoke_summary.json",
        {
            "format": REAL_SMOKE_FORMAT,
            "schema_version": REAL_SMOKE_SCHEMA_VERSION,
            "diagnostic_only": True,
            "eligible_for_formal_training": False,
            "roots": roots,
            "episodes": 24,
            "eligible_anchor_windows": 24,
        },
    )
    return verify_real_smoke(output)


def verify_real_smoke(
    output_root: Path | str, *, write_report: bool = True
) -> dict[str, object]:
    output = Path(output_root).expanduser().resolve()
    reports: dict[str, object] = {}
    parameter_sets: dict[str, set[str]] = {}
    topology_sets: dict[str, set[str]] = {}
    all_seeds: set[int] = set()
    for partition in PARTITIONS:
        root = output / f"riskentry_real_smoke_{partition}_v2"
        manifest = json.loads((root / "dataset_bundle_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("eligible_anchor_window_count") != 8 or manifest.get("diagnostic_only") is not True:
            raise RealBundleV2Error("real smoke root manifest count/eligibility mismatch")
        side_root = root / "riskentry_actor_sidecar"
        allowed = ("train", "val", "test") if partition == "id" else ("test",)
        paths = [
            path
            for split in allowed
            for path in sorted((side_root / split / "episodes").glob("episode_*"))
        ]
        if len(paths) != 8:
            raise RealBundleV2Error("real smoke requires eight episodes per partition")
        cells: Counter[tuple[str, str]] = Counter()
        source_observed = 0
        near_count = 0
        params: set[str] = set()
        topologies: set[str] = set()
        for path in paths:
            metadata = json.loads((path / "episode.json").read_text(encoding="utf-8"))
            if metadata.get("physical_source") != "live_metadrive_object_registry":
                raise RealBundleV2Error("episode is not backed by a live simulator registry")
            arrays = {
                name: np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                for name in SIDECAR_ARRAY_DTYPES
            }
            if any(not isinstance(array, np.memmap) for array in arrays.values()):
                raise RealBundleV2Error("v2 sidecar arrays must be mmap-readable")
            if any(arrays[name].dtype != dtype for name, dtype in SIDECAR_ARRAY_DTYPES.items()):
                raise RealBundleV2Error("v2 sidecar dtype mismatch")
            timeline, actors = arrays["actor_valid_mask"].shape
            expected_shapes = {
                "actor_state": (timeline, actors, 8),
                "actor_state_valid_mask": (timeline, actors, 8),
                "lane_index": (timeline, actors),
                "lane_state": (timeline, actors, 4),
                "lane_valid_mask": (timeline, actors),
                "actor_observation_mask": (timeline, actors),
                "observed_actor_state": (timeline, actors, 8),
                "observed_actor_state_valid_mask": (timeline, actors, 8),
                "platoon_comm_available": (timeline, 2),
                "platoon_comm_delay_s": (timeline, 2),
                "platoon_comm_valid_mask": (timeline, 2),
            }
            if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
                raise RealBundleV2Error("v2 sidecar shape mismatch")
            if any(not np.isfinite(array).all() for array in (arrays["actor_state"], arrays["lane_state"], arrays["observed_actor_state"], arrays["platoon_comm_delay_s"])):
                raise RealBundleV2Error("real smoke contains non-finite values")
            observed_valid = np.asarray(arrays["observed_actor_state_valid_mask"])
            if np.any(np.asarray(arrays["observed_actor_state"])[~observed_valid] != 0.0):
                raise RealBundleV2Error("unobserved online state is not zero-masked")
            anchors = np.asarray(arrays["base_sample_step_index"])
            if anchors.shape != (1,) or int(anchors[0]) < HISTORY_STEPS or int(anchors[0]) + FUTURE_STEPS >= timeline:
                raise RealBundleV2Error("real smoke anchor lacks history/future")
            graph = {
                "topology_id": metadata["topology"]["topology_id"],
                "lane_relations": metadata["topology"]["lane_relations"],
                "time_varying_lane_states": metadata["topology"]["time_varying_lane_states"],
            }
            if metadata["topology"]["canonical_hash"] != _sha256_payload(graph):
                raise RealBundleV2Error("real smoke topology hash mismatch")
            lane_ids = {row["lane_id"] for row in metadata["lanes"]}
            if any(row["source_lane_id"] not in lane_ids or row["target_lane_id"] not in lane_ids for row in graph["lane_relations"]):
                raise RealBundleV2Error("real smoke topology relation does not resolve")
            cells[(metadata["scenario_family"], metadata["severity"])] += 1
            params.add(str(metadata["parameter_combination_id"]))
            topologies.add(str(metadata["topology_id"]))
            seed = int(metadata["spawn_seed"])
            if seed in all_seeds:
                raise RealBundleV2Error("spawn seed leaks across roots")
            all_seeds.add(seed)
            if metadata["severity"] == "near_critical":
                near_count += 1
                source_id = metadata["entry_event"]["source_actor_id"]
                actor_index = next(row["actor_index"] for row in metadata["actors"] if row["actor_id"] == source_id)
                onset = int(metadata["entry_event"]["onset_step"])
                source_observed += int(arrays["actor_observation_mask"][onset, actor_index])
        expected = Counter((family, severity) for family in SCENARIO_FAMILIES for severity in SEVERITIES)
        if cells != expected:
            raise RealBundleV2Error("real smoke scenario/severity cells are incomplete")
        if source_observed != near_count:
            raise RealBundleV2Error("entry source observability is below 100% in smoke")
        parameter_sets[partition] = params
        topology_sets[partition] = topologies
        reports[partition] = {
            "episodes": 8,
            "eligible_anchor_windows": 8,
            "entry_source_observable_rate": 1.0,
        }
    if parameter_sets["id"] & parameter_sets["compositional_ood"]:
        raise RealBundleV2Error("ID and compositional-OOD parameter IDs overlap")
    if topology_sets["id"] & topology_sets["topology_ood"]:
        raise RealBundleV2Error("ID and topology-OOD topology IDs overlap")
    report = {
        "format": f"{REAL_SMOKE_FORMAT}-verification",
        "schema_version": REAL_SMOKE_SCHEMA_VERSION,
        "status": "pass",
        "diagnostic_only": True,
        "eligible_for_formal_training": False,
        "bundle_protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "partitions": reports,
        "checks": {
            "live_simulator_provenance": "pass",
            "protocol_and_physical_arrays": "pass",
            "eligible_windows": "pass",
            "entry_events_and_observation": "pass",
            "communication": "pass",
            "topology": "pass",
            "partition_leakage": "pass",
        },
    }
    report["report_sha256"] = _sha256_payload(report)
    if write_report:
        _write_json(output / "real_smoke_verification_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    report = (
        verify_real_smoke(args.output_root, write_report=False)
        if args.verify_only
        else collect_real_smoke(args.output_root, max_attempts=args.max_attempts)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RealBundleV2Error",
    "RealV2EpisodeSpec",
    "build_real_episode_payload",
    "collect_real_smoke",
    "verify_real_smoke",
]
