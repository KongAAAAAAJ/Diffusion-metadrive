"""Atomic storage for the RiskEntry MetaDrive actor sidecar.

The sidecar is deliberately separate from the joint-BEV episode directory.
This module implements the producer sink, lossless dense array conversion,
atomic episode commits, and resumable split manifests.  It does not drive the
simulator; one-pass collector integration belongs to Round 13.97d.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from expert_dataset.joint_bev_storage import (
    SPLIT_NAMES,
    STORAGE_FORMAT as BASE_STORAGE_FORMAT,
    STORAGE_SCHEMA_VERSION as BASE_STORAGE_SCHEMA_VERSION,
)
from expert_dataset.riskentry_sidecar_adapter import (
    PLATOON_ROLES,
    RAW_EVENT_TYPES,
    SidecarActorRecord,
    SidecarActorSnapshot,
    SidecarLaneRecord,
    SidecarRawEvent,
    SidecarRawFrame,
)
from expert_dataset.joint_risk_bundle_contract import (
    EXTERNAL_ACTOR_ID_PATTERN,
    PLATOON_AGENT_TO_ACTOR_ID,
)


SIDECAR_FORMAT = "riskentry-metadrive-actor-sidecar"
SIDECAR_SCHEMA_VERSION = "1.0.0"
EPISODE_PATTERN = re.compile(r"^episode_(\d{8})$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ACTOR_STATE_CHANNELS = (
    "world_x_m",
    "world_y_m",
    "heading_rad",
    "velocity_x_mps",
    "velocity_y_mps",
    "acceleration_x_mps2",
    "acceleration_y_mps2",
    "yaw_rate_radps",
)
LANE_STATE_CHANNELS = (
    "lane_s_m",
    "lane_lateral_m",
    "lane_heading_error_rad",
    "lane_width_m",
)
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
}
EPISODE_FILE_NAMES = frozenset(
    {"episode.json", *(f"{name}.npy" for name in SIDECAR_ARRAY_DTYPES)}
)
OUTCOMES = frozenset(
    {"success", "collision", "out_of_road", "out_of_route", "terminated", "truncated"}
)
LANE_TYPES = frozenset(
    {"mainline", "merge", "exit", "shoulder", "intersection", "unknown"}
)
DANGEROUS_OUTCOMES = frozenset(OUTCOMES - {"success"})


class RiskEntrySidecarStorageError(RuntimeError):
    """Raised when the sidecar is corrupt or violates the frozen protocol."""


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RiskEntrySidecarStorageError("metadata float values must be finite")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item())
    raise RiskEntrySidecarStorageError(
        f"metadata value is not JSON serializable: {type(value).__name__}"
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RiskEntrySidecarStorageError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise RiskEntrySidecarStorageError(f"JSON root must be an object: {path}")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def sidecar_dataset_contract(base_dataset_fingerprint: str) -> dict[str, object]:
    fingerprint = str(base_dataset_fingerprint)
    if SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise RiskEntrySidecarStorageError(
            "base_dataset_fingerprint must be a lowercase SHA256 digest"
        )
    return {
        "format": SIDECAR_FORMAT,
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "base_format": BASE_STORAGE_FORMAT,
        "base_schema_version": BASE_STORAGE_SCHEMA_VERSION,
        "base_dataset_fingerprint": fingerprint,
        "split_policy": "mirror_base_episode_split",
        "timeline_policy": "all_decision_boundaries_including_terminal_post_step",
        "missing_value_policy": "zero_plus_explicit_mask",
        "actor_state_channels": list(ACTOR_STATE_CHANNELS),
        "lane_state_channels": list(LANE_STATE_CHANNELS),
    }


def sidecar_dataset_fingerprint(base_dataset_fingerprint: str) -> str:
    contract = sidecar_dataset_contract(base_dataset_fingerprint)
    return hashlib.sha256(_canonical_json(contract).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SidecarEpisodeStart:
    """Metadata frozen before raw state zero is appended."""

    episode_index: int
    split: str
    scenario_id: str
    local_route: str
    spawn_seed: int
    decision_dt_s: float
    base_dataset_fingerprint: str
    scenario_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.episode_index, bool) or int(self.episode_index) < 0:
            raise RiskEntrySidecarStorageError("episode_index must be non-negative")
        if self.split not in SPLIT_NAMES:
            raise RiskEntrySidecarStorageError(f"split must be one of {SPLIT_NAMES}")
        if not self.scenario_id or not self.local_route:
            raise RiskEntrySidecarStorageError("scenario_id and local_route must be non-empty")
        if isinstance(self.spawn_seed, bool) or int(self.spawn_seed) < 0:
            raise RiskEntrySidecarStorageError("spawn_seed must be non-negative")
        if not math.isfinite(self.decision_dt_s) or not math.isclose(
            float(self.decision_dt_s), 0.1, rel_tol=0.0, abs_tol=1e-9
        ):
            raise RiskEntrySidecarStorageError(
                "the shared bundle decision_dt_s is frozen to 0.1"
            )
        if SHA256_PATTERN.fullmatch(str(self.base_dataset_fingerprint)) is None:
            raise RiskEntrySidecarStorageError(
                "base_dataset_fingerprint must be a lowercase SHA256 digest"
            )
        if not isinstance(self.scenario_parameters, Mapping):
            raise RiskEntrySidecarStorageError("scenario_parameters must be a mapping")
        scenario_hash = self.scenario_parameters.get("scenario_contract_sha256")
        if not isinstance(scenario_hash, str) or SHA256_PATTERN.fullmatch(scenario_hash) is None:
            raise RiskEntrySidecarStorageError(
                "scenario_parameters must contain scenario_contract_sha256"
            )


@dataclass(frozen=True)
class StoredSidecarEpisode:
    episode_index: int
    split: str
    directory: str
    raw_steps: int
    actor_count: int
    base_samples: int
    outcome: str

    def manifest_entry(self) -> dict[str, object]:
        return asdict(self) | {"episode_index": self.episode_index}


def _derive_outcome(events: Sequence[SidecarRawEvent] | Sequence[Mapping[str, Any]]) -> str:
    event_types = {
        event.event_type if isinstance(event, SidecarRawEvent) else str(event.get("event_type", ""))
        for event in events
    }
    if event_types & {"collision_vehicle", "collision_object", "collision_sidewalk"}:
        return "collision"
    if "out_of_road" in event_types:
        return "out_of_road"
    if "out_of_route" in event_types:
        return "out_of_route"
    if "truncated" in event_types:
        return "truncated"
    if "terminated" in event_types:
        return "terminated"
    return "success"


def _validate_actor_records(
    records: Sequence[SidecarActorRecord], timeline_length: int
) -> tuple[SidecarActorRecord, ...]:
    result = tuple(sorted(records, key=lambda item: item.actor_index))
    if len(result) < 3:
        raise RiskEntrySidecarStorageError("episode must register at least three actors")
    if [item.actor_index for item in result] != list(range(len(result))):
        raise RiskEntrySidecarStorageError("actor_index must be contiguous from zero")
    actor_ids = [item.actor_id for item in result]
    source_ids = [item.source_object_id for item in result]
    if len(set(actor_ids)) != len(actor_ids) or len(set(source_ids)) != len(source_ids):
        raise RiskEntrySidecarStorageError("actor and source object IDs must be unique")
    expected_platoon = [
        (index, actor_id, agent_id, PLATOON_ROLES[agent_id])
        for index, (agent_id, actor_id) in enumerate(PLATOON_AGENT_TO_ACTOR_ID.items())
    ]
    observed_platoon = [
        (item.actor_index, item.actor_id, item.source_object_id, item.platoon_role)
        for item in result[:3]
    ]
    if observed_platoon != expected_platoon or any(
        item.actor_type != "platoon" for item in result[:3]
    ):
        raise RiskEntrySidecarStorageError(
            "actor rows 0..2 must be agent0/1/2 -> P0/P1/P2 with fixed roles"
        )
    external = result[3:]
    expected_external_order = sorted(
        external, key=lambda item: (item.first_seen_step, item.source_object_id)
    )
    if list(external) != expected_external_order:
        raise RiskEntrySidecarStorageError(
            "external actors must follow (first_seen_step, source_object_id) order"
        )
    for external_index, item in enumerate(external):
        if (
            item.actor_type != "external"
            or item.platoon_role is not None
            or EXTERNAL_ACTOR_ID_PATTERN.fullmatch(item.actor_id) is None
            or item.actor_id != f"V{external_index:03d}"
        ):
            raise RiskEntrySidecarStorageError("external actor identity contract mismatch")
    for item in result:
        if (
            item.first_seen_step < 0
            or item.first_seen_step >= timeline_length
            or not math.isfinite(item.length_m)
            or not math.isfinite(item.width_m)
            or item.length_m <= 0.0
            or item.width_m <= 0.0
        ):
            raise RiskEntrySidecarStorageError("invalid actor table row")
    return result


def _validate_lane_records(records: Sequence[SidecarLaneRecord]) -> tuple[SidecarLaneRecord, ...]:
    result = tuple(sorted(records, key=lambda item: item.lane_index))
    if [item.lane_index for item in result] != list(range(len(result))):
        raise RiskEntrySidecarStorageError("lane_index must be contiguous from zero")
    if len({item.lane_id for item in result}) != len(result) or len(
        {item.source_lane_id for item in result}
    ) != len(result):
        raise RiskEntrySidecarStorageError("lane IDs must be unique")
    for index, item in enumerate(result):
        if (
            item.lane_id != f"L{index:03d}"
            or not item.source_lane_id
            or item.lane_type not in LANE_TYPES
        ):
            raise RiskEntrySidecarStorageError("invalid lane table row")
    return result


def _build_dense_arrays(
    *,
    metadata: SidecarEpisodeStart,
    frames: Sequence[SidecarRawFrame],
    actor_records: Sequence[SidecarActorRecord],
    lane_records: Sequence[SidecarLaneRecord],
    base_sample_step_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    timeline_length = len(frames)
    if timeline_length == 0:
        raise RiskEntrySidecarStorageError("cannot commit an empty raw timeline")
    actors = _validate_actor_records(actor_records, timeline_length)
    lanes = _validate_lane_records(lane_records)
    actor_by_id = {item.actor_id: item for item in actors}
    lane_by_id = {item.lane_id: item for item in lanes}
    actor_count = len(actors)

    step_index = np.arange(timeline_length, dtype=np.int64)
    timestamp_s = step_index.astype(np.float64) * float(metadata.decision_dt_s)
    actor_state = np.zeros((timeline_length, actor_count, 8), dtype=np.float32)
    actor_state_valid = np.zeros((timeline_length, actor_count, 8), dtype=np.bool_)
    actor_valid = np.zeros((timeline_length, actor_count), dtype=np.bool_)
    lane_index = np.full((timeline_length, actor_count), -1, dtype=np.int32)
    lane_state = np.zeros((timeline_length, actor_count, 4), dtype=np.float32)
    lane_valid = np.zeros((timeline_length, actor_count), dtype=np.bool_)

    for expected_step, frame in enumerate(frames):
        expected_time = expected_step * float(metadata.decision_dt_s)
        if frame.step_index != expected_step or not math.isclose(
            frame.timestamp_s, expected_time, rel_tol=0.0, abs_tol=1e-6
        ):
            raise RiskEntrySidecarStorageError("raw frame timeline is not contiguous")
        seen: set[str] = set()
        for snapshot in frame.actors:
            if snapshot.actor_id in seen:
                raise RiskEntrySidecarStorageError("actor appears twice in one raw frame")
            seen.add(snapshot.actor_id)
            record = actor_by_id.get(snapshot.actor_id)
            if record is None:
                raise RiskEntrySidecarStorageError("raw frame references an unknown actor")
            if (
                snapshot.source_object_id != record.source_object_id
                or snapshot.actor_type != record.actor_type
                or not math.isclose(snapshot.length_m, record.length_m, abs_tol=1e-6)
                or not math.isclose(snapshot.width_m, record.width_m, abs_tol=1e-6)
            ):
                raise RiskEntrySidecarStorageError("raw actor identity/static attributes changed")
            index = record.actor_index
            state = np.asarray(
                [
                    snapshot.world_x_m,
                    snapshot.world_y_m,
                    snapshot.heading_rad,
                    snapshot.velocity_x_mps,
                    snapshot.velocity_y_mps,
                    snapshot.acceleration_x_mps2,
                    snapshot.acceleration_y_mps2,
                    snapshot.yaw_rate_radps,
                ],
                dtype=np.float32,
            )
            if not np.isfinite(state).all():
                raise RiskEntrySidecarStorageError("raw actor state contains NaN or infinity")
            actor_state[expected_step, index] = state
            actor_state_valid[expected_step, index, :5] = True
            actor_state_valid[expected_step, index, 5:7] = bool(
                snapshot.acceleration_valid
            )
            actor_state_valid[expected_step, index, 7] = bool(snapshot.yaw_rate_valid)
            actor_valid[expected_step, index] = True
            if snapshot.lane_valid:
                lane = lane_by_id.get(snapshot.lane_id)
                if lane is None:
                    raise RiskEntrySidecarStorageError("raw frame references an unknown lane")
                lane_values = np.asarray(
                    [
                        snapshot.lane_s_m,
                        snapshot.lane_lateral_m,
                        snapshot.lane_heading_error_rad,
                        snapshot.lane_width_m,
                    ],
                    dtype=np.float32,
                )
                if not np.isfinite(lane_values).all() or lane_values[3] <= 0.0:
                    raise RiskEntrySidecarStorageError("invalid active lane state")
                lane_index[expected_step, index] = np.int32(lane.lane_index)
                lane_state[expected_step, index] = lane_values
                lane_valid[expected_step, index] = True

    for record in actors:
        active_steps = np.flatnonzero(actor_valid[:, record.actor_index])
        if active_steps.size == 0 or int(active_steps[0]) != record.first_seen_step:
            raise RiskEntrySidecarStorageError("actor first_seen_step does not match timeline")
        appearance = actor_valid[:, record.actor_index] & np.concatenate(
            (np.asarray([True]), ~actor_valid[:-1, record.actor_index])
        )
        if np.any(actor_state_valid[appearance, record.actor_index, 5:]):
            raise RiskEntrySidecarStorageError(
                "acceleration/yaw rate must be invalid on first sight and reappearance"
            )
    if not np.all(actor_valid[0, :3]):
        raise RiskEntrySidecarStorageError("initial state must contain P0/P1/P2")

    base_steps = np.asarray(base_sample_step_indices, dtype=np.int64)
    if base_steps.ndim != 1:
        raise RiskEntrySidecarStorageError("base_sample_step_index must be one-dimensional")
    if len(base_steps) and (
        np.any(np.diff(base_steps) <= 0)
        or int(base_steps[0]) < 0
        or int(base_steps[-1]) >= timeline_length
    ):
        raise RiskEntrySidecarStorageError(
            "base sample steps must be strictly ordered and on the raw timeline"
        )
    return {
        "step_index": step_index,
        "timestamp_s": timestamp_s,
        "actor_state": actor_state,
        "actor_state_valid_mask": actor_state_valid,
        "actor_valid_mask": actor_valid,
        "lane_index": lane_index,
        "lane_state": lane_state,
        "lane_valid_mask": lane_valid,
        "base_sample_step_index": np.ascontiguousarray(base_steps),
    }


def validate_sidecar_episode_payload(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> None:
    """Validate a committed episode against the frozen RiskEntry schema."""

    expected_metadata_keys = {
        "format",
        "schema_version",
        "complete",
        "episode_index",
        "split",
        "base_dataset_fingerprint",
        "decision_dt_s",
        "scenario_id",
        "local_route",
        "spawn_seed",
        "scenario_parameters",
        "actors",
        "lanes",
        "key_actor_ids",
        "events",
        "retention",
    }
    if set(metadata) != expected_metadata_keys:
        raise RiskEntrySidecarStorageError("episode metadata field set mismatch")
    if metadata.get("format") != SIDECAR_FORMAT:
        raise RiskEntrySidecarStorageError("sidecar format mismatch")
    if metadata.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise RiskEntrySidecarStorageError("sidecar schema version mismatch")
    if metadata.get("complete") is not True:
        raise RiskEntrySidecarStorageError("episode is not marked complete")
    episode_index_value = metadata.get("episode_index")
    spawn_seed_value = metadata.get("spawn_seed")
    if (
        isinstance(episode_index_value, bool)
        or not isinstance(episode_index_value, int)
        or episode_index_value < 0
        or isinstance(spawn_seed_value, bool)
        or not isinstance(spawn_seed_value, int)
        or spawn_seed_value < 0
    ):
        raise RiskEntrySidecarStorageError("episode_index/spawn_seed must be non-negative integers")
    if metadata.get("split") not in SPLIT_NAMES:
        raise RiskEntrySidecarStorageError("invalid episode split")
    if SHA256_PATTERN.fullmatch(str(metadata.get("base_dataset_fingerprint", ""))) is None:
        raise RiskEntrySidecarStorageError("invalid base dataset fingerprint")
    decision_dt_s = float(metadata.get("decision_dt_s", 0.0))
    if not math.isclose(decision_dt_s, 0.1, rel_tol=0.0, abs_tol=1e-9):
        raise RiskEntrySidecarStorageError("sidecar decision_dt_s must be 0.1")
    if not str(metadata.get("scenario_id", "")) or not str(metadata.get("local_route", "")):
        raise RiskEntrySidecarStorageError("scenario and route must be non-empty")
    scenario_parameters = metadata.get("scenario_parameters")
    if not isinstance(scenario_parameters, Mapping) or SHA256_PATTERN.fullmatch(
        str(scenario_parameters.get("scenario_contract_sha256", ""))
    ) is None:
        raise RiskEntrySidecarStorageError("scenario contract SHA256 is missing")
    _jsonable(scenario_parameters)

    actor_rows = metadata.get("actors")
    lane_rows = metadata.get("lanes")
    if not isinstance(actor_rows, list) or not isinstance(lane_rows, list):
        raise RiskEntrySidecarStorageError("actors and lanes must be arrays")
    try:
        actor_records = tuple(SidecarActorRecord(**row) for row in actor_rows)
        lane_records = tuple(SidecarLaneRecord(**row) for row in lane_rows)
    except (TypeError, ValueError) as exc:
        raise RiskEntrySidecarStorageError("invalid actor/lane table fields") from exc

    if set(arrays) != set(SIDECAR_ARRAY_DTYPES):
        raise RiskEntrySidecarStorageError("sidecar physical array set mismatch")
    for name, dtype in SIDECAR_ARRAY_DTYPES.items():
        if np.asarray(arrays[name]).dtype != dtype:
            raise RiskEntrySidecarStorageError(f"{name} dtype mismatch")
    steps = np.asarray(arrays["step_index"])
    timestamps = np.asarray(arrays["timestamp_s"])
    if steps.ndim != 1 or len(steps) == 0 or not np.array_equal(
        steps, np.arange(len(steps), dtype=np.int64)
    ):
        raise RiskEntrySidecarStorageError("step_index must be contiguous from zero")
    timeline_length = len(steps)
    if timestamps.shape != (timeline_length,) or not np.allclose(
        timestamps,
        np.arange(timeline_length, dtype=np.float64) * decision_dt_s,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RiskEntrySidecarStorageError("timestamp timeline mismatch")
    actor_records = _validate_actor_records(actor_records, timeline_length)
    lane_records = _validate_lane_records(lane_records)
    actor_count = len(actor_records)
    expected_shapes = {
        "actor_state": (timeline_length, actor_count, 8),
        "actor_state_valid_mask": (timeline_length, actor_count, 8),
        "actor_valid_mask": (timeline_length, actor_count),
        "lane_index": (timeline_length, actor_count),
        "lane_state": (timeline_length, actor_count, 4),
        "lane_valid_mask": (timeline_length, actor_count),
    }
    for name, shape in expected_shapes.items():
        if np.asarray(arrays[name]).shape != shape:
            raise RiskEntrySidecarStorageError(f"{name} shape mismatch")
    actor_state = np.asarray(arrays["actor_state"])
    state_valid = np.asarray(arrays["actor_state_valid_mask"])
    actor_valid = np.asarray(arrays["actor_valid_mask"])
    lane_index = np.asarray(arrays["lane_index"])
    lane_state = np.asarray(arrays["lane_state"])
    lane_valid = np.asarray(arrays["lane_valid_mask"])
    if not np.isfinite(actor_state).all() or not np.isfinite(lane_state).all():
        raise RiskEntrySidecarStorageError("sidecar arrays contain NaN or infinity")
    headings = actor_state[..., 2][actor_valid]
    if np.any(headings < -np.pi) or np.any(headings >= np.pi):
        raise RiskEntrySidecarStorageError("active actor heading is outside [-pi,pi)")
    if np.any(state_valid & ~actor_valid[..., None]):
        raise RiskEntrySidecarStorageError("inactive actor has valid state channels")
    if not all(np.all(state_valid[..., channel][actor_valid]) for channel in range(5)):
        raise RiskEntrySidecarStorageError("active actors require pose, heading, and velocity")
    if not np.array_equal(state_valid[..., 5], state_valid[..., 6]):
        raise RiskEntrySidecarStorageError("acceleration x/y validity must match")
    if np.any(actor_state[~state_valid] != 0.0):
        raise RiskEntrySidecarStorageError("invalid actor state channels must contain zero")
    if np.any(lane_valid & ~actor_valid):
        raise RiskEntrySidecarStorageError("inactive actor has valid lane state")
    if np.any(lane_index[~lane_valid] != -1) or np.any(lane_state[~lane_valid] != 0.0):
        raise RiskEntrySidecarStorageError("invalid lane state must use -1 and zero values")
    if np.any(lane_index[lane_valid] < 0) or np.any(
        lane_index[lane_valid] >= len(lane_records)
    ):
        raise RiskEntrySidecarStorageError("lane_index points outside the lane table")
    if np.any(lane_state[..., 3][lane_valid] <= 0.0):
        raise RiskEntrySidecarStorageError("valid lane observations require positive width")
    for record in actor_records:
        active_steps = np.flatnonzero(actor_valid[:, record.actor_index])
        if active_steps.size == 0 or int(active_steps[0]) != record.first_seen_step:
            raise RiskEntrySidecarStorageError("actor first_seen_step mismatch")
        appearance = actor_valid[:, record.actor_index] & np.concatenate(
            (np.asarray([True]), ~actor_valid[:-1, record.actor_index])
        )
        if np.any(state_valid[appearance, record.actor_index, 5:]):
            raise RiskEntrySidecarStorageError("derivatives are valid on actor appearance")
    if not np.all(actor_valid[0, :3]):
        raise RiskEntrySidecarStorageError("initial frame lacks P0/P1/P2")

    base_steps = np.asarray(arrays["base_sample_step_index"])
    if base_steps.ndim != 1 or (
        len(base_steps)
        and (
            np.any(np.diff(base_steps) <= 0)
            or int(base_steps[0]) < 0
            or int(base_steps[-1]) >= timeline_length
        )
    ):
        raise RiskEntrySidecarStorageError("invalid base sample step mapping")

    actor_ids = {item.actor_id for item in actor_records}
    key_actor_ids = metadata.get("key_actor_ids")
    if not isinstance(key_actor_ids, Mapping) or any(
        not isinstance(key, str) or not key or value not in actor_ids
        for key, value in key_actor_ids.items()
    ):
        raise RiskEntrySidecarStorageError("invalid key_actor_ids mapping")
    event_rows = metadata.get("events")
    if not isinstance(event_rows, list):
        raise RiskEntrySidecarStorageError("events must be an array")
    previous_event_step = -1
    events: list[SidecarRawEvent] = []
    for row in event_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "event_type",
            "step_index",
            "timestamp_s",
            "actor_ids",
            "terminal",
            "details",
        }:
            raise RiskEntrySidecarStorageError("event field set mismatch")
        if (
            not isinstance(row["actor_ids"], list)
            or not isinstance(row["terminal"], bool)
            or not isinstance(row["details"], Mapping)
        ):
            raise RiskEntrySidecarStorageError("event field types mismatch")
        try:
            details = dict(row["details"])
            _jsonable(details)
            event = SidecarRawEvent(
                event_type=str(row["event_type"]),
                step_index=int(row["step_index"]),
                timestamp_s=float(row["timestamp_s"]),
                actor_ids=tuple(str(value) for value in row["actor_ids"]),
                terminal=bool(row["terminal"]),
                details=details,
            )
        except (TypeError, ValueError) as exc:
            raise RiskEntrySidecarStorageError("invalid event row") from exc
        if (
            event.step_index < previous_event_step
            or event.step_index >= timeline_length
            or not math.isclose(
                event.timestamp_s,
                float(timestamps[event.step_index]),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or any(actor_id not in actor_ids for actor_id in event.actor_ids)
        ):
            raise RiskEntrySidecarStorageError("event timeline or actor reference mismatch")
        previous_event_step = event.step_index
        events.append(event)
    retention = metadata.get("retention")
    if not isinstance(retention, Mapping) or set(retention) != {
        "outcome",
        "kept_despite_dangerous_outcome",
    }:
        raise RiskEntrySidecarStorageError("retention field set mismatch")
    outcome = str(retention.get("outcome", ""))
    if (
        outcome not in OUTCOMES
        or outcome != _derive_outcome(events)
        or not isinstance(retention.get("kept_despite_dangerous_outcome"), bool)
        or bool(retention.get("kept_despite_dangerous_outcome"))
        != (outcome in DANGEROUS_OUTCOMES)
    ):
        raise RiskEntrySidecarStorageError("retention outcome does not match raw events")


class _SidecarEpisodeBuffer:
    def __init__(self, metadata: SidecarEpisodeStart) -> None:
        self.metadata = metadata
        self.frames: list[SidecarRawFrame] = []
        self.events: list[SidecarRawEvent] = []
        self.actor_records: dict[int, SidecarActorRecord] = {}
        self.lane_records: dict[int, SidecarLaneRecord] = {}
        self.key_actor_ids: dict[str, str] = {}

    def update_registry(
        self,
        *,
        actor_records: Sequence[SidecarActorRecord],
        lane_records: Sequence[SidecarLaneRecord],
        key_actor_ids: Mapping[str, str],
    ) -> None:
        for record in actor_records:
            existing = self.actor_records.get(record.actor_index)
            if existing is not None and existing != record:
                raise RiskEntrySidecarStorageError("actor registry row changed")
            self.actor_records[record.actor_index] = record
        for record in lane_records:
            existing = self.lane_records.get(record.lane_index)
            if existing is not None and existing != record:
                raise RiskEntrySidecarStorageError("lane registry row changed")
            self.lane_records[record.lane_index] = record
        for role, actor_id in key_actor_ids.items():
            previous = self.key_actor_ids.get(str(role))
            if previous is not None and previous != str(actor_id):
                raise RiskEntrySidecarStorageError("key actor identity changed")
            self.key_actor_ids[str(role)] = str(actor_id)

    def append_frame(
        self,
        *,
        step_index: int,
        timestamp_s: float,
        actors: Sequence[SidecarActorSnapshot],
    ) -> None:
        expected_step = len(self.frames)
        expected_time = expected_step * float(self.metadata.decision_dt_s)
        if step_index != expected_step or not math.isclose(
            timestamp_s, expected_time, rel_tol=0.0, abs_tol=1e-6
        ):
            raise RiskEntrySidecarStorageError("raw timeline gap or timestamp mismatch")
        snapshots = tuple(actors)
        if len({item.actor_id for item in snapshots}) != len(snapshots):
            raise RiskEntrySidecarStorageError("duplicate actor in raw frame")
        self.frames.append(
            SidecarRawFrame(
                step_index=int(step_index),
                timestamp_s=float(timestamp_s),
                actors=snapshots,
            )
        )

    def append_event(self, event: SidecarRawEvent) -> None:
        if not self.frames or event.step_index >= len(self.frames):
            raise RiskEntrySidecarStorageError("event must reference an appended raw frame")
        if self.events and event.step_index < self.events[-1].step_index:
            raise RiskEntrySidecarStorageError("events must be appended in timeline order")
        expected_time = self.frames[event.step_index].timestamp_s
        if not math.isclose(event.timestamp_s, expected_time, rel_tol=0.0, abs_tol=1e-6):
            raise RiskEntrySidecarStorageError("event timestamp does not match its raw frame")
        self.events.append(event)


class SidecarSplitWriter:
    def __init__(self, split_root: Path, split: str, base_dataset_fingerprint: str) -> None:
        if split not in SPLIT_NAMES:
            raise RiskEntrySidecarStorageError(f"unknown split: {split}")
        self.split_root = split_root
        self.split = split
        self.base_dataset_fingerprint = base_dataset_fingerprint
        self.episodes_root = split_root / "episodes"
        self.manifest_path = split_root / "manifest.json"
        self.episodes_root.mkdir(parents=True, exist_ok=True)
        self.episodes: dict[int, StoredSidecarEpisode] = {}
        self.ignored_temporary_directories: tuple[str, ...] = ()

    @staticmethod
    def episode_name(episode_index: int) -> str:
        return f"episode_{episode_index:08d}"

    def validate_episode_directory(self, path: Path) -> StoredSidecarEpisode:
        match = EPISODE_PATTERN.fullmatch(path.name)
        if match is None:
            raise RiskEntrySidecarStorageError(f"unexpected episode directory: {path}")
        metadata = _read_json(path / "episode.json")
        episode_index = int(match.group(1))
        if (
            int(metadata.get("episode_index", -1)) != episode_index
            or metadata.get("split") != self.split
            or metadata.get("base_dataset_fingerprint") != self.base_dataset_fingerprint
        ):
            raise RiskEntrySidecarStorageError("episode directory identity mismatch")
        actual_names = {item.name for item in path.iterdir()}
        if actual_names != EPISODE_FILE_NAMES:
            raise RiskEntrySidecarStorageError(
                f"episode file set mismatch: missing={sorted(EPISODE_FILE_NAMES - actual_names)}, "
                f"unexpected={sorted(actual_names - EPISODE_FILE_NAMES)}"
            )
        arrays: dict[str, np.ndarray] = {}
        for name in SIDECAR_ARRAY_DTYPES:
            array_path = path / f"{name}.npy"
            try:
                array = np.load(array_path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise RiskEntrySidecarStorageError(
                    f"unable to mmap sidecar array: {array_path}"
                ) from exc
            if not isinstance(array, np.memmap):
                raise RiskEntrySidecarStorageError("sidecar arrays must be mmap-readable")
            arrays[name] = array
        validate_sidecar_episode_payload(metadata, arrays)
        return StoredSidecarEpisode(
            episode_index=episode_index,
            split=self.split,
            directory=path.name,
            raw_steps=int(len(arrays["step_index"])),
            actor_count=int(arrays["actor_state"].shape[1]),
            base_samples=int(len(arrays["base_sample_step_index"])),
            outcome=str(metadata["retention"]["outcome"]),
        )

    def scan(self, *, rebuild_manifest: bool) -> dict[int, StoredSidecarEpisode]:
        allowed = {"episodes", "manifest.json"}
        unexpected = []
        for item in self.split_root.iterdir():
            if item.name in allowed or item.name.startswith(".manifest.json.tmp-"):
                continue
            unexpected.append(item.name)
        if unexpected:
            raise RiskEntrySidecarStorageError(
                f"unexpected entries in {self.split} split: {sorted(unexpected)}"
            )
        if self.manifest_path.exists():
            _read_json(self.manifest_path)
        episodes: dict[int, StoredSidecarEpisode] = {}
        temporary: list[str] = []
        for path in sorted(self.episodes_root.iterdir()):
            if path.is_dir() and path.name.startswith(".episode_") and ".tmp-" in path.name:
                temporary.append(path.name)
                continue
            if not path.is_dir():
                raise RiskEntrySidecarStorageError(f"unexpected episode-root file: {path}")
            episode = self.validate_episode_directory(path)
            if episode.episode_index in episodes:
                raise RiskEntrySidecarStorageError("duplicate episode index in split")
            episodes[episode.episode_index] = episode
        self.episodes = episodes
        self.ignored_temporary_directories = tuple(temporary)
        if rebuild_manifest:
            self.write_manifest()
        return dict(episodes)

    def manifest_payload(self) -> dict[str, object]:
        entries = [self.episodes[index].manifest_entry() for index in sorted(self.episodes)]
        for entry in entries:
            entry.pop("split", None)
        return {
            "format": SIDECAR_FORMAT,
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "split": self.split,
            "episode_count": len(entries),
            "raw_steps": sum(int(item["raw_steps"]) for item in entries),
            "base_samples": sum(int(item["base_samples"]) for item in entries),
            "episodes": entries,
        }

    def write_manifest(self) -> None:
        _atomic_write_json(self.manifest_path, self.manifest_payload())

    def commit(
        self,
        buffer: _SidecarEpisodeBuffer,
        *,
        base_sample_step_indices: Sequence[int],
    ) -> StoredSidecarEpisode:
        episode_index = int(buffer.metadata.episode_index)
        if episode_index in self.episodes:
            raise RiskEntrySidecarStorageError(f"episode {episode_index} already exists")
        final_path = self.episodes_root / self.episode_name(episode_index)
        if final_path.exists():
            raise RiskEntrySidecarStorageError(f"episode path already exists: {final_path}")
        arrays = _build_dense_arrays(
            metadata=buffer.metadata,
            frames=buffer.frames,
            actor_records=tuple(buffer.actor_records.values()),
            lane_records=tuple(buffer.lane_records.values()),
            base_sample_step_indices=base_sample_step_indices,
        )
        actor_records = _validate_actor_records(
            tuple(buffer.actor_records.values()), len(buffer.frames)
        )
        lane_records = _validate_lane_records(tuple(buffer.lane_records.values()))
        actor_ids = {item.actor_id for item in actor_records}
        if any(value not in actor_ids for value in buffer.key_actor_ids.values()):
            raise RiskEntrySidecarStorageError("key_actor_ids references an unknown actor")
        events = tuple(buffer.events)
        outcome = _derive_outcome(events)
        metadata = {
            "format": SIDECAR_FORMAT,
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "complete": True,
            "episode_index": episode_index,
            "split": self.split,
            "base_dataset_fingerprint": buffer.metadata.base_dataset_fingerprint,
            "decision_dt_s": float(buffer.metadata.decision_dt_s),
            "scenario_id": buffer.metadata.scenario_id,
            "local_route": buffer.metadata.local_route,
            "spawn_seed": int(buffer.metadata.spawn_seed),
            "scenario_parameters": _jsonable(dict(buffer.metadata.scenario_parameters)),
            "actors": [_jsonable(asdict(item)) for item in actor_records],
            "lanes": [_jsonable(asdict(item)) for item in lane_records],
            "key_actor_ids": _jsonable(dict(sorted(buffer.key_actor_ids.items()))),
            "events": [_jsonable(asdict(item)) for item in events],
            "retention": {
                "outcome": outcome,
                "kept_despite_dangerous_outcome": outcome in DANGEROUS_OUTCOMES,
            },
        }
        validate_sidecar_episode_payload(metadata, arrays)
        temporary = self.episodes_root / (
            f".{self.episode_name(episode_index)}.tmp-{uuid.uuid4().hex}"
        )
        temporary.mkdir()
        for name, array in arrays.items():
            output_path = temporary / f"{name}.npy"
            with output_path.open("wb") as stream:
                np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
        _atomic_write_json(temporary / "episode.json", metadata)
        _fsync_directory(temporary)
        os.replace(temporary, final_path)
        _fsync_directory(self.episodes_root)
        episode = self.validate_episode_directory(final_path)
        self.episodes[episode_index] = episode
        self.write_manifest()
        return episode


class RiskEntrySidecarDatasetStore:
    """Single-active-episode sink with atomic split-local persistence."""

    CONTRACT_FILE = "dataset_contract.json"
    LOCK_FILE = ".writer.lock"

    def __init__(
        self,
        dataset_root: Path | str,
        *,
        base_dataset_fingerprint: str,
        resume: bool,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser()
        self.base_dataset_fingerprint = str(base_dataset_fingerprint)
        self.contract = sidecar_dataset_contract(self.base_dataset_fingerprint)
        self.dataset_fingerprint = sidecar_dataset_fingerprint(
            self.base_dataset_fingerprint
        )
        existed = self.dataset_root.exists()
        existing_entries = set(self.dataset_root.iterdir()) if existed else set()
        if existing_entries and not resume:
            raise RiskEntrySidecarStorageError(
                "sidecar root is not empty and resume is disabled"
            )
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self._lock_stream = (self.dataset_root / self.LOCK_FILE).open("a+b")
        try:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_stream.close()
            raise RiskEntrySidecarStorageError(
                "another sidecar writer is active for this root"
            ) from exc
        self.contract_path = self.dataset_root / self.CONTRACT_FILE
        self.writers = {
            split: SidecarSplitWriter(
                self.dataset_root / split, split, self.base_dataset_fingerprint
            )
            for split in SPLIT_NAMES
        }
        self._active: _SidecarEpisodeBuffer | None = None
        self.last_rejection_reason: str | None = None
        try:
            if resume:
                self._load_contract()
            else:
                self._initialize_contract()
            self._scan()
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "RiskEntrySidecarDatasetStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc, traceback
        has_unfinished_episode = self._active is not None
        self.close()
        if exc_type is None and has_unfinished_episode:
            raise RiskEntrySidecarStorageError(
                "sidecar store closed with an unfinished active episode"
            )

    def close(self) -> None:
        self._active = None
        stream = getattr(self, "_lock_stream", None)
        if stream is None or stream.closed:
            return
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()

    def _initialize_contract(self) -> None:
        allowed = {self.LOCK_FILE, *SPLIT_NAMES}
        unexpected = {path.name for path in self.dataset_root.iterdir()} - allowed
        if unexpected:
            raise RiskEntrySidecarStorageError(
                f"new sidecar root has unexpected entries: {sorted(unexpected)}"
            )
        _atomic_write_json(self.contract_path, self.contract)

    def _load_contract(self) -> None:
        if not self.contract_path.is_file() or _read_json(self.contract_path) != self.contract:
            raise RiskEntrySidecarStorageError("sidecar resume contract mismatch")
        allowed = {self.LOCK_FILE, self.CONTRACT_FILE, *SPLIT_NAMES}
        unexpected = []
        for path in self.dataset_root.iterdir():
            if path.name in allowed or path.name.startswith(f".{self.CONTRACT_FILE}.tmp-"):
                continue
            unexpected.append(path.name)
        if unexpected:
            raise RiskEntrySidecarStorageError(
                f"sidecar root has unexpected entries: {sorted(unexpected)}"
            )

    def _scan(self) -> None:
        owner: dict[int, str] = {}
        for split, writer in self.writers.items():
            for episode_index in writer.scan(rebuild_manifest=True):
                previous = owner.get(episode_index)
                if previous is not None:
                    raise RiskEntrySidecarStorageError(
                        f"episode {episode_index} exists in both {previous} and {split}"
                    )
                owner[episode_index] = split

    def begin_episode(self, metadata: SidecarEpisodeStart) -> None:
        if self._active is not None:
            raise RiskEntrySidecarStorageError("an episode is already active")
        if metadata.base_dataset_fingerprint != self.base_dataset_fingerprint:
            raise RiskEntrySidecarStorageError("episode/base dataset fingerprint mismatch")
        if any(
            metadata.episode_index in writer.episodes for writer in self.writers.values()
        ):
            raise RiskEntrySidecarStorageError(
                f"episode {metadata.episode_index} already exists"
            )
        self._active = _SidecarEpisodeBuffer(metadata)
        self.last_rejection_reason = None

    def update_registry(
        self,
        *,
        actor_records: Sequence[SidecarActorRecord],
        lane_records: Sequence[SidecarLaneRecord],
        key_actor_ids: Mapping[str, str],
    ) -> None:
        self._require_active().update_registry(
            actor_records=actor_records,
            lane_records=lane_records,
            key_actor_ids=key_actor_ids,
        )

    def append_frame(
        self,
        *,
        step_index: int,
        timestamp_s: float,
        actors: Sequence[SidecarActorSnapshot],
    ) -> None:
        self._require_active().append_frame(
            step_index=step_index, timestamp_s=timestamp_s, actors=actors
        )

    def append_event(self, event: SidecarRawEvent) -> None:
        self._require_active().append_event(event)

    def append_capture(
        self,
        *,
        frame: SidecarRawFrame,
        events: Sequence[SidecarRawEvent],
        actor_records: Sequence[SidecarActorRecord],
        lane_records: Sequence[SidecarLaneRecord],
        key_actor_ids: Mapping[str, str],
    ) -> None:
        self.update_registry(
            actor_records=actor_records,
            lane_records=lane_records,
            key_actor_ids=key_actor_ids,
        )
        self.append_frame(
            step_index=frame.step_index,
            timestamp_s=frame.timestamp_s,
            actors=frame.actors,
        )
        for event in events:
            self.append_event(event)

    def commit_episode(
        self, *, base_sample_step_indices: Sequence[int]
    ) -> StoredSidecarEpisode:
        buffer = self._require_active()
        writer = self.writers[buffer.metadata.split]
        episode = writer.commit(
            buffer, base_sample_step_indices=base_sample_step_indices
        )
        self._active = None
        return episode

    def reject_episode(self, *, reason_code: str) -> None:
        self._require_active()
        reason = str(reason_code).strip()
        if not reason:
            raise RiskEntrySidecarStorageError("rejection reason must be non-empty")
        self.last_rejection_reason = reason
        self._active = None

    def _require_active(self) -> _SidecarEpisodeBuffer:
        if self._active is None:
            raise RiskEntrySidecarStorageError("no sidecar episode is active")
        return self._active

    def summary(self) -> dict[str, object]:
        return {
            "format": SIDECAR_FORMAT,
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "base_dataset_fingerprint": self.base_dataset_fingerprint,
            "sidecar_dataset_fingerprint": self.dataset_fingerprint,
            "splits": {
                split: {
                    "episodes": len(writer.episodes),
                    "raw_steps": sum(item.raw_steps for item in writer.episodes.values()),
                    "base_samples": sum(item.base_samples for item in writer.episodes.values()),
                    "ignored_temporary_directories": list(
                        writer.ignored_temporary_directories
                    ),
                }
                for split, writer in self.writers.items()
            },
        }


__all__ = [
    "ACTOR_STATE_CHANNELS",
    "EPISODE_FILE_NAMES",
    "LANE_STATE_CHANNELS",
    "RiskEntrySidecarDatasetStore",
    "RiskEntrySidecarStorageError",
    "SIDECAR_ARRAY_DTYPES",
    "SIDECAR_FORMAT",
    "SIDECAR_SCHEMA_VERSION",
    "SidecarEpisodeStart",
    "SidecarSplitWriter",
    "StoredSidecarEpisode",
    "sidecar_dataset_contract",
    "sidecar_dataset_fingerprint",
    "validate_sidecar_episode_payload",
]
