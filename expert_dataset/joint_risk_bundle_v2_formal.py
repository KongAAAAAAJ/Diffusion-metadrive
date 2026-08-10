"""Quota-driven, resumable live collector for formal bundle-v2 datasets.

The pilot and 70k configurations execute this exact module.  They differ only
in frozen YAML values such as quota, output root, and formal eligibility.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import shutil
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from expert_dataset.collect_joint_bev import JointBEVSample, SensorlessJointBEVPlatoonEnv
from expert_dataset.joint_bev_storage import EpisodeSplitAssigner, EpisodeSplitConfig
from expert_dataset.joint_risk_bundle_v2_fixture import (
    BUNDLE_FORMAT,
    BUNDLE_SCHEMA_VERSION,
    PARTITIONS,
    PROTOCOL_PATH,
    SCENARIO_FAMILIES,
    SEVERITIES,
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
from expert_dataset.joint_risk_bundle_v2_real import (
    FUTURE_STEPS,
    HISTORY_STEPS,
    RealBundleV2Error,
    RealV2EpisodeSpec,
    _env_config,
    _episode_specs,
    base_arrays_from_samples,
    build_real_episode_payload,
    collect_real_v2_episode,
    eligible_anchor_violation,
)
from scenarios.definitions import get_scenario_definition


FORMAL_TARGETS = {
    "id": 50_000,
    "compositional_ood": 10_000,
    "topology_ood": 10_000,
}
CELL_ORDER = tuple(
    (family, severity)
    for family in SCENARIO_FAMILIES
    for severity in SEVERITIES
)
RUN_MODES = ("formal_pilot", "formal")
COLLECTOR_FORMAT = "metadrive-joint-risk-bundle-v2-formal-collector"
COLLECTOR_SCHEMA_VERSION = 1


class FormalV2CollectionError(RuntimeError):
    """Raised for a formal-path configuration, quota, or resume violation."""


@dataclass(frozen=True)
class FormalV2Config:
    output_root: Path
    run_mode: str
    resume: bool
    split_seed: int
    max_episode_steps: int
    anchor_steps: tuple[int, ...]
    max_attempts_per_cell: int
    target_windows: Mapping[str, int]
    dataset_instance_prefix: str

    def __post_init__(self) -> None:
        if self.run_mode not in RUN_MODES:
            raise FormalV2CollectionError(f"invalid run_mode: {self.run_mode}")
        if not self.resume:
            raise FormalV2CollectionError("formal v2 collection requires resume=true")
        if set(self.target_windows) != set(PARTITIONS):
            raise FormalV2CollectionError("target_windows must define all three partitions")
        normalized_targets: dict[str, int] = {}
        for partition in PARTITIONS:
            target = self.target_windows[partition]
            if isinstance(target, bool) or int(target) <= 0 or int(target) % len(CELL_ORDER):
                raise FormalV2CollectionError(
                    f"{partition} target must be a positive multiple of eight"
                )
            normalized_targets[partition] = int(target)
        object.__setattr__(self, "target_windows", normalized_targets)
        anchors = tuple(sorted({int(step) for step in self.anchor_steps}))
        if not anchors or anchors[0] < HISTORY_STEPS:
            raise FormalV2CollectionError("anchor steps must provide 2 s history")
        if anchors[-1] + FUTURE_STEPS > self.max_episode_steps:
            raise FormalV2CollectionError("anchor steps exceed the configured 5 s future")
        object.__setattr__(self, "anchor_steps", anchors)
        if self.max_episode_steps <= 0 or self.max_attempts_per_cell <= 0:
            raise FormalV2CollectionError("episode/attempt limits must be positive")
        if not self.dataset_instance_prefix.strip():
            raise FormalV2CollectionError("dataset_instance_prefix must be non-empty")
        if self.run_mode == "formal" and normalized_targets != FORMAL_TARGETS:
            raise FormalV2CollectionError(
                "formal mode must use the frozen 50k/10k/10k target"
            )

    @property
    def eligible_for_formal_training(self) -> bool:
        return self.run_mode == "formal"

    def quota_per_cell(self, partition: str) -> int:
        return int(self.target_windows[partition]) // len(CELL_ORDER)

    def frozen_payload(self) -> dict[str, object]:
        return {
            "format": COLLECTOR_FORMAT,
            "schema_version": COLLECTOR_SCHEMA_VERSION,
            "run_mode": self.run_mode,
            "resume": self.resume,
            "split_seed": self.split_seed,
            "max_episode_steps": self.max_episode_steps,
            "anchor_steps": list(self.anchor_steps),
            "max_attempts_per_cell": self.max_attempts_per_cell,
            "target_windows": dict(self.target_windows),
            "dataset_instance_prefix": self.dataset_instance_prefix,
            "eligible_for_formal_training": self.eligible_for_formal_training,
            "formal_targets": FORMAL_TARGETS,
        }


def load_formal_v2_config(path: Path | str) -> FormalV2Config:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"dataset", "collection"}:
        raise FormalV2CollectionError("config must contain only dataset and collection")
    dataset = payload["dataset"]
    collection = payload["collection"]
    if not isinstance(dataset, Mapping) or set(dataset) != {
        "output_root",
        "dataset_instance_prefix",
    }:
        raise FormalV2CollectionError("invalid dataset section")
    if not isinstance(collection, Mapping) or set(collection) != {
        "run_mode",
        "resume",
        "split_seed",
        "max_episode_steps",
        "anchor_steps",
        "max_attempts_per_cell",
        "target_windows",
    }:
        raise FormalV2CollectionError("invalid collection section")
    target_windows = collection["target_windows"]
    if not isinstance(target_windows, Mapping):
        raise FormalV2CollectionError("target_windows must be a mapping")
    resume = collection["resume"]
    if not isinstance(resume, bool):
        raise FormalV2CollectionError("collection.resume must be a boolean")
    anchor_steps = collection["anchor_steps"]
    if (
        not isinstance(anchor_steps, Sequence)
        or isinstance(anchor_steps, (str, bytes))
    ):
        raise FormalV2CollectionError("collection.anchor_steps must be a sequence")
    return FormalV2Config(
        output_root=Path(str(dataset["output_root"])).expanduser().resolve(),
        dataset_instance_prefix=str(dataset["dataset_instance_prefix"]),
        run_mode=str(collection["run_mode"]),
        resume=resume,
        split_seed=int(collection["split_seed"]),
        max_episode_steps=int(collection["max_episode_steps"]),
        anchor_steps=tuple(int(value) for value in anchor_steps),
        max_attempts_per_cell=int(collection["max_attempts_per_cell"]),
        target_windows={str(key): int(value) for key, value in target_windows.items()},
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )


def _read_index(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if [int(row["episode_index"]) for row in rows] != list(range(len(rows))):
        raise FormalV2CollectionError("bundle episode index is not contiguous")
    return rows


class FormalV2PartitionWriter:
    """Single-writer atomic episode transaction and resume boundary."""

    def __init__(self, config: FormalV2Config, partition: str) -> None:
        self.config = config
        self.partition = partition
        self.root = config.output_root / f"{config.dataset_instance_prefix}_{partition}_v2"
        self.base_root = self.root / "platoon_joint_bev"
        self.sidecar_root = self.root / "riskentry_actor_sidecar"
        self.index_path = self.root / "bundle_episode_index.jsonl"
        self.state_path = self.root / "formal_collection_state.json"
        self.contract_path = self.root / "formal_collector_contract.json"
        self.pending_root = self.root / ".formal_episode_pending"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.root / ".formal_writer.lock").open("a+b")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            raise FormalV2CollectionError("another formal writer is active") from exc

        partition_payload = config.frozen_payload() | {"partition": partition}
        self.contract = partition_payload | {
            "collector_contract_sha256": _sha256_payload(partition_payload),
            "bundle_protocol_sha256": _sha256_file(PROTOCOL_PATH),
        }
        if self.contract_path.exists():
            observed = json.loads(self.contract_path.read_text(encoding="utf-8"))
            if observed != self.contract:
                self.close()
                raise FormalV2CollectionError("formal resume contract mismatch")
        else:
            unexpected = {
                path.name for path in self.root.iterdir()
            } - {".formal_writer.lock"}
            if unexpected:
                self.close()
                raise FormalV2CollectionError(
                    f"new formal root is not empty: {sorted(unexpected)}"
                )
            _atomic_json(self.contract_path, self.contract)
            _atomic_bytes(self.index_path, b"")

        self.scenario_contract = self._scenario_contract()
        self.scenario_sha = _sha256_payload(self.scenario_contract)
        self.base_fingerprint = _sha256_payload(
            {
                "collector_contract_sha256": self.contract["collector_contract_sha256"],
                "partition": partition,
                "scenario_contract_sha256": self.scenario_sha,
            }
        )
        self.sidecar_contract = _sidecar_contract(self.base_fingerprint)
        self.sidecar_fingerprint = _sha256_payload(self.sidecar_contract)
        self.allowed_splits = ("train", "val", "test") if partition == "id" else ("test",)
        self.base_root.mkdir(exist_ok=True)
        self.sidecar_root.mkdir(exist_ok=True)
        (self.base_root / ".writer.lock").touch(exist_ok=True)
        self._validate_or_write_contracts()
        self._recover_pending()
        self.rows = _read_index(self.index_path)
        self._materialize_metadata()

    def close(self) -> None:
        stream = getattr(self, "_lock", None)
        if stream is not None and not stream.closed:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()

    def __enter__(self) -> "FormalV2PartitionWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()

    def _scenario_contract(self) -> dict[str, object]:
        scenario_definitions = []
        for spec in _episode_specs(self.partition):
            definition = get_scenario_definition(spec.scenario_id)
            scenario_definitions.append(
                {
                    "scenario_family": spec.scenario_family,
                    "severity": spec.severity,
                    "scenario_id": spec.scenario_id,
                    "local_route": spec.local_route,
                    "definition_sha256": _sha256_payload(asdict(definition)),
                }
            )
        return {
            "format": "metadrive-riskentry-formal-v2-scenario-contract",
            "schema_version": 1,
            "run_mode": self.config.run_mode,
            "eligible_for_formal_training": self.config.eligible_for_formal_training,
            "benchmark_partition": self.partition,
            "scenario_families": list(SCENARIO_FAMILIES),
            "severities": list(SEVERITIES),
            "decision_dt_s": 0.1,
            "history_seconds": 2.0,
            "future_seconds": 5.0,
            "anchor_steps": list(self.config.anchor_steps),
            "target_windows": int(self.config.target_windows[self.partition]),
            "quota_per_cell": self.config.quota_per_cell(self.partition),
            "visitation_policy": "stable_lane_follow_6mps",
            "label_policy": "fresh_rulemaker_normal_planner_per_anchor",
            "online_observation_policy": "ideal_current_state_range_80m_v1",
            "communication_policy": "always_available_50ms_v1",
            "scenario_definitions": scenario_definitions,
        }

    def _base_contract_payload(self) -> dict[str, object]:
        payload = _base_contract(self.partition, self.base_fingerprint)
        split_assignment = dict(payload["split_assignment"])
        split_assignment["seed"] = self.config.split_seed
        split_assignment["unit"] = "matched_pair_id"
        payload["split_assignment"] = split_assignment
        return payload

    def _validate_or_write_contracts(self) -> None:
        expected = {
            self.base_root / "dataset_contract.json": self._base_contract_payload(),
            self.sidecar_root / "dataset_contract.json": self.sidecar_contract,
            self.root / "scenario_contract.json": self.scenario_contract,
        }
        for path, payload in expected.items():
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")) != payload:
                    raise FormalV2CollectionError(f"resume contract mismatch: {path}")
            else:
                _atomic_json(path, payload)

    def _recover_pending(self) -> None:
        if not self.pending_root.exists():
            return
        record_path = self.pending_root / "record.json"
        if not record_path.is_file():
            raise FormalV2CollectionError("pending formal transaction has no record")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        episode_index = int(record["episode_index"])
        split = str(record["split"])
        episode_name = f"episode_{episode_index:08d}"
        for component, root in (("base", self.base_root), ("sidecar", self.sidecar_root)):
            source = self.pending_root / component
            target = root / split / "episodes" / episode_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                if not source.exists():
                    raise FormalV2CollectionError("pending transaction lost a component")
                os.replace(source, target)
        rows = _read_index(self.index_path)
        if episode_index == len(rows):
            rows.append(dict(record["index_row"]))
            _atomic_bytes(
                self.index_path,
                "".join(_canonical_json(row) + "\n" for row in rows).encode("utf-8"),
            )
        elif episode_index >= len(rows) or rows[episode_index] != record["index_row"]:
            raise FormalV2CollectionError("pending transaction conflicts with bundle index")
        shutil.rmtree(self.pending_root)

    def cell_counts(self) -> Counter[tuple[str, str]]:
        counts: Counter[tuple[str, str]] = Counter()
        for row in self.rows:
            if row["base_status"] != "committed":
                continue
            path = (
                self.base_root
                / str(row["split"])
                / "episodes"
                / f"episode_{int(row['episode_index']):08d}"
                / "episode.json"
            )
            metadata = json.loads(path.read_text(encoding="utf-8"))
            attributes = metadata["attributes"]
            counts[(attributes["scenario_family"], attributes["severity"])] += int(
                metadata["joint_samples"]
            )
        return counts

    def matched_pair_records(
        self, family: str
    ) -> dict[str, dict[str, dict[str, object]]]:
        pairs: dict[str, dict[str, dict[str, object]]] = {}
        for row in self.rows:
            path = (
                self.base_root
                / str(row["split"])
                / "episodes"
                / f"episode_{int(row['episode_index']):08d}"
                / "episode.json"
            )
            metadata = json.loads(path.read_text(encoding="utf-8"))
            attributes = dict(metadata["attributes"])
            if attributes.get("scenario_family") != family:
                continue
            pair_id = str(attributes.get("matched_pair_id", ""))
            severity = str(attributes.get("severity", ""))
            if not pair_id or severity not in SEVERITIES:
                raise FormalV2CollectionError("invalid matched-pair episode metadata")
            pairs.setdefault(pair_id, {})[severity] = {
                "row": row,
                "attributes": attributes,
            }
        return pairs

    def _materialize_metadata(self) -> None:
        self.rows = _read_index(self.index_path)
        base_rows = {split: [] for split in self.allowed_splits}
        side_rows = {split: [] for split in self.allowed_splits}
        parameter_ids: list[str] = []
        topology_ids: list[str] = []
        for row in self.rows:
            split = str(row["split"])
            episode_name = f"episode_{int(row['episode_index']):08d}"
            if row["base_status"] == "committed":
                base_meta = json.loads(
                    (self.base_root / split / "episodes" / episode_name / "episode.json").read_text(
                        encoding="utf-8"
                    )
                )
                base_rows[split].append(
                    {
                        "episode_index": int(row["episode_index"]),
                        "directory": episode_name,
                        "joint_samples": int(base_meta["joint_samples"]),
                        "attributes": base_meta["attributes"],
                    }
                )
            side_meta = json.loads(
                (self.sidecar_root / split / "episodes" / episode_name / "episode.json").read_text(
                    encoding="utf-8"
                )
            )
            side_rows[split].append(
                {
                    "episode_index": int(row["episode_index"]),
                    "directory": episode_name,
                    "raw_steps": int(row["raw_steps"]),
                    "actor_count": len(side_meta["actors"]),
                    "base_samples": int(row["base_samples"]),
                    "outcome": row["outcome"],
                }
            )
            parameter_ids.append(str(side_meta["parameter_combination_id"]))
            topology_ids.append(str(side_meta["topology_id"]))
        _write_component_manifests(
            root=self.base_root,
            format_name=self._base_contract_payload()["format"],
            schema_version=2,
            allowed_splits=self.allowed_splits,
            rows_by_split=base_rows,
            sidecar=False,
        )
        _write_component_manifests(
            root=self.sidecar_root,
            format_name=SIDECAR_FORMAT,
            schema_version=SIDECAR_SCHEMA_VERSION,
            allowed_splits=self.allowed_splits,
            rows_by_split=side_rows,
            sidecar=True,
        )
        counts = self.cell_counts()
        eligible = int(sum(counts.values()))
        manifest = {
            "format": BUNDLE_FORMAT,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "run_mode": self.config.run_mode,
            "diagnostic_only": not self.config.eligible_for_formal_training,
            "eligible_for_formal_training": self.config.eligible_for_formal_training,
            "physical_source": "live_metadrive_object_registry",
            "dataset_instance_id": f"{self.config.dataset_instance_prefix}_{self.partition}",
            "benchmark_partition": self.partition,
            "bundle_protocol_sha256": _sha256_file(PROTOCOL_PATH),
            "scenario_contract_sha256": self.scenario_sha,
            "base_dataset_fingerprint": self.base_fingerprint,
            "sidecar_dataset_fingerprint": self.sidecar_fingerprint,
            "parameter_space_id": f"formal_v2_{self.partition}_parameters_v1",
            "parameter_tuple_set_sha256": _sha256_payload(sorted(parameter_ids)),
            "topology_set_id": f"formal_v2_{self.partition}_topologies_v1",
            "topology_id_set_sha256": _sha256_payload(sorted(set(topology_ids))),
            "target_eligible_anchor_windows": int(self.config.target_windows[self.partition]),
            "eligible_anchor_window_count": eligible,
            "counting_unit": "eligible_anchor_windows_with_2s_history_and_valid_5s_future",
            "scenario_catalog": list(SCENARIO_FAMILIES),
            "episode_split_manifest": {
                split: [int(row["episode_index"]) for row in rows]
                for split, rows in base_rows.items()
            },
            "base_directory": "platoon_joint_bev",
            "sidecar_directory": "riskentry_actor_sidecar",
        }
        _atomic_json(self.root / "dataset_bundle_manifest.json", manifest)
        _atomic_json(
            self.base_root / "collection_state.json",
            {
                "schema_version": 2,
                "next_episode_index": len(self.rows),
                "attempted_episodes": len(self.rows),
                "stored_episodes": len(self.rows),
                "rejected_episodes": 0,
                "total_joint_samples": eligible,
                "rejection_reasons": {},
            },
        )
        _atomic_json(
            self.state_path,
            {
                "format": COLLECTOR_FORMAT,
                "schema_version": COLLECTOR_SCHEMA_VERSION,
                "collector_contract_sha256": self.contract["collector_contract_sha256"],
                "next_episode_index": len(self.rows),
                "attempted_episodes": len(self.rows),
                "eligible_anchor_window_count": eligible,
                "target_eligible_anchor_windows": int(self.config.target_windows[self.partition]),
                "cell_counts": {
                    f"{family}/{severity}": int(counts[(family, severity)])
                    for family, severity in CELL_ORDER
                },
                "complete": eligible == int(self.config.target_windows[self.partition]),
            },
        )

    def commit(
        self,
        *,
        spec: RealV2EpisodeSpec,
        metadata: Mapping[str, object],
        side_arrays: Mapping[str, np.ndarray],
        samples: Sequence[JointBEVSample],
        sample_steps: Sequence[int],
    ) -> None:
        if self.pending_root.exists():
            raise FormalV2CollectionError("another formal episode transaction is pending")
        if spec.episode_index != len(self.rows):
            raise FormalV2CollectionError("formal episode index is not contiguous")
        if not samples or len(samples) != len(sample_steps):
            raise FormalV2CollectionError("formal episode has no aligned eligible samples")
        episode_name = f"episode_{spec.episode_index:08d}"
        pending_base = self.pending_root / "base"
        pending_side = self.pending_root / "sidecar"
        base_attributes = {
            "scenario_id": spec.scenario_id,
            "local_route": spec.local_route,
            "spawn_seed": spec.spawn_seed,
            "sidecar_dataset_fingerprint": self.sidecar_fingerprint,
            "selected_sample_steps": [int(step) for step in sample_steps],
            "benchmark_partition": self.partition,
            "scenario_family": spec.scenario_family,
            "severity": spec.severity,
            "matched_pair_id": spec.matched_pair_id,
            "run_mode": self.config.run_mode,
            "diagnostic_only": not self.config.eligible_for_formal_training,
            "eligible_for_formal_training": self.config.eligible_for_formal_training,
        }
        _write_json(
            pending_base / "episode.json",
            {
                "format": self._base_contract_payload()["format"],
                "schema_version": 2,
                "complete": True,
                "episode_index": spec.episode_index,
                "split": spec.split,
                "joint_samples": len(samples),
                "attributes": base_attributes,
            },
        )
        for name, array in base_arrays_from_samples(samples).items():
            _write_npy(pending_base / f"{name}.npy", array)
        _write_json(pending_side / "episode.json", metadata)
        for name, array in side_arrays.items():
            _write_npy(pending_side / f"{name}.npy", array)
        index_row = {
            "episode_index": spec.episode_index,
            "split": spec.split,
            "scenario_id": spec.scenario_id,
            "matched_pair_id": spec.matched_pair_id,
            "local_route": spec.local_route,
            "spawn_seed": spec.spawn_seed,
            "base_status": "committed",
            "base_rejection_reason": None,
            "sidecar_status": "committed",
            "sidecar_rejection_reason": None,
            "raw_steps": int(len(side_arrays["step_index"])),
            "base_samples": len(samples),
            "outcome": str(metadata["retention"]["outcome"]),
        }
        _write_json(
            self.pending_root / "record.json",
            {
                "episode_index": spec.episode_index,
                "split": spec.split,
                "index_row": index_row,
            },
        )
        target_base = self.base_root / spec.split / "episodes" / episode_name
        target_side = self.sidecar_root / spec.split / "episodes" / episode_name
        target_base.parent.mkdir(parents=True, exist_ok=True)
        target_side.parent.mkdir(parents=True, exist_ok=True)
        os.replace(pending_base, target_base)
        os.replace(pending_side, target_side)
        rows = [*self.rows, index_row]
        _atomic_bytes(
            self.index_path,
            "".join(_canonical_json(row) + "\n" for row in rows).encode("utf-8"),
        )
        shutil.rmtree(self.pending_root)
        self.rows = rows
        self._materialize_metadata()


def _split_for(config: FormalV2Config, partition: str, matched_pair_id: str) -> str:
    if partition != "id":
        return "test"
    return EpisodeSplitAssigner(
        EpisodeSplitConfig(0.8, 0.1, 0.1, config.split_seed)
    ).split_for_key(matched_pair_id)


def _episode_spec(
    config: FormalV2Config,
    partition: str,
    family: str,
    severity: str,
    *,
    episode_index: int,
    cell_episode_index: int,
    attempt: int,
) -> RealV2EpisodeSpec:
    template = next(
        row
        for row in _episode_specs(partition)
        if row.scenario_family == family and row.severity == severity
    )
    partition_offset = PARTITIONS.index(partition) * 10_000_000
    family_offset = SCENARIO_FAMILIES.index(family) * 1_000_000
    seed = (
        1_000_000
        + partition_offset
        + family_offset
        + cell_episode_index * config.max_attempts_per_cell
        + attempt
    )
    matched_pair_id = (
        f"{partition}_{family}_pair_{int(cell_episode_index):06d}"
    )
    return RealV2EpisodeSpec(
        **{
            **asdict(template),
            "episode_index": int(episode_index),
            "split": _split_for(config, partition, matched_pair_id),
            "spawn_seed": int(seed),
            "matched_pair_index": int(cell_episode_index),
        }
    )


def _collect_episode_attempt(
    config: FormalV2Config,
    spec: RealV2EpisodeSpec,
    base_fingerprint: str,
    scenario_sha: str,
    remaining: int,
    *,
    required_steps: Sequence[int] | None = None,
) -> tuple[RealV2EpisodeSpec, dict[str, object], dict[str, np.ndarray], tuple[JointBEVSample, ...], tuple[int, ...]]:
    env_config = _env_config(spec)
    env = SensorlessJointBEVPlatoonEnv(env_config)
    try:
        rollout = collect_real_v2_episode(
            env,
            max_steps=config.max_episode_steps,
            reset_seed=spec.spawn_seed,
            sample_steps=config.anchor_steps,
        )
        metadata, side_arrays, _, _ = build_real_episode_payload(
            spec,
            rollout,
            base_fingerprint=base_fingerprint,
            scenario_contract_sha256=scenario_sha,
            env_config=env_config,
        )
        sample_by_step = dict(zip(rollout.sample_step_indices, rollout.samples))
        eligible_steps = tuple(
            step
            for step in (
                tuple(int(value) for value in required_steps)
                if required_steps is not None
                else config.anchor_steps
            )
            if step in sample_by_step
            and eligible_anchor_violation(spec, rollout, side_arrays, step) is None
        )[: int(remaining)]
        if required_steps is not None and eligible_steps != tuple(required_steps):
            raise FormalV2CollectionError(
                "matched control does not provide the near-critical anchor set"
            )
        if not eligible_steps:
            raise FormalV2CollectionError("episode produced no eligible formal anchors")
        samples = tuple(sample_by_step[step] for step in eligible_steps)
        side_arrays["base_sample_step_index"] = np.asarray(
            eligible_steps, dtype=np.int64
        )
        metadata["diagnostic_only"] = not config.eligible_for_formal_training
        metadata["eligible_for_formal_training"] = config.eligible_for_formal_training
        metadata["run_mode"] = config.run_mode
        metadata["eligible_anchor_steps"] = list(eligible_steps)
        metadata["communication_policy_id"] = "always_available_50ms_v1"
        return spec, metadata, side_arrays, samples, eligible_steps
    finally:
        env.close()


def _isolated_collect_worker(
    sender: object,
    config: FormalV2Config,
    spec: RealV2EpisodeSpec,
    base_fingerprint: str,
    scenario_sha: str,
    remaining: int,
    required_steps: tuple[int, ...] | None,
) -> None:
    try:
        payload = _collect_episode_attempt(
            config,
            spec,
            base_fingerprint,
            scenario_sha,
            remaining,
            required_steps=required_steps,
        )
        sender.send(("ok", payload))
    except BaseException as exc:
        sender.send(
            (
                "error",
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
        )
    finally:
        sender.close()


def _collect_episode_for_quota(
    config: FormalV2Config,
    writer: FormalV2PartitionWriter,
    family: str,
    severity: str,
    remaining: int,
    *,
    cell_episode_index: int | None = None,
    required_steps: Sequence[int] | None = None,
    episode_index: int | None = None,
) -> tuple[RealV2EpisodeSpec, dict[str, object], dict[str, np.ndarray], tuple[JointBEVSample, ...], tuple[int, ...]]:
    if cell_episode_index is None:
        cell_episode_index = sum(
            1
            for row in writer.rows
            if row["scenario_id"]
            == next(
                template.scenario_id
                for template in _episode_specs(writer.partition)
                if template.scenario_family == family
                and template.severity == severity
            )
        )
    last_error: Exception | None = None
    for attempt in range(config.max_attempts_per_cell):
        spec = _episode_spec(
            config,
            writer.partition,
            family,
            severity,
            episode_index=(
                len(writer.rows) if episode_index is None else int(episode_index)
            ),
            cell_episode_index=cell_episode_index,
            attempt=attempt,
        )
        context = mp.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_isolated_collect_worker,
            args=(
                sender,
                config,
                spec,
                writer.base_fingerprint,
                writer.scenario_sha,
                int(remaining),
                (
                    tuple(int(value) for value in required_steps)
                    if required_steps is not None
                    else None
                ),
            ),
        )
        try:
            process.start()
            sender.close()
            status, payload = receiver.recv()
            process.join()
            if status != "ok" or process.exitcode != 0:
                raise FormalV2CollectionError(str(payload))
            spec, metadata, side_arrays, samples, eligible_steps = payload
            print(
                f"[INFO] formal-v2 partition={writer.partition} family={family} "
                f"severity={severity} episode={spec.episode_index} seed={spec.spawn_seed} "
                f"anchors={len(eligible_steps)} raw_steps={len(side_arrays['step_index'])}",
                flush=True,
            )
            return spec, metadata, side_arrays, samples, eligible_steps
        except Exception as exc:
            last_error = exc
            print(
                f"[WARNING] formal-v2 retry partition={writer.partition} family={family} "
                f"severity={severity} attempt={attempt + 1}: {exc}",
                flush=True,
            )
        finally:
            receiver.close()
            if process.is_alive():
                process.terminate()
                process.join()
    raise FormalV2CollectionError(
        f"cell {writer.partition}/{family}/{severity} failed after "
        f"{config.max_attempts_per_cell} attempts: {last_error}"
    )


def _pair_window_allocations(
    *,
    remaining_windows: int,
    anchors_per_episode: int,
    parallel_workers: int,
    remaining_episode_budget: int | None,
) -> tuple[int, ...]:
    """Allocate one bounded window cap to each concurrently collected pair.

    A matched pair consumes exactly two episode indices.  The allocations sum
    to no more than the per-severity cell remainder, so concurrent results can
    never overfill a frozen quota even when every requested anchor is valid.
    """

    if remaining_windows <= 0 or anchors_per_episode <= 0:
        return ()
    if parallel_workers <= 0:
        raise FormalV2CollectionError("parallel_workers must be positive")
    pair_limit = int(parallel_workers)
    if remaining_episode_budget is not None:
        if remaining_episode_budget < 0:
            raise FormalV2CollectionError("remaining episode budget cannot be negative")
        pair_limit = min(pair_limit, int(remaining_episode_budget) // 2)
    allocations: list[int] = []
    unallocated = int(remaining_windows)
    while len(allocations) < pair_limit and unallocated > 0:
        value = min(int(anchors_per_episode), unallocated)
        allocations.append(value)
        unallocated -= value
    return tuple(allocations)


def _collect_matched_pair_for_quota(
    config: FormalV2Config,
    writer: FormalV2PartitionWriter,
    family: str,
    *,
    pair_index: int,
    episode_index: int,
    pair_windows: int,
):
    """Collect one isolated near-critical/control pair without writing it."""

    near = _collect_episode_for_quota(
        config,
        writer,
        family,
        "near_critical",
        int(pair_windows),
        cell_episode_index=int(pair_index),
        episode_index=int(episode_index),
    )
    control = _collect_episode_for_quota(
        config,
        writer,
        family,
        "control",
        len(near[4]),
        cell_episode_index=int(pair_index),
        required_steps=near[4],
        episode_index=int(episode_index) + 1,
    )
    return near, control


def _collect_pair_batch(
    config: FormalV2Config,
    writer: FormalV2PartitionWriter,
    family: str,
    *,
    pair_index_base: int,
    episode_index_base: int,
    allocations: Sequence[int],
    parallel_workers: int,
):
    """Collect independent pairs concurrently and return them in index order.

    Each underlying episode still executes in its own ``spawn`` subprocess.
    Threads only coordinate several independent subprocess/Pipe lifecycles;
    they never mutate dataset state.  The caller is the sole ordered writer.
    """

    caps = tuple(int(value) for value in allocations)
    if not caps:
        return ()
    if any(value <= 0 for value in caps):
        raise FormalV2CollectionError("pair allocations must be positive")
    worker_count = min(int(parallel_workers), len(caps))
    if worker_count <= 0:
        raise FormalV2CollectionError("parallel_workers must be positive")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _collect_matched_pair_for_quota,
                config,
                writer,
                family,
                pair_index=int(pair_index_base) + offset,
                episode_index=int(episode_index_base) + 2 * offset,
                pair_windows=cap,
            )
            for offset, cap in enumerate(caps)
        ]
        return tuple(future.result() for future in futures)


def run_formal_v2_collection(
    config: FormalV2Config,
    *,
    max_new_episodes: int | None = None,
    parallel_workers: int = 1,
) -> dict[str, object]:
    if max_new_episodes is not None and max_new_episodes <= 0:
        raise FormalV2CollectionError("max_new_episodes must be positive")
    if isinstance(parallel_workers, bool) or int(parallel_workers) <= 0:
        raise FormalV2CollectionError("parallel_workers must be a positive integer")
    parallel_workers = int(parallel_workers)
    started_at = time.perf_counter()
    config.output_root.mkdir(parents=True, exist_ok=True)
    run_contract_path = config.output_root / "formal_v2_run_contract.json"
    frozen_payload = config.frozen_payload()
    if run_contract_path.exists():
        observed = json.loads(run_contract_path.read_text(encoding="utf-8"))
        if observed != frozen_payload:
            raise FormalV2CollectionError("formal run resume contract mismatch")
    else:
        unexpected = {path.name for path in config.output_root.iterdir()}
        if unexpected:
            raise FormalV2CollectionError(
                f"new formal output root is not empty: {sorted(unexpected)}"
            )
        _atomic_json(run_contract_path, frozen_payload)
    new_episodes = 0
    new_windows = 0
    summaries: dict[str, object] = {}
    for partition in PARTITIONS:
        with FormalV2PartitionWriter(config, partition) as writer:
            while True:
                counts = writer.cell_counts()
                quota = config.quota_per_cell(partition)
                pending_families = [
                    family
                    for family in SCENARIO_FAMILIES
                    if any(counts[(family, severity)] < quota for severity in SEVERITIES)
                ]
                if not pending_families:
                    break
                remaining_budget = (
                    None
                    if max_new_episodes is None
                    else max_new_episodes - new_episodes
                )
                family = pending_families[0]
                pair_records = writer.matched_pair_records(family)
                unmatched = [
                    (pair_id, records)
                    for pair_id, records in pair_records.items()
                    if set(records) != set(SEVERITIES)
                ]
                if unmatched:
                    if len(unmatched) != 1:
                        raise FormalV2CollectionError(
                            f"multiple unmatched pairs for {partition}/{family}"
                        )
                    if remaining_budget is not None and remaining_budget < 1:
                        break
                    pair_id, records = unmatched[0]
                    if len(records) != 1:
                        raise FormalV2CollectionError(f"invalid pair {pair_id}")
                    existing_severity = next(iter(records))
                    missing_severity = next(
                        severity for severity in SEVERITIES if severity not in records
                    )
                    existing = records[existing_severity]
                    required_steps = tuple(
                        int(step)
                        for step in existing["attributes"]["selected_sample_steps"]
                    )
                    pair_index = int(pair_id.rsplit("_", 1)[1])
                    result = _collect_episode_for_quota(
                        config,
                        writer,
                        family,
                        missing_severity,
                        len(required_steps),
                        cell_episode_index=pair_index,
                        required_steps=required_steps,
                        episode_index=len(writer.rows),
                    )
                    spec, metadata, arrays, samples, sample_steps = result
                    writer.commit(
                        spec=spec,
                        metadata=metadata,
                        side_arrays=arrays,
                        samples=samples,
                        sample_steps=sample_steps,
                    )
                    new_episodes += 1
                    new_windows += len(samples)
                    continue
                remaining = min(
                    quota - counts[(family, severity)] for severity in SEVERITIES
                )
                if remaining <= 0:
                    raise FormalV2CollectionError(
                        f"matched-pair quota diverged for {partition}/{family}"
                    )
                allocations = _pair_window_allocations(
                    remaining_windows=remaining,
                    anchors_per_episode=len(config.anchor_steps),
                    parallel_workers=parallel_workers,
                    remaining_episode_budget=remaining_budget,
                )
                if not allocations:
                    break
                batches = _collect_pair_batch(
                    config,
                    writer,
                    family,
                    pair_index_base=len(pair_records),
                    episode_index_base=len(writer.rows),
                    allocations=allocations,
                    parallel_workers=parallel_workers,
                )
                for near, control in batches:
                    for result in (near, control):
                        spec, metadata, arrays, samples, sample_steps = result
                        writer.commit(
                            spec=spec,
                            metadata=metadata,
                            side_arrays=arrays,
                            samples=samples,
                            sample_steps=sample_steps,
                        )
                        new_episodes += 1
                        new_windows += len(samples)
            state = json.loads(writer.state_path.read_text(encoding="utf-8"))
            summaries[partition] = state
        if max_new_episodes is not None and new_episodes >= max_new_episodes:
            break
    complete = all(
        int(summary.get("eligible_anchor_window_count", -1))
        == int(config.target_windows[partition])
        for partition, summary in summaries.items()
    ) and set(summaries) == set(PARTITIONS)
    elapsed_s = time.perf_counter() - started_at
    report = {
        "format": f"{COLLECTOR_FORMAT}-run-report",
        "schema_version": COLLECTOR_SCHEMA_VERSION,
        "run_mode": config.run_mode,
        "eligible_for_formal_training": config.eligible_for_formal_training,
        "new_episodes": new_episodes,
        "new_eligible_anchor_windows": new_windows,
        "execution": {
            "parallel_workers": parallel_workers,
            "elapsed_s": elapsed_s,
            "episodes_per_hour": (
                0.0 if elapsed_s <= 0.0 else new_episodes * 3600.0 / elapsed_s
            ),
            "eligible_anchor_windows_per_hour": (
                0.0 if elapsed_s <= 0.0 else new_windows * 3600.0 / elapsed_s
            ),
            "episode_isolation": "one_spawn_subprocess_per_episode",
            "commit_order": "episode_index_serial",
        },
        "complete": complete,
        "partitions": summaries,
    }
    _atomic_json(config.output_root / "formal_v2_collection_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-new-episodes", type=int)
    parser.add_argument("--parallel-workers", type=int, default=1)
    args = parser.parse_args(argv)
    report = run_formal_v2_collection(
        load_formal_v2_config(args.config),
        max_new_episodes=args.max_new_episodes,
        parallel_workers=args.parallel_workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CELL_ORDER",
    "COLLECTOR_FORMAT",
    "FORMAL_TARGETS",
    "FormalV2CollectionError",
    "FormalV2Config",
    "FormalV2PartitionWriter",
    "load_formal_v2_config",
    "run_formal_v2_collection",
]
