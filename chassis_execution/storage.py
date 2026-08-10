"""CF-3 atomic mmap storage, Dataset and full verifier."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import (
    CHASSIS_STATE_FIELDS,
    CONTROLLER_CONTEXT_FIELDS,
    CONTROL_FIELDS,
    INITIAL_STATE_FIELDS,
    VEHICLE_CONDITION_FIELDS,
    ChassisExecutionTarget,
)
from .dataset import (
    ChassisExecutionDatasetMetadata,
    assign_run_group_split,
    canonical_sha256,
    validate_atomic_run_group_splits,
)


SPLITS = ("train", "val", "test")
ARRAY_SPECS: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
    "tau_cmd": (np.dtype(np.float32), (3, 8, 3)),
    "initial_state": (np.dtype(np.float32), (3, len(INITIAL_STATE_FIELDS))),
    "vehicle_condition": (np.dtype(np.float32), (3, len(VEHICLE_CONDITION_FIELDS))),
    "controller_context": (np.dtype(np.float32), (3, len(CONTROLLER_CONTEXT_FIELDS))),
    "controller_mode": (np.dtype(np.int64), (3,)),
    "agent_role": (np.dtype(np.int64), (3,)),
    "executed_trajectory": (np.dtype(np.float32), (3, 40, 3)),
    "chassis_state": (np.dtype(np.float32), (3, 40, len(CHASSIS_STATE_FIELDS))),
    "applied_control": (np.dtype(np.float32), (3, 40, len(CONTROL_FIELDS))),
    "state_valid_mask": (np.dtype(np.bool_), (3, 40)),
}
SPLIT_FILES = frozenset({f"{name}.npy" for name in ARRAY_SPECS} | {"manifest.jsonl"})
ROOT_FILES = frozenset({"dataset_contract.json", "checksums.json", *SPLITS})


class ChassisExecutionStorageError(ValueError):
    """Raised when a CF-3 physical dataset violates its frozen contract."""


@dataclass(frozen=True)
class ChassisExecutionSampleIdentity:
    run_id: str
    run_group_id: str
    maneuver: str

    def __post_init__(self) -> None:
        if not self.run_id or not self.run_group_id or not self.maneuver:
            raise ChassisExecutionStorageError("run identity fields must be non-empty")


@dataclass(frozen=True)
class ChassisExecutionStorageProvenance:
    data_origin: str
    diagnostic_only: bool
    eligible_for_formal_training: bool
    cf2_real_windows_smoke_passed: bool
    diagnostic_override_id: str | None
    generator_config_sha256: str

    def __post_init__(self) -> None:
        if self.data_origin not in {"synthetic_virtual", "trucksim"}:
            raise ChassisExecutionStorageError("data_origin must be synthetic_virtual or trucksim")
        _require_sha256(self.generator_config_sha256, "generator_config_sha256")
        if self.data_origin == "synthetic_virtual":
            if (
                not self.diagnostic_only
                or self.eligible_for_formal_training
                or self.cf2_real_windows_smoke_passed
                or not self.diagnostic_override_id
            ):
                raise ChassisExecutionStorageError(
                    "synthetic_virtual data must remain diagnostic, ineligible and bound to an override"
                )
        if self.eligible_for_formal_training and (
            self.diagnostic_only or not self.cf2_real_windows_smoke_passed
        ):
            raise ChassisExecutionStorageError(
                "formal eligibility requires non-diagnostic data and a passed CF-2 real smoke"
            )


def _require_sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ChassisExecutionStorageError(f"{name} must be a lowercase SHA256")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChassisExecutionStorageError(f"cannot read {name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ChassisExecutionStorageError(f"{name} must contain a JSON object")
    return payload


def _target_arrays(target: ChassisExecutionTarget) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, (dtype, suffix) in ARRAY_SPECS.items():
        tensor = getattr(target, name)
        if tensor.device.type != "cpu":
            raise ChassisExecutionStorageError(f"target.{name} must be on CPU for storage")
        value = tensor.detach().numpy()
        if value.dtype != dtype or value.shape[1:] != suffix:
            raise ChassisExecutionStorageError(
                f"target.{name} physical shape/dtype mismatch: {value.shape}, {value.dtype}"
            )
        arrays[name] = value
    return arrays


def write_chassis_execution_dataset(
    root: Path | str,
    *,
    target: ChassisExecutionTarget,
    identities: Sequence[ChassisExecutionSampleIdentity],
    metadata: ChassisExecutionDatasetMetadata,
    provenance: ChassisExecutionStorageProvenance,
) -> dict[str, Any]:
    """Atomically write one immutable split-first mmap dataset."""

    output_root = Path(root).expanduser().resolve()
    if output_root.exists():
        raise ChassisExecutionStorageError(f"dataset root already exists: {output_root}")
    arrays = _target_arrays(target)
    sample_count = int(next(iter(arrays.values())).shape[0])
    if sample_count <= 0 or len(identities) != sample_count:
        raise ChassisExecutionStorageError("identity count must equal positive target batch size")
    run_ids = [identity.run_id for identity in identities]
    if len(set(run_ids)) != len(run_ids):
        raise ChassisExecutionStorageError("run_id values must be unique")
    if any(value.shape[0] != sample_count for value in arrays.values()):
        raise ChassisExecutionStorageError("target arrays disagree on sample count")
    rows = [
        {
            **asdict(identity),
            "split": assign_run_group_split(identity.run_group_id, salt=metadata.split_salt),
        }
        for identity in identities
    ]
    validate_atomic_run_group_splits(rows, salt=metadata.split_salt)
    split_indices = {
        split: np.asarray(
            [index for index, row in enumerate(rows) if row["split"] == split],
            dtype=np.int64,
        )
        for split in SPLITS
    }
    if any(indices.size == 0 for indices in split_indices.values()):
        raise ChassisExecutionStorageError("train, val and test splits must all be non-empty")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        checksums: dict[str, str] = {}
        for split, indices in split_indices.items():
            split_root = staging / split
            split_root.mkdir()
            for name, value in arrays.items():
                path = split_root / f"{name}.npy"
                np.save(path, np.ascontiguousarray(value[indices]), allow_pickle=False)
                checksums[str(path.relative_to(staging))] = _file_sha256(path)
            manifest_path = split_root / "manifest.jsonl"
            with manifest_path.open("w", encoding="utf-8") as stream:
                for row_index, source_index in enumerate(indices.tolist()):
                    stream.write(
                        json.dumps(
                            {**rows[source_index], "row_index": row_index},
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            checksums[str(manifest_path.relative_to(staging))] = _file_sha256(manifest_path)
        _write_json(staging / "checksums.json", checksums)
        base_contract = {
            "format": "chassis_execution_storage_v1",
            "schema_version": 1,
            "sample_count": sample_count,
            "split_counts": {split: int(indices.size) for split, indices in split_indices.items()},
            "array_specs": {
                name: {"dtype": dtype.name, "sample_shape": list(shape)}
                for name, (dtype, shape) in ARRAY_SPECS.items()
            },
            "dataset_metadata": metadata.payload(),
            "dataset_metadata_fingerprint": metadata.fingerprint(),
            "provenance": asdict(provenance),
        }
        dataset_fingerprint = canonical_sha256(
            {"contract": base_contract, "payload_checksums": checksums}
        )
        contract = {**base_contract, "dataset_fingerprint": dataset_fingerprint}
        _write_json(staging / "dataset_contract.json", contract)
        os.replace(staging, output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return contract


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    line_number = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row is not an object")
                rows.append(row)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ChassisExecutionStorageError(f"invalid manifest {path}:{line_number}: {exc}") from exc
    return rows


def _load_contract(root: Path) -> tuple[dict[str, Any], ChassisExecutionDatasetMetadata, ChassisExecutionStorageProvenance]:
    contract = _read_json(root / "dataset_contract.json", "dataset_contract.json")
    if contract.get("format") != "chassis_execution_storage_v1" or contract.get("schema_version") != 1:
        raise ChassisExecutionStorageError("dataset storage format/schema mismatch")
    try:
        metadata = ChassisExecutionDatasetMetadata(**contract["dataset_metadata"])
        provenance = ChassisExecutionStorageProvenance(**contract["provenance"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ChassisExecutionStorageError(f"invalid dataset contract: {exc}") from exc
    if contract.get("dataset_metadata_fingerprint") != metadata.fingerprint():
        raise ChassisExecutionStorageError("dataset metadata fingerprint mismatch")
    return contract, metadata, provenance


def verify_chassis_execution_dataset(root: Path | str) -> dict[str, Any]:
    """Fully scan a CF-3 dataset, including mmap and every numerical sample."""

    dataset_root = Path(root).expanduser().resolve()
    entries = {path.name for path in dataset_root.iterdir()} if dataset_root.is_dir() else set()
    if entries != set(ROOT_FILES):
        raise ChassisExecutionStorageError("dataset root file set mismatch")
    contract, metadata, provenance = _load_contract(dataset_root)
    checksums = _read_json(dataset_root / "checksums.json", "checksums.json")
    expected_checksum_paths = {
        f"{split}/{name}" for split in SPLITS for name in SPLIT_FILES
    }
    if set(checksums) != expected_checksum_paths:
        raise ChassisExecutionStorageError("checksums.json payload set mismatch")
    for relative_name, expected in checksums.items():
        if _file_sha256(dataset_root / relative_name) != _require_sha256(
            expected, f"checksum[{relative_name}]"
        ):
            raise ChassisExecutionStorageError(f"checksum mismatch for {relative_name}")
    expected_fingerprint = canonical_sha256(
        {
            "contract": {
                key: value for key, value in contract.items() if key != "dataset_fingerprint"
            },
            "payload_checksums": checksums,
        }
    )
    if contract.get("dataset_fingerprint") != expected_fingerprint:
        raise ChassisExecutionStorageError("dataset fingerprint mismatch")
    all_rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    stats = {
        "trajectory_command_execution_mae_m": 0.0,
        "maximum_abs_rollover_index": 0.0,
        "maximum_throttle": 0.0,
        "maximum_brake": 0.0,
    }
    total_difference_sum = 0.0
    total_difference_count = 0
    maneuver_counts: Counter[str] = Counter()
    controller_mode_counts: Counter[str] = Counter()
    for split in SPLITS:
        split_root = dataset_root / split
        if not split_root.is_dir() or {path.name for path in split_root.iterdir()} != set(SPLIT_FILES):
            raise ChassisExecutionStorageError(f"{split} file set mismatch")
        rows = _load_manifest(split_root / "manifest.jsonl")
        split_counts[split] = len(rows)
        arrays: dict[str, np.ndarray] = {}
        for name, (dtype, suffix) in ARRAY_SPECS.items():
            value = np.load(split_root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if not isinstance(value, np.memmap):
                raise ChassisExecutionStorageError(f"{split}/{name}.npy is not mmap-readable")
            if value.dtype != dtype or value.shape != (len(rows),) + suffix:
                raise ChassisExecutionStorageError(f"{split}/{name} shape/dtype mismatch")
            if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
                raise ChassisExecutionStorageError(f"{split}/{name} contains non-finite values")
            arrays[name] = value
        for row_index, row in enumerate(rows):
            if row.get("row_index") != row_index or row.get("split") != split:
                raise ChassisExecutionStorageError(f"{split} manifest row alignment mismatch")
            if not isinstance(row.get("maneuver"), str) or not row["maneuver"]:
                raise ChassisExecutionStorageError(f"{split} manifest maneuver is invalid")
            maneuver_counts[row["maneuver"]] += 1
            all_rows.append(row)
        if not bool(arrays["state_valid_mask"].all()):
            raise ChassisExecutionStorageError("all fixed four-second windows must be valid")
        expected_roles = np.broadcast_to(np.arange(3, dtype=np.int64), arrays["agent_role"].shape)
        if not np.array_equal(arrays["agent_role"], expected_roles):
            raise ChassisExecutionStorageError("agent roles must remain [0,1,2]")
        if not bool(np.isin(arrays["controller_mode"], (0, 1)).all()):
            raise ChassisExecutionStorageError("controller_mode contains an invalid enum")
        controller_mode_counts["independent"] += int((arrays["controller_mode"] == 0).sum())
        controller_mode_counts["formation_locked"] += int(
            (arrays["controller_mode"] == 1).sum()
        )
        for condition_index, condition_name in enumerate(VEHICLE_CONDITION_FIELDS):
            try:
                low, high = metadata.vehicle_parameter_ranges[condition_name]
            except KeyError as exc:
                raise ChassisExecutionStorageError(
                    f"vehicle range missing for {condition_name}"
                ) from exc
            values = arrays["vehicle_condition"][..., condition_index]
            if bool(((values < low) | (values > high)).any()):
                raise ChassisExecutionStorageError(
                    f"vehicle condition {condition_name} is outside dataset range"
                )
        speed = arrays["chassis_state"][..., CHASSIS_STATE_FIELDS.index("longitudinal_speed_mps")]
        rollover = arrays["chassis_state"][..., CHASSIS_STATE_FIELDS.index("rollover_index")]
        throttle = arrays["applied_control"][..., CONTROL_FIELDS.index("throttle_normalized")]
        brake = arrays["applied_control"][..., CONTROL_FIELDS.index("brake_normalized")]
        if bool((speed < 0.0).any()) or bool((np.abs(rollover) > 1.0 + 1e-6).any()):
            raise ChassisExecutionStorageError("chassis state violates speed/rollover bounds")
        if bool(((throttle < 0.0) | (throttle > 1.0)).any()) or bool(
            ((brake < 0.0) | (brake > 1.0)).any()
        ):
            raise ChassisExecutionStorageError("applied controls violate normalized bounds")
        # Compare command and execution at common 0.5 s boundaries.
        execution_half_second = arrays["executed_trajectory"][:, :, 4::5, :2]
        command_xy = arrays["tau_cmd"][..., :2]
        total_difference_sum += float(np.abs(execution_half_second - command_xy).sum())
        total_difference_count += int(command_xy.size)
        stats["maximum_abs_rollover_index"] = max(
            stats["maximum_abs_rollover_index"], float(np.max(np.abs(rollover)))
        )
        stats["maximum_throttle"] = max(stats["maximum_throttle"], float(throttle.max()))
        stats["maximum_brake"] = max(stats["maximum_brake"], float(brake.max()))
    validate_atomic_run_group_splits(all_rows, salt=metadata.split_salt)
    run_ids = [str(row.get("run_id", "")) for row in all_rows]
    if any(not run_id for run_id in run_ids) or len(set(run_ids)) != len(run_ids):
        raise ChassisExecutionStorageError("run_id values must be non-empty and unique")
    if split_counts != contract.get("split_counts") or sum(split_counts.values()) != contract.get("sample_count"):
        raise ChassisExecutionStorageError("dataset sample counts do not match contract")
    stats["trajectory_command_execution_mae_m"] = (
        total_difference_sum / max(total_difference_count, 1)
    )
    if provenance.data_origin == "synthetic_virtual" and stats["trajectory_command_execution_mae_m"] <= 1e-5:
        raise ChassisExecutionStorageError("synthetic executed trajectory must not copy tau_cmd")
    return {
        "format": "chassis_execution_dataset_verification_report_v1",
        "status": "passed",
        "root": str(dataset_root),
        "dataset_fingerprint": contract["dataset_fingerprint"],
        "dataset_metadata_fingerprint": metadata.fingerprint(),
        "data_origin": provenance.data_origin,
        "diagnostic_only": provenance.diagnostic_only,
        "eligible_for_formal_training": provenance.eligible_for_formal_training,
        "sample_count": contract["sample_count"],
        "split_counts": split_counts,
        "maneuver_counts": dict(sorted(maneuver_counts.items())),
        "controller_mode_counts": dict(sorted(controller_mode_counts.items())),
        "statistics": stats,
    }


class ChassisExecutionDataset(Dataset[dict[str, torch.Tensor]]):
    """One-split mmap Dataset for CF-4 surrogate training."""

    def __init__(self, root: Path | str, *, split: str) -> None:
        if split not in SPLITS:
            raise ChassisExecutionStorageError("split must be train, val or test")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        contract, _, provenance = _load_contract(self.root)
        self.dataset_fingerprint = str(contract["dataset_fingerprint"])
        self.diagnostic_only = provenance.diagnostic_only
        self.eligible_for_formal_training = provenance.eligible_for_formal_training
        split_root = self.root / split
        if not split_root.is_dir() or {path.name for path in split_root.iterdir()} != set(SPLIT_FILES):
            raise ChassisExecutionStorageError(f"{split} file set mismatch")
        self.rows = _load_manifest(split_root / "manifest.jsonl")
        self.arrays: dict[str, np.ndarray] = {}
        for name, (dtype, suffix) in ARRAY_SPECS.items():
            value = np.load(split_root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if not isinstance(value, np.memmap):
                raise ChassisExecutionStorageError(f"{split}/{name}.npy is not mmap-readable")
            if value.dtype != dtype or value.shape != (len(self.rows),) + suffix:
                raise ChassisExecutionStorageError(f"{split}/{name} shape/dtype mismatch")
            self.arrays[name] = value

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return {
            name: torch.from_numpy(np.array(value[index], copy=True))
            for name, value in self.arrays.items()
        }
