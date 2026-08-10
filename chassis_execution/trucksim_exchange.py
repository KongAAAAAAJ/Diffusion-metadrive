"""Strict Windows TruckSim export loading and CF-1 target conversion.

CF-2 intentionally does not run TruckSim.  It validates an immutable exchange
bundle produced by the Windows-side controller/TruckSim integration and turns
one synchronized three-role run into one :class:`ChassisExecutionTarget`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .contracts import (
    CHASSIS_STATE_FIELDS,
    CONTROLLER_CONTEXT_FIELDS,
    CONTROL_FIELDS,
    EXECUTION_TIMES_S,
    INITIAL_STATE_FIELDS,
    NUM_ROLES,
    VEHICLE_CONDITION_FIELDS,
    ChassisExecutionTarget,
)
from .dataset import (
    ChassisExecutionDatasetMetadata,
    assign_run_group_split,
    canonical_sha256,
)


COLLECTION_CONTRACT_FILE = "export_contract.json"
SIGNAL_MAPPING_FILE = "signal_mapping.json"
RUN_FILES = frozenset(
    {
        "run.json",
        "tau_cmd.npy",
        "initial_state.npy",
        "vehicle_condition.npy",
        "controller_context.npy",
        "controller_mode.npy",
        "agent_role.npy",
        "raw_time_s.npy",
        "raw_export.npy",
        "raw_controller_log.npy",
        "files.sha256.json",
    }
)
CHECKSUM_FILES = RUN_FILES - {"files.sha256.json"}

REQUIRED_MANEUVERS = frozenset(
    {
        "constant_speed",
        "accelerate",
        "brake",
        "stop",
        "lane_change_left",
        "lane_change_right",
        "brake_and_steer",
        "formation_gap_recovery",
    }
)

WORLD_POSE_FIELDS = ("world_x_m", "world_y_m", "heading_rad")

TARGET_UNITS = {
    "world_x_m": "m",
    "world_y_m": "m",
    "heading_rad": "rad",
    "longitudinal_speed_mps": "m/s",
    "lateral_speed_mps": "m/s",
    "longitudinal_acceleration_mps2": "m/s^2",
    "lateral_acceleration_mps2": "m/s^2",
    "yaw_rate_rad_s": "rad/s",
    "roll_rad": "rad",
    "roll_rate_rad_s": "rad/s",
    "road_wheel_angle_rad": "rad",
    "signed_longitudinal_control": "1",
    "mean_wheel_speed_rad_s": "rad/s",
    "rollover_index": "1",
    "throttle_normalized": "1",
    "brake_normalized": "1",
    "total_mass_kg": "kg",
    "payload_mass_kg": "kg",
    "cg_height_m": "m",
    "wheelbase_m": "m",
    "front_track_m": "m",
    "rear_track_m": "m",
    "front_cornering_stiffness_n_per_rad": "N/rad",
    "rear_cornering_stiffness_n_per_rad": "N/rad",
    "roll_stiffness_nm_per_rad": "N*m/rad",
    "roll_damping_nms_per_rad": "N*m*s/rad",
    "tire_friction_coefficient": "1",
    "road_friction_coefficient": "1",
    "drive_actuator_delay_s": "s",
    "brake_actuator_delay_s": "s",
}


class TruckSimExportError(ValueError):
    """Raised when a TruckSim exchange artifact violates the CF-2 contract."""


def _require_sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise TruckSimExportError(f"{name} must be a lowercase SHA256")
    return text


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TruckSimExportError(f"cannot read {name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TruckSimExportError(f"{name} must contain a JSON object")
    return payload


def _require_keys(payload: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    missing = sorted(set(keys) - set(payload))
    if missing:
        raise TruckSimExportError(f"{name} missing required keys: {missing}")


def _load_array(
    path: Path,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int | None, ...],
    name: str,
) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise TruckSimExportError(f"cannot load {name}: {exc}") from exc
    expected_dtype = np.dtype(dtype)
    if not isinstance(value, np.ndarray) or value.dtype != expected_dtype:
        actual = getattr(value, "dtype", type(value).__name__)
        raise TruckSimExportError(f"{name} dtype must be {expected_dtype}, got {actual}")
    if value.ndim != len(shape) or any(
        expected is not None and int(actual) != expected
        for actual, expected in zip(value.shape, shape)
    ):
        raise TruckSimExportError(f"{name} shape {value.shape} does not match {shape}")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise TruckSimExportError(f"{name} must contain only finite values")
    return value


@dataclass(frozen=True)
class TruckSimExportCollectionContract:
    purpose: str
    metadata: ChassisExecutionDatasetMetadata


@dataclass(frozen=True)
class VerifiedTruckSimRun:
    """One fully verified synchronized three-role TruckSim run."""

    root: Path
    run_id: str
    run_group_id: str
    split: str
    maneuver: str
    component_hashes: Mapping[str, str]
    metadata: ChassisExecutionDatasetMetadata
    tau_cmd: np.ndarray
    initial_state: np.ndarray
    vehicle_condition: np.ndarray
    controller_context: np.ndarray
    controller_mode: np.ndarray
    agent_role: np.ndarray
    executed_trajectory: np.ndarray
    chassis_state: np.ndarray
    applied_control: np.ndarray


def load_export_collection_contract(root: Path) -> TruckSimExportCollectionContract:
    payload = _read_json(root / COLLECTION_CONTRACT_FILE, name=COLLECTION_CONTRACT_FILE)
    _require_keys(
        payload,
        (
            "format",
            "schema_version",
            "purpose",
            "controller_contract_sha256",
            "trajectory_optimizer_sha256",
            "trucksim_project_sha256",
            "signal_mapping_sha256",
            "split_salt",
            "vehicle_parameter_ranges",
        ),
        COLLECTION_CONTRACT_FILE,
    )
    if payload["format"] != "trucksim_execution_export_collection_v1":
        raise TruckSimExportError("export collection format mismatch")
    if payload["schema_version"] != 1:
        raise TruckSimExportError("export collection schema_version must be 1")
    purpose = str(payload["purpose"])
    if purpose not in {"fixture", "real_smoke", "dataset_candidate"}:
        raise TruckSimExportError("collection purpose is invalid")
    ranges = payload["vehicle_parameter_ranges"]
    if not isinstance(ranges, dict) or set(ranges) != set(VEHICLE_CONDITION_FIELDS):
        raise TruckSimExportError(
            "vehicle_parameter_ranges must cover every frozen vehicle condition"
        )
    try:
        metadata = ChassisExecutionDatasetMetadata(
            format="chassis_execution_dataset_v1",
            schema_version=1,
            controller_contract_sha256=str(payload["controller_contract_sha256"]),
            trajectory_optimizer_sha256=str(payload["trajectory_optimizer_sha256"]),
            trucksim_project_sha256=str(payload["trucksim_project_sha256"]),
            signal_mapping_sha256=str(payload["signal_mapping_sha256"]),
            split_salt=str(payload["split_salt"]),
            command_dt_s=0.5,
            execution_dt_s=0.1,
            horizon_s=4.0,
            coordinate_frame="per_role_current_ego_local",
            units="SI",
            vehicle_parameter_ranges={
                str(name): (float(bounds[0]), float(bounds[1]))
                for name, bounds in ranges.items()
            },
        )
    except (ValueError, TypeError, IndexError) as exc:
        raise TruckSimExportError(f"invalid export collection metadata: {exc}") from exc
    return TruckSimExportCollectionContract(purpose=purpose, metadata=metadata)


def _validate_field_names(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise TruckSimExportError(f"{name} must be a non-empty list of field names")
    fields = tuple(value)
    if len(fields) != len(set(fields)):
        raise TruckSimExportError(f"{name} contains duplicate field names")
    return fields


def load_signal_mapping(root: Path, *, expected_sha256: str) -> dict[str, Any]:
    payload = _read_json(root / SIGNAL_MAPPING_FILE, name=SIGNAL_MAPPING_FILE)
    _require_keys(
        payload,
        (
            "format",
            "schema_version",
            "conversion_formula",
            "raw_export_fields",
            "controller_log_fields",
            "world_pose",
            "initial_state",
            "chassis_state",
            "applied_control",
            "vehicle_condition",
        ),
        SIGNAL_MAPPING_FILE,
    )
    if payload["format"] != "trucksim_signal_mapping_v1" or payload["schema_version"] != 1:
        raise TruckSimExportError("signal mapping format/schema mismatch")
    if payload["conversion_formula"] != "si = raw * scale * sign + offset":
        raise TruckSimExportError("signal mapping conversion formula mismatch")
    actual_sha256 = canonical_sha256(payload)
    if actual_sha256 != _require_sha256(expected_sha256, "signal_mapping_sha256"):
        raise TruckSimExportError(
            f"signal mapping hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    export_fields = _validate_field_names(payload["raw_export_fields"], name="raw_export_fields")
    controller_fields = _validate_field_names(
        payload["controller_log_fields"], name="controller_log_fields"
    )
    expected_groups = {
        "world_pose": WORLD_POSE_FIELDS,
        "initial_state": INITIAL_STATE_FIELDS,
        "chassis_state": CHASSIS_STATE_FIELDS,
        "applied_control": CONTROL_FIELDS,
        "vehicle_condition": VEHICLE_CONDITION_FIELDS,
    }
    for group_name, fields in expected_groups.items():
        group = payload[group_name]
        if not isinstance(group, dict) or set(group) != set(fields):
            raise TruckSimExportError(
                f"signal mapping group {group_name} must cover exactly {tuple(fields)}"
            )
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for canonical_name in fields:
            binding = group[canonical_name]
            if not isinstance(binding, dict):
                raise TruckSimExportError(f"mapping {group_name}.{canonical_name} must be an object")
            _require_keys(
                binding,
                ("source", "fields", "source_unit", "target_unit", "scale", "offset", "sign"),
                f"mapping {group_name}.{canonical_name}",
            )
            source = str(binding["source"])
            if source not in {"trucksim_export", "controller_log", "project_parameter"}:
                raise TruckSimExportError(f"mapping {group_name}.{canonical_name} has invalid source")
            source_fields = _validate_field_names(
                binding["fields"], name=f"mapping {group_name}.{canonical_name}.fields"
            )
            reduction = str(binding.get("reduction", "identity"))
            if reduction not in {"identity", "mean"}:
                raise TruckSimExportError(f"mapping {group_name}.{canonical_name} has invalid reduction")
            if reduction == "identity" and len(source_fields) != 1:
                raise TruckSimExportError(
                    f"mapping {group_name}.{canonical_name} identity requires one field"
                )
            if source == "trucksim_export" and not set(source_fields).issubset(export_fields):
                raise TruckSimExportError(f"mapping {group_name}.{canonical_name} references unknown export")
            if source == "controller_log" and not set(source_fields).issubset(controller_fields):
                raise TruckSimExportError(f"mapping {group_name}.{canonical_name} references unknown controller field")
            if source == "project_parameter" and tuple(source_fields) != (canonical_name,):
                raise TruckSimExportError(
                    f"project parameter mapping {canonical_name} must use its canonical field name"
                )
            if str(binding["target_unit"]) != TARGET_UNITS[canonical_name]:
                raise TruckSimExportError(f"mapping {group_name}.{canonical_name} target unit mismatch")
            if not str(binding["source_unit"]):
                raise TruckSimExportError(f"mapping {group_name}.{canonical_name} source unit is required")
            try:
                scale = float(binding["scale"])
                offset = float(binding["offset"])
                sign = float(binding["sign"])
            except (TypeError, ValueError) as exc:
                raise TruckSimExportError(f"mapping {group_name}.{canonical_name} conversion invalid") from exc
            if not math.isfinite(scale) or scale == 0.0 or not math.isfinite(offset) or sign not in {-1.0, 1.0}:
                raise TruckSimExportError(f"mapping {group_name}.{canonical_name} conversion invalid")
            if source == "project_parameter" and (
                str(binding["source_unit"]) != TARGET_UNITS[canonical_name]
                or scale != 1.0
                or offset != 0.0
                or sign != 1.0
            ):
                raise TruckSimExportError(
                    f"project parameter {canonical_name} must already be stored in canonical SI units"
                )
            identity = (source, source_fields)
            if identity in seen:
                raise TruckSimExportError(f"mapping group {group_name} contains a duplicate source binding")
            seen.add(identity)
    rollover = payload["chassis_state"]["rollover_index"]
    if rollover["source"] != "trucksim_export" or rollover.get("native") is not True:
        raise TruckSimExportError(
            "rollover_index must bind a native TruckSim export; derived fallback is forbidden"
        )
    return payload


def _mapped_series(
    mapping: Mapping[str, Any],
    *,
    raw_export: np.ndarray,
    raw_controller: np.ndarray,
    export_fields: tuple[str, ...],
    controller_fields: tuple[str, ...],
) -> np.ndarray:
    source = mapping["source"]
    names = tuple(mapping["fields"])
    if source == "trucksim_export":
        indices = [export_fields.index(name) for name in names]
        values = raw_export[..., indices]
    elif source == "controller_log":
        indices = [controller_fields.index(name) for name in names]
        values = raw_controller[..., indices]
    else:
        raise TruckSimExportError("project parameters are static and cannot map a time series")
    if mapping.get("reduction", "identity") == "mean":
        values = values.mean(axis=-1)
    else:
        values = values[..., 0]
    return (
        values * float(mapping["scale"]) * float(mapping["sign"])
        + float(mapping["offset"])
    )


def _interpolate_state(raw_time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    output = np.empty((NUM_ROLES, target_time.size), dtype=np.float64)
    for role in range(NUM_ROLES):
        output[role] = np.interp(target_time, raw_time, values[role])
    return output


def _interpolate_heading(raw_time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    output = np.empty((NUM_ROLES, target_time.size), dtype=np.float64)
    for role in range(NUM_ROLES):
        unwrapped = np.unwrap(values[role])
        output[role] = np.interp(target_time, raw_time, unwrapped)
    return output


def _sample_control_zoh(raw_time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    # Targets are state boundaries.  Store the control active immediately
    # before each boundary, i.e. the command that produced that target state.
    indices = np.searchsorted(raw_time, target_time, side="left") - 1
    if (indices < 0).any() or (indices >= raw_time.size).any():
        raise TruckSimExportError("control target time would require extrapolation")
    return values[:, indices]


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def _world_to_ego_local(world_pose: np.ndarray, initial_pose: np.ndarray) -> np.ndarray:
    dx = world_pose[..., 0] - initial_pose[:, None, 0]
    dy = world_pose[..., 1] - initial_pose[:, None, 1]
    heading = initial_pose[:, None, 2]
    result = np.empty_like(world_pose, dtype=np.float64)
    result[..., 0] = np.cos(heading) * dx + np.sin(heading) * dy
    result[..., 1] = -np.sin(heading) * dx + np.cos(heading) * dy
    result[..., 2] = _wrap_angle(world_pose[..., 2] - heading)
    return result


def _verify_checksums(run_root: Path) -> None:
    payload = _read_json(run_root / "files.sha256.json", name="files.sha256.json")
    if set(payload) != set(CHECKSUM_FILES):
        raise TruckSimExportError("files.sha256.json must cover the exact run payload")
    for relative_name, expected in sorted(payload.items()):
        actual = file_sha256(run_root / relative_name)
        if actual != _require_sha256(expected, f"checksum[{relative_name}]"):
            raise TruckSimExportError(f"checksum mismatch for {relative_name}")


def _verify_vehicle_ranges(
    vehicle_condition: np.ndarray, metadata: ChassisExecutionDatasetMetadata
) -> None:
    for index, name in enumerate(VEHICLE_CONDITION_FIELDS):
        low, high = metadata.vehicle_parameter_ranges[name]
        values = vehicle_condition[:, index]
        if not bool(((values >= low) & (values <= high)).all()):
            raise TruckSimExportError(f"vehicle condition {name} is outside frozen range")


def load_trucksim_export(root: Path | str, run_id: str) -> VerifiedTruckSimRun:
    """Load, validate, resample and canonicalize one exported TruckSim run."""

    export_root = Path(root).expanduser().resolve()
    collection = load_export_collection_contract(export_root)
    mapping = load_signal_mapping(
        export_root, expected_sha256=collection.metadata.signal_mapping_sha256
    )
    run_name = str(run_id)
    if not run_name or Path(run_name).name != run_name:
        raise TruckSimExportError("run_id must be one path-safe directory name")
    run_root = export_root / run_name
    if run_root.is_symlink() or not run_root.is_dir():
        raise TruckSimExportError(f"run directory does not exist: {run_root}")
    entries = tuple(run_root.iterdir())
    actual_files = {path.name for path in entries}
    if actual_files != set(RUN_FILES) or any(
        path.is_symlink() or not path.is_file() for path in entries
    ):
        raise TruckSimExportError(
            f"run file set mismatch: missing={sorted(RUN_FILES - actual_files)}, "
            f"extra={sorted(actual_files - RUN_FILES)}"
        )
    _verify_checksums(run_root)
    manifest = _read_json(run_root / "run.json", name="run.json")
    _require_keys(
        manifest,
        (
            "format",
            "schema_version",
            "run_id",
            "run_group_id",
            "status",
            "source",
            "generated_at",
            "maneuver",
            "coordinate_frame",
            "units",
            "solver_dt_s",
            "controller_dt_s",
            "export_dt_s",
            "component_hashes",
            "component_identity",
            "raw_export_fields",
            "controller_log_fields",
            "initial_world_pose",
        ),
        "run.json",
    )
    if manifest["format"] != "trucksim_execution_run_v1" or manifest["schema_version"] != 1:
        raise TruckSimExportError("run format/schema mismatch")
    if manifest["run_id"] != run_name:
        raise TruckSimExportError("run.json run_id must equal its directory name")
    if manifest["status"] != "complete":
        raise TruckSimExportError("only complete TruckSim runs are accepted")
    if manifest["source"] != "tau_cmd":
        raise TruckSimExportError("TruckSim controller input source must be tau_cmd")
    if manifest["coordinate_frame"] != "trucksim_world_right_handed_x_forward_y_left":
        raise TruckSimExportError("TruckSim world coordinate frame mismatch")
    if manifest["units"] != "mapped_to_SI":
        raise TruckSimExportError("run units declaration must be mapped_to_SI")
    run_group_id = str(manifest["run_group_id"])
    if not run_group_id:
        raise TruckSimExportError("run_group_id is required")
    maneuver = str(manifest["maneuver"])
    if maneuver not in REQUIRED_MANEUVERS:
        raise TruckSimExportError(f"unknown maneuver: {maneuver}")
    component_hashes = manifest["component_hashes"]
    required_hashes = (
        "controller_contract_sha256",
        "trajectory_optimizer_sha256",
        "trucksim_project_sha256",
        "signal_mapping_sha256",
    )
    if not isinstance(component_hashes, dict) or set(component_hashes) != set(required_hashes):
        raise TruckSimExportError("run component_hashes must cover the four frozen identities")
    for name in required_hashes:
        value = _require_sha256(component_hashes[name], f"component_hashes.{name}")
        if value != getattr(collection.metadata, name):
            raise TruckSimExportError(f"run {name} does not match collection contract")
    identities = manifest["component_identity"]
    required_identities = {
        "trucksim_sim_sha256",
        "solver_dll_sha256",
        "controller_config_sha256",
        "vehicle_config_sha256",
    }
    if not isinstance(identities, dict) or set(identities) != required_identities:
        raise TruckSimExportError("component_identity must cover sim, solver, controller and vehicle")
    for name, value in identities.items():
        _require_sha256(value, f"component_identity.{name}")
    export_fields = _validate_field_names(manifest["raw_export_fields"], name="run.raw_export_fields")
    controller_fields = _validate_field_names(
        manifest["controller_log_fields"], name="run.controller_log_fields"
    )
    if export_fields != tuple(mapping["raw_export_fields"]):
        raise TruckSimExportError("run raw_export_fields differ from signal mapping")
    if controller_fields != tuple(mapping["controller_log_fields"]):
        raise TruckSimExportError("run controller_log_fields differ from signal mapping")
    for timing_name in ("solver_dt_s", "controller_dt_s", "export_dt_s"):
        try:
            timing = float(manifest[timing_name])
        except (TypeError, ValueError) as exc:
            raise TruckSimExportError(f"{timing_name} must be finite and positive") from exc
        if not math.isfinite(timing) or timing <= 0.0:
            raise TruckSimExportError(f"{timing_name} must be finite and positive")

    tau_cmd = _load_array(run_root / "tau_cmd.npy", dtype=np.float32, shape=(3, 8, 3), name="tau_cmd")
    initial_state = _load_array(
        run_root / "initial_state.npy", dtype=np.float32, shape=(3, len(INITIAL_STATE_FIELDS)), name="initial_state"
    )
    vehicle_condition = _load_array(
        run_root / "vehicle_condition.npy",
        dtype=np.float32,
        shape=(3, len(VEHICLE_CONDITION_FIELDS)),
        name="vehicle_condition",
    )
    controller_context = _load_array(
        run_root / "controller_context.npy",
        dtype=np.float32,
        shape=(3, len(CONTROLLER_CONTEXT_FIELDS)),
        name="controller_context",
    )
    controller_mode = _load_array(
        run_root / "controller_mode.npy", dtype=np.int64, shape=(3,), name="controller_mode"
    )
    agent_role = _load_array(run_root / "agent_role.npy", dtype=np.int64, shape=(3,), name="agent_role")
    if not np.array_equal(agent_role, np.arange(3, dtype=np.int64)):
        raise TruckSimExportError("agent_role must be [0,1,2]")
    if not bool(np.isin(controller_mode, (0, 1)).all()):
        raise TruckSimExportError("controller_mode must use 0=independent or 1=formation_locked")
    raw_time = _load_array(run_root / "raw_time_s.npy", dtype=np.float64, shape=(None,), name="raw_time_s")
    if raw_time.size < 2 or abs(float(raw_time[0])) > 1e-9 or float(raw_time[-1]) < 4.0 - 1e-9:
        raise TruckSimExportError("raw time must cover state boundaries from t=0 through t=4s")
    differences = np.diff(raw_time)
    export_dt = float(manifest["export_dt_s"])
    tolerance = max(1e-9, export_dt * 1e-5)
    if not bool((differences > 0.0).all()) or not bool(np.allclose(differences, export_dt, rtol=0.0, atol=tolerance)):
        raise TruckSimExportError("raw time must be strictly increasing, uniform and gap-free")
    raw_export = _load_array(
        run_root / "raw_export.npy",
        dtype=np.float64,
        shape=(3, raw_time.size, len(export_fields)),
        name="raw_export",
    )
    raw_controller = _load_array(
        run_root / "raw_controller_log.npy",
        dtype=np.float64,
        shape=(3, raw_time.size, len(controller_fields)),
        name="raw_controller_log",
    )
    _verify_vehicle_ranges(vehicle_condition, collection.metadata)
    if bool((initial_state[:, INITIAL_STATE_FIELDS.index("longitudinal_speed_mps")] < 0.0).any()):
        raise TruckSimExportError("initial longitudinal speed must be non-negative")
    positive_condition_indices = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    if bool((vehicle_condition[:, positive_condition_indices] <= 0.0).any()):
        raise TruckSimExportError("physical vehicle conditions and friction values must be positive")
    if bool((vehicle_condition[:, 1] < 0.0).any()):
        raise TruckSimExportError("payload mass must be non-negative")
    if bool((vehicle_condition[:, 12:] < 0.0).any()):
        raise TruckSimExportError("actuator delays must be non-negative")

    target_time = np.asarray(EXECUTION_TIMES_S, dtype=np.float64)
    mapped: dict[str, dict[str, np.ndarray]] = {}
    for group_name in ("world_pose", "initial_state", "chassis_state", "applied_control"):
        mapped[group_name] = {
            field: _mapped_series(
                mapping[group_name][field],
                raw_export=raw_export,
                raw_controller=raw_controller,
                export_fields=export_fields,
                controller_fields=controller_fields,
            )
            for field in mapping[group_name]
        }

    measured_initial = np.stack(
        [mapped["initial_state"][field][:, 0] for field in INITIAL_STATE_FIELDS], axis=-1
    ).astype(np.float32)
    if not np.allclose(initial_state, measured_initial, rtol=1e-5, atol=1e-4):
        maximum = float(np.max(np.abs(initial_state - measured_initial)))
        raise TruckSimExportError(
            f"initial_state does not match raw t=0 mapping (max_abs_error={maximum:.6g})"
        )
    initial_world_pose = np.asarray(manifest["initial_world_pose"], dtype=np.float64)
    if initial_world_pose.shape != (3, 3) or not np.isfinite(initial_world_pose).all():
        raise TruckSimExportError("initial_world_pose must be finite [3,3]")
    measured_world_pose = np.stack(
        [mapped["world_pose"][field][:, 0] for field in WORLD_POSE_FIELDS], axis=-1
    )
    pose_error = measured_world_pose - initial_world_pose
    pose_error[:, 2] = _wrap_angle(pose_error[:, 2])
    if not np.allclose(pose_error, 0.0, rtol=0.0, atol=1e-6):
        raise TruckSimExportError("initial_world_pose does not match raw t=0 mapping")

    world_pose = np.empty((3, target_time.size, 3), dtype=np.float64)
    world_pose[..., 0] = _interpolate_state(
        raw_time, mapped["world_pose"]["world_x_m"], target_time
    )
    world_pose[..., 1] = _interpolate_state(
        raw_time, mapped["world_pose"]["world_y_m"], target_time
    )
    world_pose[..., 2] = _interpolate_heading(
        raw_time, mapped["world_pose"]["heading_rad"], target_time
    )
    executed = _world_to_ego_local(world_pose, initial_world_pose).astype(np.float32)
    chassis = np.stack(
        [
            _interpolate_state(raw_time, mapped["chassis_state"][field], target_time)
            for field in CHASSIS_STATE_FIELDS
        ],
        axis=-1,
    ).astype(np.float32)
    controls = np.stack(
        [
            _sample_control_zoh(raw_time, mapped["applied_control"][field], target_time)
            for field in CONTROL_FIELDS
        ],
        axis=-1,
    ).astype(np.float32)
    speed_index = CHASSIS_STATE_FIELDS.index("longitudinal_speed_mps")
    rollover_index = CHASSIS_STATE_FIELDS.index("rollover_index")
    throttle_index = CONTROL_FIELDS.index("throttle_normalized")
    brake_index = CONTROL_FIELDS.index("brake_normalized")
    if bool((chassis[..., speed_index] < 0.0).any()):
        raise TruckSimExportError("longitudinal speed must be non-negative")
    if bool((np.abs(chassis[..., rollover_index]) > 1.0 + 1e-6).any()):
        raise TruckSimExportError("native rollover_index must be normalized to [-1,1]")
    for index, name in ((throttle_index, "throttle"), (brake_index, "brake")):
        if bool(((controls[..., index] < 0.0) | (controls[..., index] > 1.0)).any()):
            raise TruckSimExportError(f"{name} must be normalized to [0,1]")

    verified = VerifiedTruckSimRun(
        root=run_root,
        run_id=run_name,
        run_group_id=run_group_id,
        split=assign_run_group_split(run_group_id, salt=collection.metadata.split_salt),
        maneuver=maneuver,
        component_hashes={name: str(component_hashes[name]) for name in required_hashes},
        metadata=collection.metadata,
        tau_cmd=tau_cmd,
        initial_state=initial_state,
        vehicle_condition=vehicle_condition,
        controller_context=controller_context,
        controller_mode=controller_mode,
        agent_role=agent_role,
        executed_trajectory=executed,
        chassis_state=chassis,
        applied_control=controls,
    )
    # Exercise the frozen CF-1 constructor as the final boundary check.
    convert_verified_run(verified)
    return verified


def convert_verified_run(run: VerifiedTruckSimRun) -> ChassisExecutionTarget:
    """Convert one verified run into a batch-of-one CF-1 training target."""

    def floating(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0)

    return ChassisExecutionTarget(
        tau_cmd=floating(run.tau_cmd),
        initial_state=floating(run.initial_state),
        vehicle_condition=floating(run.vehicle_condition),
        controller_context=floating(run.controller_context),
        controller_mode=torch.from_numpy(np.ascontiguousarray(run.controller_mode)).unsqueeze(0),
        agent_role=torch.from_numpy(np.ascontiguousarray(run.agent_role)).unsqueeze(0),
        executed_trajectory=floating(run.executed_trajectory),
        chassis_state=floating(run.chassis_state),
        applied_control=floating(run.applied_control),
        state_valid_mask=torch.ones((1, 3, 40), dtype=torch.bool),
    )


def verify_trucksim_export_root(root: Path | str) -> dict[str, Any]:
    """Verify every run and enforce the real-smoke excitation gate when requested."""

    export_root = Path(root).expanduser().resolve()
    collection = load_export_collection_contract(export_root)
    run_ids = sorted(path.name for path in export_root.iterdir() if path.is_dir() and path.name.startswith("run_"))
    if not run_ids:
        raise TruckSimExportError("export root contains no run_* directories")
    runs = [load_trucksim_export(export_root, run_id) for run_id in run_ids]
    maneuvers = {run.maneuver for run in runs}
    modes = np.concatenate([run.controller_mode for run in runs])
    excitation: dict[str, float] = {}
    if collection.purpose == "real_smoke":
        missing = sorted(REQUIRED_MANEUVERS - maneuvers)
        if missing:
            raise TruckSimExportError(f"real smoke is missing maneuvers: {missing}")
        if not bool((modes == 0).any()) or not bool((modes == 1).any()):
            raise TruckSimExportError("real smoke must cover independent and formation_locked modes")
        by_maneuver = {
            name: [run for run in runs if run.maneuver == name]
            for name in REQUIRED_MANEUVERS
        }
        throttle_index = CONTROL_FIELDS.index("throttle_normalized")
        brake_index = CONTROL_FIELDS.index("brake_normalized")
        if max(
            float(run.applied_control[..., throttle_index].max())
            for run in by_maneuver["accelerate"]
        ) <= 0.05:
            raise TruckSimExportError("accelerate smoke has no measurable throttle response")
        if max(
            float(run.applied_control[..., brake_index].max())
            for name in ("brake", "stop")
            for run in by_maneuver[name]
        ) <= 0.05:
            raise TruckSimExportError("brake/stop smoke has no measurable brake response")
        lateral_runs = [
            run
            for name in ("lane_change_left", "lane_change_right", "brake_and_steer")
            for run in by_maneuver[name]
        ]
        fields_and_thresholds = {
            "road_wheel_angle_rad": 1e-4,
            "yaw_rate_rad_s": 1e-4,
            "lateral_acceleration_mps2": 1e-4,
            "roll_rad": 1e-5,
            "rollover_index": 1e-5,
        }
        for field, threshold in fields_and_thresholds.items():
            if field == "road_wheel_angle_rad":
                arrays = [run.applied_control[..., CONTROL_FIELDS.index(field)] for run in lateral_runs]
            else:
                arrays = [run.chassis_state[..., CHASSIS_STATE_FIELDS.index(field)] for run in lateral_runs]
            dynamic_ranges = [float(np.ptp(array)) for array in arrays]
            excitation[field] = min(dynamic_ranges)
            if any(dynamic_range <= threshold for dynamic_range in dynamic_ranges):
                raise TruckSimExportError(
                    f"a real-smoke lateral run has constant or mis-mapped {field}"
                )
        if not all(
            bool((run.controller_mode == 1).all())
            for run in by_maneuver["formation_gap_recovery"]
        ):
            raise TruckSimExportError(
                "formation_gap_recovery must use formation_locked for all roles"
            )
    return {
        "format": "trucksim_export_verification_report_v1",
        "purpose": collection.purpose,
        "root": str(export_root),
        "run_count": len(runs),
        "run_ids": run_ids,
        "maneuvers": sorted(maneuvers),
        "splits": {
            split: sum(run.split == split for run in runs) for split in ("train", "val", "test")
        },
        "signal_mapping_sha256": collection.metadata.signal_mapping_sha256,
        "dataset_metadata_fingerprint": collection.metadata.fingerprint(),
        "real_smoke_excitation": excitation,
        "target_shapes": {
            "tau_cmd": [1, 3, 8, 3],
            "executed_trajectory": [1, 3, 40, 3],
            "chassis_state": [1, 3, 40, 8],
            "applied_control": [1, 3, 40, 3],
        },
        "status": "passed",
    }
