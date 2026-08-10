"""Dataset identity and split rules for synchronized TruckSim runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


class ChassisExecutionDatasetError(ValueError):
    """Raised when a surrogate dataset cannot satisfy its frozen identity."""


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ChassisExecutionDatasetError(f"{name} must be a lowercase SHA256")
    return text


def assign_run_group_split(run_group_id: str, *, salt: str) -> str:
    """Assign a complete TruckSim run atomically using deterministic 80/10/10 hashing."""

    group = str(run_group_id)
    split_salt = str(salt)
    if not group or not split_salt:
        raise ChassisExecutionDatasetError("run_group_id and split salt are required")
    value = int.from_bytes(
        hashlib.sha256(f"{split_salt}:{group}".encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    ) % 10_000
    if value < 8_000:
        return "train"
    if value < 9_000:
        return "val"
    return "test"


def validate_atomic_run_group_splits(
    rows: Sequence[Mapping[str, object]], *, salt: str
) -> None:
    observed: dict[str, str] = {}
    for row in rows:
        group = str(row.get("run_group_id", ""))
        split = str(row.get("split", ""))
        if split not in {"train", "val", "test"}:
            raise ChassisExecutionDatasetError("split must be train, val, or test")
        expected = assign_run_group_split(group, salt=salt)
        if split != expected:
            raise ChassisExecutionDatasetError(
                f"run group {group} has split {split}, expected {expected}"
            )
        prior = observed.setdefault(group, split)
        if prior != split:
            raise ChassisExecutionDatasetError(
                f"run group {group} leaks across {prior} and {split}"
            )


@dataclass(frozen=True)
class ChassisExecutionDatasetMetadata:
    format: str
    schema_version: int
    controller_contract_sha256: str
    trajectory_optimizer_sha256: str
    trucksim_project_sha256: str
    signal_mapping_sha256: str
    split_salt: str
    command_dt_s: float
    execution_dt_s: float
    horizon_s: float
    coordinate_frame: str
    units: str
    vehicle_parameter_ranges: Mapping[str, tuple[float, float]]

    def __post_init__(self) -> None:
        if self.format != "chassis_execution_dataset_v1" or self.schema_version != 1:
            raise ChassisExecutionDatasetError("dataset format/schema mismatch")
        for name in (
            "controller_contract_sha256",
            "trajectory_optimizer_sha256",
            "trucksim_project_sha256",
            "signal_mapping_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        if not self.split_salt:
            raise ChassisExecutionDatasetError("split_salt is required")
        if (
            not math.isclose(float(self.command_dt_s), 0.5)
            or not math.isclose(float(self.execution_dt_s), 0.1)
            or not math.isclose(float(self.horizon_s), 4.0)
        ):
            raise ChassisExecutionDatasetError(
                "dataset timing must be command=0.5s, execution=0.1s, horizon=4s"
            )
        if self.coordinate_frame != "per_role_current_ego_local":
            raise ChassisExecutionDatasetError("coordinate frame is not frozen v1")
        if self.units != "SI":
            raise ChassisExecutionDatasetError("all surrogate signals must use SI units")
        if not self.vehicle_parameter_ranges:
            raise ChassisExecutionDatasetError("vehicle parameter ranges are required")
        normalized: dict[str, tuple[float, float]] = {}
        for name, bounds in sorted(self.vehicle_parameter_ranges.items()):
            if len(bounds) != 2:
                raise ChassisExecutionDatasetError(f"{name} range must contain min/max")
            low, high = float(bounds[0]), float(bounds[1])
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ChassisExecutionDatasetError(f"{name} range is invalid")
            normalized[str(name)] = (low, high)
        object.__setattr__(self, "vehicle_parameter_ranges", normalized)

    def payload(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        return canonical_sha256(self.payload())
