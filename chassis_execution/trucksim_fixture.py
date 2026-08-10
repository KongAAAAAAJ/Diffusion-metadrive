"""Generate a deterministic CF-2 exchange fixture (never formal training data)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contracts import (
    CONTROLLER_CONTEXT_FIELDS,
    INITIAL_STATE_FIELDS,
    VEHICLE_CONDITION_FIELDS,
)
from .dataset import canonical_sha256
from .trucksim_exchange import CHECKSUM_FILES, TARGET_UNITS, file_sha256


RAW_EXPORT_FIELDS = (
    "world_x_m",
    "world_y_m",
    "heading_rad",
    "longitudinal_speed_mps",
    "lateral_speed_mps",
    "longitudinal_acceleration_mps2",
    "lateral_acceleration_mps2",
    "yaw_rate_rad_s",
    "roll_rad",
    "roll_rate_rad_s",
    "road_wheel_angle_rad",
    "wheel_fl_rad_s",
    "wheel_fr_rad_s",
    "wheel_rl_rad_s",
    "wheel_rr_rad_s",
    "rollover_index",
)

CONTROLLER_LOG_FIELDS = (
    "signed_longitudinal_control",
    "road_wheel_angle_rad",
    "throttle_normalized",
    "brake_normalized",
)


def _binding(
    field: str,
    *,
    source: str,
    source_unit: str | None = None,
    fields: tuple[str, ...] | None = None,
    reduction: str = "identity",
    native: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": source,
        "fields": list(fields or (field,)),
        "source_unit": source_unit or TARGET_UNITS[field],
        "target_unit": TARGET_UNITS[field],
        "scale": 1.0,
        "offset": 0.0,
        "sign": 1.0,
        "reduction": reduction,
    }
    if native:
        payload["native"] = True
    return payload


def canonical_signal_mapping() -> dict[str, object]:
    """Return the executable example mapping used by tests and Windows handoff."""

    initial = {
        field: _binding(
            field,
            source=("controller_log" if field == "signed_longitudinal_control" else "trucksim_export"),
        )
        for field in INITIAL_STATE_FIELDS
    }
    initial["mean_wheel_speed_rad_s"] = _binding(
        "mean_wheel_speed_rad_s",
        source="trucksim_export",
        fields=("wheel_fl_rad_s", "wheel_fr_rad_s", "wheel_rl_rad_s", "wheel_rr_rad_s"),
        reduction="mean",
    )
    return {
        "format": "trucksim_signal_mapping_v1",
        "schema_version": 1,
        "conversion_formula": "si = raw * scale * sign + offset",
        "raw_export_fields": list(RAW_EXPORT_FIELDS),
        "controller_log_fields": list(CONTROLLER_LOG_FIELDS),
        "world_pose": {
            "world_x_m": _binding("world_x_m", source="trucksim_export"),
            "world_y_m": _binding("world_y_m", source="trucksim_export"),
            "heading_rad": _binding("heading_rad", source="trucksim_export"),
        },
        "initial_state": initial,
        "chassis_state": {
            field: _binding(
                field,
                source="trucksim_export",
                native=(field == "rollover_index"),
            )
            for field in (
                "longitudinal_speed_mps",
                "lateral_speed_mps",
                "longitudinal_acceleration_mps2",
                "lateral_acceleration_mps2",
                "yaw_rate_rad_s",
                "roll_rad",
                "roll_rate_rad_s",
                "rollover_index",
            )
        },
        "applied_control": {
            "road_wheel_angle_rad": _binding(
                "road_wheel_angle_rad", source="trucksim_export"
            ),
            "throttle_normalized": _binding(
                "throttle_normalized", source="controller_log"
            ),
            "brake_normalized": _binding(
                "brake_normalized", source="controller_log"
            ),
        },
        "vehicle_condition": {
            field: _binding(field, source="project_parameter")
            for field in VEHICLE_CONDITION_FIELDS
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_synthetic_trucksim_export_fixture(
    root: Path | str,
    *,
    raw_dt_s: float = 0.05,
    run_id: str = "run_fixture_000",
) -> Path:
    """Write a minimal valid fixture with explicit synthetic provenance."""

    export_root = Path(root)
    export_root.mkdir(parents=True, exist_ok=False)
    steps = round(4.0 / float(raw_dt_s))
    if raw_dt_s <= 0.0 or not np.isclose(steps * raw_dt_s, 4.0):
        raise ValueError("raw_dt_s must divide the four-second horizon exactly")
    mapping = canonical_signal_mapping()
    mapping_hash = canonical_sha256(mapping)
    _write_json(export_root / "signal_mapping.json", mapping)
    ranges = {
        "total_mass_kg": [8_000.0, 18_000.0],
        "payload_mass_kg": [0.0, 8_000.0],
        "cg_height_m": [0.8, 2.5],
        "wheelbase_m": [4.0, 8.0],
        "front_track_m": [1.5, 3.0],
        "rear_track_m": [1.5, 3.0],
        "front_cornering_stiffness_n_per_rad": [20_000.0, 200_000.0],
        "rear_cornering_stiffness_n_per_rad": [20_000.0, 200_000.0],
        "roll_stiffness_nm_per_rad": [100_000.0, 1_000_000.0],
        "roll_damping_nms_per_rad": [10_000.0, 100_000.0],
        "tire_friction_coefficient": [0.2, 1.2],
        "road_friction_coefficient": [0.2, 1.2],
        "drive_actuator_delay_s": [0.001, 1.0],
        "brake_actuator_delay_s": [0.001, 1.0],
    }
    component_hashes = {
        "controller_contract_sha256": "1" * 64,
        "trajectory_optimizer_sha256": "2" * 64,
        "trucksim_project_sha256": "3" * 64,
        "signal_mapping_sha256": mapping_hash,
    }
    _write_json(
        export_root / "export_contract.json",
        {
            "format": "trucksim_execution_export_collection_v1",
            "schema_version": 1,
            "purpose": "fixture",
            **component_hashes,
            "split_salt": "trucksim-fixture-v1",
            "vehicle_parameter_ranges": ranges,
        },
    )

    run_root = export_root / run_id
    run_root.mkdir()
    times = np.arange(steps + 1, dtype=np.float64) * float(raw_dt_s)
    command_times = np.arange(1, 9, dtype=np.float32) * 0.5
    tau_cmd = np.zeros((3, 8, 3), dtype=np.float32)
    tau_cmd[..., 0] = 8.0 * command_times
    initial_state = np.zeros((3, len(INITIAL_STATE_FIELDS)), dtype=np.float32)
    initial_state[:, 0] = 8.0
    initial_state[:, -1] = 16.0
    conditions = np.ones((3, len(VEHICLE_CONDITION_FIELDS)), dtype=np.float32)
    conditions[:, 0] = 12_000.0
    conditions[:, 1] = 2_000.0
    conditions[:, 2] = 1.4
    conditions[:, 3] = 5.7
    conditions[:, 4:6] = 2.1
    conditions[:, 6:8] = 80_000.0
    conditions[:, 8] = 300_000.0
    conditions[:, 9] = 30_000.0
    conditions[:, 10:12] = 0.8
    conditions[:, 12:] = 0.2
    context = np.zeros((3, len(CONTROLLER_CONTEXT_FIELDS)), dtype=np.float32)
    context[:, 5:7] = 15.0
    controller_mode = np.zeros(3, dtype=np.int64)
    roles = np.arange(3, dtype=np.int64)
    raw_export = np.zeros((3, times.size, len(RAW_EXPORT_FIELDS)), dtype=np.float64)
    origins = np.asarray([[0.0, 0.0, 0.0], [-15.0, 0.0, 0.0], [-30.0, 0.0, 0.0]])
    for role in range(3):
        raw_export[role, :, RAW_EXPORT_FIELDS.index("world_x_m")] = origins[role, 0] + 8.0 * times
        raw_export[role, :, RAW_EXPORT_FIELDS.index("world_y_m")] = origins[role, 1]
        raw_export[role, :, RAW_EXPORT_FIELDS.index("heading_rad")] = origins[role, 2]
        raw_export[role, :, RAW_EXPORT_FIELDS.index("longitudinal_speed_mps")] = 8.0
        for wheel in ("wheel_fl_rad_s", "wheel_fr_rad_s", "wheel_rl_rad_s", "wheel_rr_rad_s"):
            raw_export[role, :, RAW_EXPORT_FIELDS.index(wheel)] = 16.0
    raw_controller = np.zeros(
        (3, times.size, len(CONTROLLER_LOG_FIELDS)), dtype=np.float64
    )
    arrays = {
        "tau_cmd.npy": tau_cmd,
        "initial_state.npy": initial_state,
        "vehicle_condition.npy": conditions,
        "controller_context.npy": context,
        "controller_mode.npy": controller_mode,
        "agent_role.npy": roles,
        "raw_time_s.npy": times,
        "raw_export.npy": raw_export,
        "raw_controller_log.npy": raw_controller,
    }
    for name, value in arrays.items():
        np.save(run_root / name, value, allow_pickle=False)
    _write_json(
        run_root / "run.json",
        {
            "format": "trucksim_execution_run_v1",
            "schema_version": 1,
            "run_id": run_id,
            "run_group_id": "fixture-group-000",
            "status": "complete",
            "source": "tau_cmd",
            "generated_at": "2026-08-10T00:00:00Z",
            "maneuver": "constant_speed",
            "coordinate_frame": "trucksim_world_right_handed_x_forward_y_left",
            "units": "mapped_to_SI",
            "solver_dt_s": 0.001,
            "controller_dt_s": 0.01,
            "export_dt_s": float(raw_dt_s),
            "component_hashes": component_hashes,
            "component_identity": {
                "trucksim_sim_sha256": "4" * 64,
                "solver_dll_sha256": "5" * 64,
                "controller_config_sha256": "6" * 64,
                "vehicle_config_sha256": "7" * 64,
            },
            "raw_export_fields": list(RAW_EXPORT_FIELDS),
            "controller_log_fields": list(CONTROLLER_LOG_FIELDS),
            "initial_world_pose": origins.tolist(),
        },
    )
    _write_json(
        run_root / "files.sha256.json",
        {name: file_sha256(run_root / name) for name in sorted(CHECKSUM_FILES)},
    )
    return export_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-dt-s", type=float, default=0.05)
    args = parser.parse_args()
    root = write_synthetic_trucksim_export_fixture(args.output, raw_dt_s=args.raw_dt_s)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
