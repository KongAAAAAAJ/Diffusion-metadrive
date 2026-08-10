from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from chassis_execution import (
    TruckSimExportError,
    convert_verified_run,
    load_trucksim_export,
    verify_trucksim_export_root,
)
from chassis_execution.dataset import canonical_sha256
from chassis_execution.trucksim_exchange import CHECKSUM_FILES, file_sha256
from chassis_execution.trucksim_fixture import (
    CONTROLLER_LOG_FIELDS,
    RAW_EXPORT_FIELDS,
    write_synthetic_trucksim_export_fixture,
)


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_checksums(run_root: Path) -> None:
    _write(
        run_root / "files.sha256.json",
        {name: file_sha256(run_root / name) for name in sorted(CHECKSUM_FILES)},
    )


def _refresh_mapping_identity(root: Path) -> None:
    mapping = _read(root / "signal_mapping.json")
    digest = canonical_sha256(mapping)
    contract = _read(root / "export_contract.json")
    contract["signal_mapping_sha256"] = digest
    _write(root / "export_contract.json", contract)
    for run_root in root.glob("run_*"):
        manifest = _read(run_root / "run.json")
        manifest["component_hashes"]["signal_mapping_sha256"] = digest
        _write(run_root / "run.json", manifest)
        _refresh_checksums(run_root)


@pytest.fixture()
def export_root(tmp_path: Path) -> Path:
    return write_synthetic_trucksim_export_fixture(tmp_path / "export", raw_dt_s=0.05)


def test_cf2_machine_protocol_is_parseable_and_freezes_no_group_axis() -> None:
    payload = _read(ROOT / "schemas" / "trucksim_execution_export_v1.json")
    assert payload["format"] == "trucksim_execution_export_protocol_v1"
    assert payload["training_sample"]["grpo_group_axis"] is False
    assert payload["timing"]["target_state_times_s"] == pytest.approx(
        np.arange(1, 41) * 0.1
    )
    assert "rollover_index" in payload["signal_mapping"]


def test_valid_non_01_export_converts_to_frozen_target(export_root: Path) -> None:
    verified = load_trucksim_export(export_root, "run_fixture_000")
    target = convert_verified_run(verified)
    assert target.tau_cmd.shape == (1, 3, 8, 3)
    assert target.executed_trajectory.shape == (1, 3, 40, 3)
    assert target.chassis_state.shape == (1, 3, 40, 8)
    assert target.applied_control.shape == (1, 3, 40, 3)
    np.testing.assert_allclose(
        target.executed_trajectory[0, :, :, 0].numpy(),
        np.tile(np.arange(1, 41, dtype=np.float32)[None, :] * 0.8, (3, 1)),
        atol=1e-6,
    )
    assert verified.split in {"train", "val", "test"}
    report = verify_trucksim_export_root(export_root)
    assert report["status"] == "passed"
    assert report["purpose"] == "fixture"


def test_heading_unwrap_and_world_to_role_local(export_root: Path) -> None:
    run_root = export_root / "run_fixture_000"
    raw = np.load(run_root / "raw_export.npy", allow_pickle=False)
    time = np.load(run_root / "raw_time_s.npy", allow_pickle=False)
    heading_index = RAW_EXPORT_FIELDS.index("heading_rad")
    x_index = RAW_EXPORT_FIELDS.index("world_x_m")
    y_index = RAW_EXPORT_FIELDS.index("world_y_m")
    raw[:, :, heading_index] = np.arctan2(
        np.sin(3.1 + 0.1 * time), np.cos(3.1 + 0.1 * time)
    )
    raw[:, :, x_index] = np.asarray([0.0, -15.0, -30.0])[:, None] - 8.0 * time
    raw[:, :, y_index] = 0.0
    np.save(run_root / "raw_export.npy", raw, allow_pickle=False)
    manifest = _read(run_root / "run.json")
    manifest["initial_world_pose"] = [[0.0, 0.0, 3.1], [-15.0, 0.0, 3.1], [-30.0, 0.0, 3.1]]
    _write(run_root / "run.json", manifest)
    _refresh_checksums(run_root)
    run = load_trucksim_export(export_root, "run_fixture_000")
    # Moving toward world -x at heading near pi is positive ego-local progress.
    assert np.all(run.executed_trajectory[:, :, 0] > 0.0)
    np.testing.assert_allclose(run.executed_trajectory[:, -1, 2], 0.4, atol=1e-5)


def test_controller_log_uses_left_continuous_zero_order_hold(export_root: Path) -> None:
    run_root = export_root / "run_fixture_000"
    raw = np.load(run_root / "raw_controller_log.npy", allow_pickle=False)
    time = np.load(run_root / "raw_time_s.npy", allow_pickle=False)
    throttle = CONTROLLER_LOG_FIELDS.index("throttle_normalized")
    raw[..., throttle] = np.minimum(time[None, :], 1.0)
    np.save(run_root / "raw_controller_log.npy", raw, allow_pickle=False)
    _refresh_checksums(run_root)
    run = load_trucksim_export(export_root, "run_fixture_000")
    # Target state t=0.1 stores the left-limit control that produced it.
    np.testing.assert_allclose(run.applied_control[:, 0, 1], 0.05, atol=1e-7)
    np.testing.assert_allclose(run.applied_control[:, 1, 1], 0.15, atol=1e-7)


def test_checksum_and_exact_file_set_are_enforced(export_root: Path) -> None:
    run_root = export_root / "run_fixture_000"
    command = np.load(run_root / "tau_cmd.npy", allow_pickle=False)
    command[0, 0, 0] += 1.0
    np.save(run_root / "tau_cmd.npy", command, allow_pickle=False)
    with pytest.raises(TruckSimExportError, match="checksum mismatch"):
        load_trucksim_export(export_root, "run_fixture_000")
    _refresh_checksums(run_root)
    (run_root / "unexpected.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(TruckSimExportError, match="file set mismatch"):
        load_trucksim_export(export_root, "run_fixture_000")


def test_tau_d_and_role_reordering_are_rejected(export_root: Path) -> None:
    run_root = export_root / "run_fixture_000"
    manifest = _read(run_root / "run.json")
    manifest["source"] = "tau_d"
    _write(run_root / "run.json", manifest)
    _refresh_checksums(run_root)
    with pytest.raises(TruckSimExportError, match="tau_cmd"):
        load_trucksim_export(export_root, "run_fixture_000")
    manifest["source"] = "tau_cmd"
    _write(run_root / "run.json", manifest)
    np.save(run_root / "agent_role.npy", np.asarray([1, 0, 2], dtype=np.int64), allow_pickle=False)
    _refresh_checksums(run_root)
    with pytest.raises(TruckSimExportError, match="agent_role"):
        load_trucksim_export(export_root, "run_fixture_000")


def test_missing_roll_and_non_native_rollover_are_rejected(export_root: Path) -> None:
    mapping_path = export_root / "signal_mapping.json"
    mapping = _read(mapping_path)
    del mapping["chassis_state"]["roll_rad"]
    _write(mapping_path, mapping)
    _refresh_mapping_identity(export_root)
    with pytest.raises(TruckSimExportError, match="must cover exactly"):
        load_trucksim_export(export_root, "run_fixture_000")

    mapping = _read(mapping_path)
    from chassis_execution.trucksim_fixture import canonical_signal_mapping

    mapping = canonical_signal_mapping()
    mapping["chassis_state"]["rollover_index"]["native"] = False
    _write(mapping_path, mapping)
    _refresh_mapping_identity(export_root)
    with pytest.raises(TruckSimExportError, match="native TruckSim"):
        load_trucksim_export(export_root, "run_fixture_000")


def test_bad_unit_and_duplicate_group_binding_are_rejected(export_root: Path) -> None:
    mapping_path = export_root / "signal_mapping.json"
    mapping = _read(mapping_path)
    mapping["chassis_state"]["yaw_rate_rad_s"]["target_unit"] = "deg/s"
    _write(mapping_path, mapping)
    _refresh_mapping_identity(export_root)
    with pytest.raises(TruckSimExportError, match="target unit"):
        load_trucksim_export(export_root, "run_fixture_000")
    mapping["chassis_state"]["yaw_rate_rad_s"] = dict(
        mapping["chassis_state"]["lateral_speed_mps"]
    )
    mapping["chassis_state"]["yaw_rate_rad_s"]["target_unit"] = "rad/s"
    _write(mapping_path, mapping)
    _refresh_mapping_identity(export_root)
    with pytest.raises(TruckSimExportError, match="duplicate source binding"):
        load_trucksim_export(export_root, "run_fixture_000")


def test_raw_time_gap_and_nonfinite_export_are_rejected(export_root: Path) -> None:
    run_root = export_root / "run_fixture_000"
    time = np.load(run_root / "raw_time_s.npy", allow_pickle=False)
    export = np.load(run_root / "raw_export.npy", allow_pickle=False)
    controller = np.load(run_root / "raw_controller_log.npy", allow_pickle=False)
    np.save(run_root / "raw_time_s.npy", np.delete(time, 5), allow_pickle=False)
    np.save(run_root / "raw_export.npy", np.delete(export, 5, axis=1), allow_pickle=False)
    np.save(run_root / "raw_controller_log.npy", np.delete(controller, 5, axis=1), allow_pickle=False)
    _refresh_checksums(run_root)
    with pytest.raises(TruckSimExportError, match="gap-free"):
        load_trucksim_export(export_root, "run_fixture_000")

    np.save(run_root / "raw_time_s.npy", time, allow_pickle=False)
    export[0, 4, 0] = np.nan
    np.save(run_root / "raw_export.npy", export, allow_pickle=False)
    np.save(run_root / "raw_controller_log.npy", controller, allow_pickle=False)
    _refresh_checksums(run_root)
    with pytest.raises(TruckSimExportError, match="finite"):
        load_trucksim_export(export_root, "run_fixture_000")


def test_initial_state_and_component_hash_mismatch_are_rejected(export_root: Path) -> None:
    run_root = export_root / "run_fixture_000"
    initial = np.load(run_root / "initial_state.npy", allow_pickle=False)
    initial[0, 0] += 0.5
    np.save(run_root / "initial_state.npy", initial, allow_pickle=False)
    _refresh_checksums(run_root)
    with pytest.raises(TruckSimExportError, match="initial_state does not match"):
        load_trucksim_export(export_root, "run_fixture_000")
    initial[0, 0] -= 0.5
    np.save(run_root / "initial_state.npy", initial, allow_pickle=False)
    manifest = _read(run_root / "run.json")
    manifest["component_hashes"]["trucksim_project_sha256"] = "f" * 64
    _write(run_root / "run.json", manifest)
    _refresh_checksums(run_root)
    with pytest.raises(TruckSimExportError, match="does not match collection"):
        load_trucksim_export(export_root, "run_fixture_000")


def test_vehicle_range_and_normalized_control_are_enforced(export_root: Path) -> None:
    run_root = export_root / "run_fixture_000"
    conditions = np.load(run_root / "vehicle_condition.npy", allow_pickle=False)
    conditions[0, 0] = 30_000.0
    np.save(run_root / "vehicle_condition.npy", conditions, allow_pickle=False)
    _refresh_checksums(run_root)
    with pytest.raises(TruckSimExportError, match="outside frozen range"):
        load_trucksim_export(export_root, "run_fixture_000")
    conditions[0, 0] = 12_000.0
    np.save(run_root / "vehicle_condition.npy", conditions, allow_pickle=False)
    controller = np.load(run_root / "raw_controller_log.npy", allow_pickle=False)
    controller[..., CONTROLLER_LOG_FIELDS.index("brake_normalized")] = 1.2
    np.save(run_root / "raw_controller_log.npy", controller, allow_pickle=False)
    _refresh_checksums(run_root)
    with pytest.raises(TruckSimExportError, match="brake must be normalized"):
        load_trucksim_export(export_root, "run_fixture_000")


def test_same_run_group_has_same_deterministic_split(export_root: Path) -> None:
    first_root = export_root / "run_fixture_000"
    second_root = export_root / "run_fixture_001"
    shutil.copytree(first_root, second_root)
    manifest = _read(second_root / "run.json")
    manifest["run_id"] = "run_fixture_001"
    _write(second_root / "run.json", manifest)
    _refresh_checksums(second_root)
    first = load_trucksim_export(export_root, "run_fixture_000")
    second = load_trucksim_export(export_root, "run_fixture_001")
    assert first.run_group_id == second.run_group_id
    assert first.split == second.split


def test_real_smoke_gate_cannot_be_satisfied_by_fixture_metadata(export_root: Path) -> None:
    contract = _read(export_root / "export_contract.json")
    contract["purpose"] = "real_smoke"
    _write(export_root / "export_contract.json", contract)
    with pytest.raises(TruckSimExportError, match="missing maneuvers"):
        verify_trucksim_export_root(export_root)


def test_complete_excited_real_smoke_gate_logic(export_root: Path) -> None:
    base_root = export_root / "run_fixture_000"
    maneuvers = (
        "constant_speed",
        "accelerate",
        "brake",
        "stop",
        "lane_change_left",
        "lane_change_right",
        "brake_and_steer",
        "formation_gap_recovery",
    )
    for index, maneuver in enumerate(maneuvers):
        run_id = f"run_smoke_{index:03d}"
        run_root = export_root / run_id
        shutil.copytree(base_root, run_root)
        manifest = _read(run_root / "run.json")
        manifest["run_id"] = run_id
        manifest["run_group_id"] = f"smoke-group-{index:03d}"
        manifest["maneuver"] = maneuver
        _write(run_root / "run.json", manifest)
        controller = np.load(run_root / "raw_controller_log.npy", allow_pickle=False)
        if maneuver == "accelerate":
            controller[..., CONTROLLER_LOG_FIELDS.index("throttle_normalized")] = 0.2
        if maneuver in {"brake", "stop", "brake_and_steer"}:
            controller[..., CONTROLLER_LOG_FIELDS.index("brake_normalized")] = 0.2
        np.save(run_root / "raw_controller_log.npy", controller, allow_pickle=False)
        if maneuver in {"lane_change_left", "lane_change_right", "brake_and_steer"}:
            raw = np.load(run_root / "raw_export.npy", allow_pickle=False)
            time = np.load(run_root / "raw_time_s.npy", allow_pickle=False)
            direction = -1.0 if maneuver == "lane_change_right" else 1.0
            wave = direction * np.sin(np.pi * time / 4.0)
            raw[..., RAW_EXPORT_FIELDS.index("road_wheel_angle_rad")] = 0.02 * wave
            raw[..., RAW_EXPORT_FIELDS.index("yaw_rate_rad_s")] = 0.01 * wave
            raw[..., RAW_EXPORT_FIELDS.index("lateral_acceleration_mps2")] = 0.1 * wave
            raw[..., RAW_EXPORT_FIELDS.index("roll_rad")] = 0.01 * wave
            raw[..., RAW_EXPORT_FIELDS.index("rollover_index")] = 0.02 * wave
            np.save(run_root / "raw_export.npy", raw, allow_pickle=False)
        if maneuver == "formation_gap_recovery":
            np.save(run_root / "controller_mode.npy", np.ones(3, dtype=np.int64), allow_pickle=False)
        _refresh_checksums(run_root)
    shutil.rmtree(base_root)
    contract = _read(export_root / "export_contract.json")
    contract["purpose"] = "real_smoke"
    _write(export_root / "export_contract.json", contract)
    report = verify_trucksim_export_root(export_root)
    assert report["status"] == "passed"
    assert report["run_count"] == 8
    assert set(report["maneuvers"]) == set(maneuvers)
    assert all(value > 0.0 for value in report["real_smoke_excitation"].values())
