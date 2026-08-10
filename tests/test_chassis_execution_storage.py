from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from chassis_execution import (
    ChassisExecutionDataset,
    ChassisExecutionStorageError,
    ChassisExecutionStorageProvenance,
    verify_chassis_execution_dataset,
    write_chassis_execution_dataset,
)
from chassis_execution.synthetic_data import (
    SyntheticChassisConfig,
    generate_synthetic_chassis_batch,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_fixture(root: Path, *, count: int = 256) -> tuple[Path, dict[str, object]]:
    target, identities, metadata, provenance = generate_synthetic_chassis_batch(
        SyntheticChassisConfig(sample_count=count)
    )
    contract = write_chassis_execution_dataset(
        root,
        target=target,
        identities=identities,
        metadata=metadata,
        provenance=provenance,
    )
    return root, contract


def test_storage_machine_protocol_is_parseable_and_freezes_shapes() -> None:
    payload = json.loads(
        (ROOT / "schemas" / "chassis_execution_storage_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["format"] == "chassis_execution_storage_protocol_v1"
    assert payload["arrays"]["tau_cmd"]["sample_shape"] == [3, 8, 3]
    assert payload["arrays"]["executed_trajectory"]["sample_shape"] == [3, 40, 3]
    assert payload["diagnostic_override"]["metadata_upgrade_to_formal"] == "forbidden"


def test_virtual_batch_is_deterministic_nontrivial_and_protocol_shaped() -> None:
    config = SyntheticChassisConfig(sample_count=256, seed=17)
    first = generate_synthetic_chassis_batch(config)
    second = generate_synthetic_chassis_batch(config)
    target, identities, _, provenance = first
    assert target.tau_cmd.shape == (256, 3, 8, 3)
    assert target.executed_trajectory.shape == (256, 3, 40, 3)
    assert target.chassis_state.shape == (256, 3, 40, 8)
    assert target.applied_control.shape == (256, 3, 40, 3)
    assert torch.equal(target.tau_cmd, second[0].tau_cmd)
    assert torch.equal(target.executed_trajectory, second[0].executed_trajectory)
    assert identities == second[1]
    assert provenance.data_origin == "synthetic_virtual"
    assert provenance.diagnostic_only is True
    assert provenance.eligible_for_formal_training is False
    assert not torch.allclose(
        target.executed_trajectory[:, :, 4::5, :2], target.tau_cmd[..., :2]
    )


def test_synthetic_provenance_cannot_be_upgraded_to_formal() -> None:
    with pytest.raises(ChassisExecutionStorageError, match="synthetic_virtual"):
        ChassisExecutionStorageProvenance(
            data_origin="synthetic_virtual",
            diagnostic_only=False,
            eligible_for_formal_training=True,
            cf2_real_windows_smoke_passed=False,
            diagnostic_override_id="cf2_real_smoke_diagnostic_override_v1",
            generator_config_sha256="0" * 64,
        )


def test_atomic_writer_verifier_and_mmap_dataset(tmp_path: Path) -> None:
    root, contract = _write_fixture(tmp_path / "dataset")
    report = verify_chassis_execution_dataset(root)
    assert report["status"] == "passed"
    assert report["sample_count"] == 256
    assert sum(report["split_counts"].values()) == 256
    assert all(value > 0 for value in report["split_counts"].values())
    assert report["dataset_fingerprint"] == contract["dataset_fingerprint"]
    assert report["data_origin"] == "synthetic_virtual"
    assert report["diagnostic_only"] is True
    assert report["eligible_for_formal_training"] is False
    assert report["statistics"]["trajectory_command_execution_mae_m"] > 0.0

    train = ChassisExecutionDataset(root, split="train")
    assert isinstance(train.arrays["tau_cmd"], np.memmap)
    sample = train[0]
    assert sample["tau_cmd"].shape == (3, 8, 3)
    assert sample["tau_cmd"].dtype == torch.float32
    assert sample["state_valid_mask"].dtype == torch.bool
    batch = next(iter(DataLoader(train, batch_size=8, shuffle=False, num_workers=0)))
    assert batch["tau_cmd"].shape == (8, 3, 8, 3)
    assert batch["executed_trajectory"].shape == (8, 3, 40, 3)
    assert batch["chassis_state"].shape == (8, 3, 40, 8)


def test_run_groups_are_atomic_and_manifest_rows_align(tmp_path: Path) -> None:
    root, _ = _write_fixture(tmp_path / "dataset")
    observed: dict[str, str] = {}
    for split in ("train", "val", "test"):
        rows = [
            json.loads(line)
            for line in (root / split / "manifest.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        for row_index, row in enumerate(rows):
            assert row["row_index"] == row_index
            assert row["split"] == split
            assert observed.setdefault(row["run_group_id"], split) == split


def test_existing_root_and_checksum_corruption_are_rejected(tmp_path: Path) -> None:
    target, identities, metadata, provenance = generate_synthetic_chassis_batch(
        SyntheticChassisConfig(sample_count=256)
    )
    root = tmp_path / "dataset"
    write_chassis_execution_dataset(
        root,
        target=target,
        identities=identities,
        metadata=metadata,
        provenance=provenance,
    )
    with pytest.raises(ChassisExecutionStorageError, match="already exists"):
        write_chassis_execution_dataset(
            root,
            target=target,
            identities=identities,
            metadata=metadata,
            provenance=provenance,
        )
    path = root / "train" / "tau_cmd.npy"
    value = np.load(path, allow_pickle=False)
    value[0, 0, 0, 0] += 1.0
    np.save(path, value, allow_pickle=False)
    with pytest.raises(ChassisExecutionStorageError, match="checksum mismatch"):
        verify_chassis_execution_dataset(root)


def test_bad_split_and_invalid_config_are_rejected(tmp_path: Path) -> None:
    root, _ = _write_fixture(tmp_path / "dataset")
    with pytest.raises(ChassisExecutionStorageError, match="split must"):
        ChassisExecutionDataset(root, split="dev")
    with pytest.raises(ValueError, match="group_size"):
        SyntheticChassisConfig(sample_count=4, group_size=0)
    with pytest.raises(ValueError, match="divisible"):
        SyntheticChassisConfig(sample_count=5, group_size=4)
