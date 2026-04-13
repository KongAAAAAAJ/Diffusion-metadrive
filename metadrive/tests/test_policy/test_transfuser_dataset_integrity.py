from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from metadrive.policy.diffusion_policy.preprocess_transfuser_dataset import (
    OUTPUT_FORMAT_DIR,
    output_shard_path,
    preprocess_shard,
)
from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config
from metadrive.policy.diffusion_policy.transfuser_features import MetaDriveTransfuserDataset, sample_to_features_targets
from metadrive.policy.diffusion_policy.verify_transfuser_dataset import (
    AUTO_DATASET_FORMAT,
    PROCESSED_DIR_DATASET_FORMAT,
    PROCESSED_NPZ_LEGACY_DATASET_FORMAT,
    RAW_DATASET_FORMAT,
    infer_dataset_format,
    verify_dataset,
    verify_preprocessed_shard,
)


def _write_split(dataset_root: Path, split: str, shard_names) -> None:
    split_path = dataset_root / "splits" / f"{split}.txt"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text("\n".join(shard_names) + "\n", encoding="utf-8")


def _preprocessed_payload(num_samples: int = 2):
    return {
        "camera_feature": np.zeros((num_samples, 3, 8, 12), dtype=np.float32),
        "lidar_feature": np.zeros((num_samples, 1, 8, 8), dtype=np.float32),
        "status_feature": np.zeros((num_samples, 8), dtype=np.float32),
        "ego_state": np.zeros((num_samples, 8), dtype=np.float32),
        "target_point": np.zeros((num_samples, 2), dtype=np.float32),
        "trajectory": np.zeros((num_samples, 8, 3), dtype=np.float32),
        "agent_states": np.zeros((num_samples, 16, 5), dtype=np.float32),
        "agent_labels": np.zeros((num_samples, 16), dtype=bool),
        "bev_semantic_map": np.zeros((num_samples, 128, 256), dtype=np.uint8),
    }


def _raw_payload(num_samples: int = 2):
    return {
        "left_camera": np.zeros((num_samples, 10, 16, 3), dtype=np.uint8),
        "front_camera": np.zeros((num_samples, 10, 16, 3), dtype=np.uint8),
        "right_camera": np.zeros((num_samples, 10, 16, 3), dtype=np.uint8),
        "lidar": np.zeros((num_samples, 32), dtype=np.float32),
        "ego_state": np.zeros((num_samples, 8), dtype=np.float32),
        "trajectory": np.zeros((num_samples, 8, 3), dtype=np.float32),
        "trajectory_raw": np.zeros((num_samples, 8, 3), dtype=np.float32),
        "trajectory_mode": np.zeros((num_samples,), dtype=np.int8),
        "agent_states": np.zeros((num_samples, 16, 5), dtype=np.float32),
        "agent_labels": np.zeros((num_samples, 16), dtype=bool),
        "bev_raster": np.zeros((num_samples, 3, 16, 16), dtype=np.uint8),
    }


def _write_npz(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def test_preprocess_shard_writes_processed_dir(tmp_path: Path):
    source_shard = tmp_path / "raw" / "shards" / "shard_000000.npz"
    output_root = tmp_path / "processed"
    _write_npz(source_shard, _raw_payload())

    output_path = output_shard_path(output_root, source_shard.name, OUTPUT_FORMAT_DIR)
    preprocess_shard(source_shard, output_path, build_transfuser_config("small"), output_format=OUTPUT_FORMAT_DIR)

    assert output_path.is_dir()
    assert (output_path / "meta.json").exists()
    for field in (
        "camera_feature",
        "lidar_feature",
        "status_feature",
        "ego_state",
        "target_point",
        "trajectory",
        "agent_states",
        "agent_labels",
        "bev_semantic_map",
    ):
        assert (output_path / f"{field}.npy").exists()

    meta = json.loads((output_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["dataset_format"] == "processed_dir"
    assert meta["storage_format"] == "dir"
    assert meta["num_samples"] == 2


def test_infer_dataset_format_for_raw_processed_dir_and_legacy(tmp_path: Path):
    raw_path = tmp_path / "raw" / "shards" / "shard_raw.npz"
    legacy_path = tmp_path / "legacy" / "shards" / "shard_legacy.npz"
    processed_dir = tmp_path / "processed" / "shards" / "shard_legacy"
    _write_npz(raw_path, _raw_payload())
    _write_npz(legacy_path, _preprocessed_payload())
    preprocess_shard(raw_path, processed_dir, build_transfuser_config("small"))

    assert infer_dataset_format(raw_path) == RAW_DATASET_FORMAT
    assert infer_dataset_format(legacy_path) == PROCESSED_NPZ_LEGACY_DATASET_FORMAT
    assert infer_dataset_format(processed_dir) == PROCESSED_DIR_DATASET_FORMAT


def test_verify_dataset_auto_detects_processed_dir_schema(tmp_path: Path):
    dataset_root = tmp_path / "processed_dataset"
    raw_shard = tmp_path / "source" / "shards" / "shard_000000.npz"
    _write_npz(raw_shard, _raw_payload())
    processed_path = output_shard_path(dataset_root, raw_shard.name, OUTPUT_FORMAT_DIR)
    preprocess_shard(raw_shard, processed_path, build_transfuser_config("small"), output_format=OUTPUT_FORMAT_DIR)
    _write_split(dataset_root, "train", [raw_shard.stem])

    report = verify_dataset(
        dataset_root=dataset_root,
        split="train",
        model_size="small",
        dataset_format=AUTO_DATASET_FORMAT,
    )

    assert report["dataset_format"] == PROCESSED_DIR_DATASET_FORMAT
    assert report["storage_format"] == OUTPUT_FORMAT_DIR
    assert report["summary"]["ok"] == 1
    assert report["shards"][0]["schema"] == PROCESSED_DIR_DATASET_FORMAT


def test_verify_dataset_only_checks_requested_split(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    raw_ok = tmp_path / "source" / "shards" / "shard_000012.npz"
    _write_npz(raw_ok, _raw_payload())
    processed_dir = output_shard_path(dataset_root, raw_ok.name, OUTPUT_FORMAT_DIR)
    preprocess_shard(raw_ok, processed_dir, build_transfuser_config("small"), output_format=OUTPUT_FORMAT_DIR)
    (dataset_root / "shards").mkdir(parents=True, exist_ok=True)
    (dataset_root / "shards" / "test_bad.npz").write_bytes(b"not a zip file")
    _write_split(dataset_root, "train", [raw_ok.stem])
    _write_split(dataset_root, "test", ["test_bad.npz"])

    report = verify_dataset(
        dataset_root=dataset_root,
        split="train",
        model_size="small",
    )

    assert report["summary"]["ok"] == 1
    assert report["summary"]["corrupt"] == 0
    assert len(report["shards"]) == 1


def test_verify_dataset_repairs_legacy_npz_to_processed_dir(tmp_path: Path):
    source_root = tmp_path / "source_dataset"
    dataset_root = tmp_path / "preprocessed_dataset"
    shard_name = "shard_000002.npz"

    _write_npz(source_root / "shards" / shard_name, _raw_payload())
    (dataset_root / "shards").mkdir(parents=True, exist_ok=True)
    (dataset_root / "shards" / shard_name).write_bytes(b"broken npz")
    _write_split(dataset_root, "train", [shard_name])

    report = verify_dataset(
        dataset_root=dataset_root,
        split="train",
        model_size="small",
        dataset_format=PROCESSED_NPZ_LEGACY_DATASET_FORMAT,
        repair_corrupt_shards=True,
        source_dataset_root=source_root,
    )

    repaired_entry = report["shards"][0]
    repaired_dir = dataset_root / "shards" / Path(shard_name).stem
    assert report["summary"]["repaired"] == 1
    assert repaired_entry["status"] == "repaired"
    assert repaired_entry["schema"] == PROCESSED_DIR_DATASET_FORMAT
    assert repaired_dir.is_dir()
    assert verify_preprocessed_shard(repaired_dir).status == "ok"


def test_verify_dataset_rejects_explicit_format_mismatch(tmp_path: Path):
    dataset_root = tmp_path / "raw_dataset"
    shard_name = "train_raw.npz"
    _write_npz(dataset_root / "shards" / shard_name, _raw_payload())
    _write_split(dataset_root, "train", [shard_name])

    with pytest.raises(ValueError, match="does not match inferred format"):
        verify_dataset(
            dataset_root=dataset_root,
            split="train",
            model_size="small",
            dataset_format=PROCESSED_DIR_DATASET_FORMAT,
        )


def test_verify_dataset_rejects_processed_dir_structural_mismatch(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    raw_shard = tmp_path / "source" / "shards" / "shard_000004.npz"
    _write_npz(raw_shard, _raw_payload())
    processed_dir = output_shard_path(dataset_root, raw_shard.name, OUTPUT_FORMAT_DIR)
    preprocess_shard(raw_shard, processed_dir, build_transfuser_config("small"), output_format=OUTPUT_FORMAT_DIR)
    _write_split(dataset_root, "train", [raw_shard.stem])

    meta = json.loads((processed_dir / "meta.json").read_text(encoding="utf-8"))
    meta["fields"]["agent_labels"]["shape"] = [1, 16]
    (processed_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="corrupt"):
        verify_dataset(
            dataset_root=dataset_root,
            split="train",
            model_size="small",
            dataset_format=PROCESSED_DIR_DATASET_FORMAT,
        )


def test_verify_dataset_rejects_raw_missing_key(tmp_path: Path):
    dataset_root = tmp_path / "raw_dataset"
    shard_name = "shard_000010.npz"
    payload = _raw_payload()
    payload.pop("lidar")
    _write_npz(dataset_root / "shards" / shard_name, payload)
    _write_split(dataset_root, "train", [shard_name])

    with pytest.raises(RuntimeError, match="corrupt"):
        verify_dataset(
            dataset_root=dataset_root,
            split="train",
            model_size="small",
            dataset_format=RAW_DATASET_FORMAT,
        )

    report = json.loads((dataset_root / "reports" / "integrity_report.json").read_text(encoding="utf-8"))
    assert report["dataset_format"] == RAW_DATASET_FORMAT
    assert report["summary"]["corrupt"] == 1


def test_verify_dataset_fails_when_source_missing(tmp_path: Path):
    dataset_root = tmp_path / "preprocessed_dataset"
    shard_name = "shard_000003.npz"

    (dataset_root / "shards").mkdir(parents=True, exist_ok=True)
    (dataset_root / "shards" / shard_name).write_bytes(b"broken npz")
    _write_split(dataset_root, "train", [shard_name])

    with pytest.raises(RuntimeError, match="missing_source"):
        verify_dataset(
            dataset_root=dataset_root,
            split="train",
            model_size="small",
            dataset_format=PROCESSED_NPZ_LEGACY_DATASET_FORMAT,
            repair_corrupt_shards=True,
            source_dataset_root=tmp_path / "missing_source",
        )


def test_raw_sample_reuses_bev_semantic_map():
    config = build_transfuser_config("small")
    sample = {key: value[0] for key, value in _raw_payload(num_samples=1).items()}
    sample["bev_semantic_map"] = np.full((84, 84), 3, dtype=np.int64)

    _, targets = sample_to_features_targets(sample, config)

    assert tuple(targets["bev_semantic_map"].shape) == (128, 256)
    assert int(targets["bev_semantic_map"][0, 0]) == 3


def test_processed_dir_dataset_returns_sample(tmp_path: Path):
    dataset_root = tmp_path / "processed_dataset"
    raw_shard = tmp_path / "source" / "shards" / "shard_000011.npz"
    _write_npz(raw_shard, _raw_payload())
    processed_dir = output_shard_path(dataset_root, raw_shard.name, OUTPUT_FORMAT_DIR)
    preprocess_shard(raw_shard, processed_dir, build_transfuser_config("small"), output_format=OUTPUT_FORMAT_DIR)
    _write_split(dataset_root, "train", [raw_shard.stem])

    dataset = MetaDriveTransfuserDataset(dataset_root, build_transfuser_config("small"), split="train")
    features, targets = dataset[0]

    assert tuple(features["camera_feature"].shape) == (3, 256, 768)
    assert tuple(targets["bev_semantic_map"].shape) == (128, 256)


def test_dataset_metadata_exposes_optional_trajectory_mode(tmp_path: Path):
    dataset_root = tmp_path / "raw_dataset"
    shard_name = "shard_000111.npz"
    payload = _raw_payload(num_samples=2)
    payload["trajectory_mode"][:] = np.asarray([1, 4], dtype=np.int8)
    _write_npz(dataset_root / "shards" / shard_name, payload)
    _write_split(dataset_root, "train", [shard_name])

    dataset = MetaDriveTransfuserDataset(dataset_root, build_transfuser_config("small"), split="train")

    metadata = dataset.get_sample_metadata(1)

    assert metadata["trajectory_mode"] == 4
