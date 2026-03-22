from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import json


DATASET_ROOT = Path("/tmp/phase2_test/test_run")
LOG_PATH = Path("/tmp/phase2_test/collect_command.log")


def _shard_files() -> list[Path]:
    shard_dir = DATASET_ROOT / "shards"
    if not shard_dir.exists():
        return []
    return sorted(shard_dir.glob("*.npz")) + sorted(shard_dir.glob("*.pkl"))


def test_collect_command_log_has_no_import_errors():
    assert LOG_PATH.exists(), f"missing collection log: {LOG_PATH}"
    text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")
    assert "ImportError" not in text
    assert "ModuleNotFoundError" not in text


def test_collection_produces_non_empty_shard_file():
    shard_files = _shard_files()
    assert len(shard_files) >= 1, shard_files
    assert any(path.stat().st_size > 0 for path in shard_files)


def test_collection_finishes_under_ten_minutes():
    manifest_path = DATASET_ROOT / "reports" / "manifest.json"
    assert manifest_path.exists(), manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert float(manifest["collection_wall_time_sec"]) < 600.0


def test_dataset_contains_required_fields():
    shard_files = _shard_files()
    assert shard_files, "no shard files found"
    shard_path = next(path for path in shard_files if path.suffix == ".npz")
    with np.load(shard_path, allow_pickle=False) as shard:
        keys = set(shard.files)
        assert ("camera" in keys) or ("rgb" in keys), keys
        assert "front_camera" in keys, keys
        assert "lidar" in keys, keys
        assert "ego_state" in keys, keys
        assert "trajectory" in keys, keys
        front_camera = np.asarray(shard["front_camera"])
        assert front_camera.ndim == 4, front_camera.shape
        assert front_camera.shape[-1] == 3, front_camera.shape
