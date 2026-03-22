from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tools.check_dataset_stats import build_dataset_stats, parse_args


DATASET_ROOT = Path("/tmp/phase2_test/test_run")


def test_accepts_dataset_root_argument():
    args = parse_args(["--dataset-root", str(DATASET_ROOT)])
    assert args.dataset_root == DATASET_ROOT


def test_outputs_required_summary_structure():
    stats = build_dataset_stats(DATASET_ROOT)
    assert isinstance(stats["total_samples"], int)
    assert isinstance(stats["scenario_coverage"], dict)
    assert isinstance(stats["trajectory_stats"], dict)
    for axis in ("x", "y", "heading"):
        axis_stats = stats["trajectory_stats"][axis]
        assert set(axis_stats.keys()) == {"min", "max", "mean", "std"}


def test_empty_directory_returns_zero_samples(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "tools/check_dataset_stats.py", "--dataset-root", str(tmp_path)],
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    stats = json.loads(result.stdout)
    assert stats["total_samples"] == 0


def test_total_samples_matches_actual_samples():
    stats = build_dataset_stats(DATASET_ROOT)
    total = 0
    for shard_path in sorted((DATASET_ROOT / "shards").glob("*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            total += int(shard["trajectory"].shape[0])
    assert stats["total_samples"] == total
