from __future__ import annotations

import json
import subprocess
import sys

from tools.check_dataset_stats import build_dataset_stats, parse_args


def test_check_integrity_argument_is_supported(data_dir):
    args = parse_args(["--dataset-root", str(data_dir), "--check-integrity"])
    assert args.check_integrity is True


def test_integrity_summary_meets_phase2_thresholds(data_dir):
    stats = build_dataset_stats(data_dir, check_integrity=True)
    integrity = stats["integrity"]

    assert stats["total_samples"] >= 500, stats
    assert integrity["sample_count_ok"] is True, integrity
    assert integrity["finite_ok"] is True, integrity
    assert integrity["trajectory_range_ok"] is True, integrity
    assert integrity["passed"] is True, integrity

    skill_coverage = integrity["skill_coverage"]
    assert skill_coverage["straight"] >= 100, skill_coverage
    assert skill_coverage["turn"] >= 50, skill_coverage
    assert skill_coverage["lane_change"] >= 30, skill_coverage
    assert skill_coverage["obstacle_avoidance"] >= 20, skill_coverage


def test_check_integrity_cli_outputs_json(data_dir, repo_root):
    result = subprocess.run(
        [
            sys.executable,
            "tools/check_dataset_stats.py",
            "--dataset-root",
            str(data_dir),
            "--check-integrity",
        ],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    stats = json.loads(result.stdout)
    assert "integrity" in stats, stats
    assert stats["integrity"]["passed"] is True, stats
