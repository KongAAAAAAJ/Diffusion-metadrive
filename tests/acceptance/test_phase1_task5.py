from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_phase1.py"
LOG_PATH = REPO_ROOT / "logs" / "phase1_acceptance.log"


SUMMARY_PATTERN = re.compile(
    r"summary:\s+success_rate=(?P<success_rate>-?\d+\.\d+)\s+"
    r"collision_rate=(?P<collision_rate>-?\d+\.\d+)\s+"
    r"formation_error=(?P<formation_error>-?\d+\.\d+)\s+"
    r"recovery_time=(?P<recovery_time>-?\d+\.\d+)\s+"
    r"min_inter_vehicle_gap=(?P<min_inter_vehicle_gap>-?\d+\.\d+)"
)


def test_phase1_rollout_acceptance():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--episodes", "5", "--render", "0"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
    )
    LOG_PATH.write_text(result.stdout, encoding="utf-8")

    assert result.returncode in (0, 139), result.stdout

    episode_lines = [line for line in result.stdout.splitlines() if line.startswith("episode=")]
    assert len(episode_lines) == 5, result.stdout

    match = SUMMARY_PATTERN.search(result.stdout)
    assert match is not None, result.stdout
    summary = {key: float(value) for key, value in match.groupdict().items()}

    assert any(value != 0.0 for value in summary.values()), summary
    assert summary["collision_rate"] == 0.0, summary
    assert summary["formation_error"] < 2.0, summary
    assert summary["min_inter_vehicle_gap"] > 0.0, summary
    assert summary["success_rate"] > 0.0, summary
