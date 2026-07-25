from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/run_multiData_collection.sh"


def test_script_runs_joint_bev_entrypoint_and_forwards_explicit_overrides() -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": "echo",
            "DATASET_CONFIG_PATH": "/tmp/data_collect.yaml",
            "DATASET_ROOT": "/tmp/joint_bev",
            "TARGET_JOINT_STEPS": "10",
            "MAX_EPISODES": "2",
            "MAX_EPISODE_STEPS": "20",
            "RESUME": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert result.stdout.strip() == (
        "-m expert_dataset.run_joint_bev_collection "
        "--config /tmp/data_collect.yaml "
        "--dataset-root /tmp/joint_bev "
        "--target-joint-steps 10 --max-episodes 2 "
        "--max-episode-steps 20 --resume 0"
    )
    assert "collect_multi_experts" not in result.stdout
    assert "run_multi_data_pipeline" not in result.stdout
