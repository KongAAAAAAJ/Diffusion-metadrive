from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/run_multiData_collection.sh"


def test_script_forwards_only_config_and_stage_to_python_entrypoint() -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": "echo",
            "DATASET_CONFIG_PATH": "/tmp/data_collect.yaml",
            "STAGE": "anchors",
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
        "-m expert_dataset.run_multi_data_pipeline "
        "--config /tmp/data_collect.yaml --stage anchors"
    )
    assert "collect_multi_experts" not in result.stdout
    assert "abstract_anchors_default" not in result.stdout
    assert "preprocess_transfuser_dataset" not in result.stdout
