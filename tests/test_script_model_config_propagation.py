from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script(script: str, env: dict[str, str], *args: str) -> str:
    merged_env = os.environ.copy()
    merged_env.update(env)
    result = subprocess.run(
        ["bash", str(REPO_ROOT / script), *args],
        cwd=REPO_ROOT,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    return result.stdout


def test_preview_scenario_passes_model_config_to_collect_expert() -> None:
    output = _run_script(
        "scripts/preview_scenario.sh",
        {
            "PYTHON_BIN": "echo",
            "MODEL_CONFIG_PATH": "/tmp/custom_model.yaml",
            "OUTPUT_ROOT": "/tmp/preview_model_config_test",
        },
        "S1_free_cruise_straight",
        "1",
        "--mode-generate",
    )

    assert "--model-config-path /tmp/custom_model.yaml" in output


def test_data_pipeline_collect_passes_model_config_to_collect_expert() -> None:
    output = _run_script(
        "scripts/run_data_pipeline.sh",
        {
            "PYTHON_BIN": "echo",
            "MODEL_CONFIG_PATH": "/tmp/custom_model.yaml",
            "OUTPUT_ROOT": "/tmp/data_pipeline_model_config_test",
            "DATASET_NAME": "model_config_smoke",
            "ANCHOR_METHOD": "dynamic",
            "STAGE": "collect",
        },
    )

    assert "--model-config-path /tmp/custom_model.yaml" in output
