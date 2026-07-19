from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/run_multiData_collection.sh"


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.update(
        {
            "PYTHON_BIN": "echo",
            "OUTPUT_ROOT": "/tmp/multi_pipeline_test",
            "DATASET_NAME": "platoon_smoke",
            "MODEL_CONFIG_PATH": "/tmp/model.yaml",
            "TRAIN_CONFIG_PATH": "/tmp/refine.yaml",
            **env,
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_collect_invokes_multi_expert_with_platoon_arguments() -> None:
    result = _run({"STAGE": "collect", "ANCHOR_METHOD": "dynamic"})

    assert result.returncode == 0, result.stdout
    assert "-m expert_dataset.collect_multi_experts" in result.stdout
    assert "--model-config-path /tmp/model.yaml" in result.stdout
    assert "--train-config-path /tmp/refine.yaml" in result.stdout
    assert "--target-samples 40000" in result.stdout
    assert "--resume 1" in result.stdout
    assert "--num-agents 3" in result.stdout
    assert '--scenario-weights {"S5_hard_brake_lead":1.0,"S6_background_merge_in":1.0}' in result.stdout
    assert "expert_dataset.collect_expert" not in result.stdout


def test_kmeans_anchor_stage_uses_multi_dataset_and_dedicated_output() -> None:
    result = _run({"STAGE": "anchors", "ANCHOR_METHOD": "k_means"})

    assert result.returncode == 0, result.stdout
    assert "-m expert_dataset.abstract_anchors_default" in result.stdout
    assert "--dataset-root /tmp/multi_pipeline_test/platoon_smoke" in result.stdout
    assert "--output-path" in result.stdout
    assert "expert_dataset/platoon_anchors.npy" in result.stdout
    assert "--split all" in result.stdout
    assert "--max-trajectories 100000" in result.stdout


def test_dynamic_anchor_stage_skips_static_extraction() -> None:
    result = _run({"STAGE": "anchors", "ANCHOR_METHOD": "dynamic"})

    assert result.returncode == 0, result.stdout
    assert "Skipping static K-Means extraction" in result.stdout
    assert "abstract_anchors_default" not in result.stdout


def test_preprocess_stage_uses_multi_dataset_paths() -> None:
    result = _run({"STAGE": "preprocess", "ANCHOR_METHOD": "dynamic"})

    assert result.returncode == 0, result.stdout
    assert "-m models.diffusion.preprocess_transfuser_dataset" in result.stdout
    assert "--input-root /tmp/multi_pipeline_test/platoon_smoke" in result.stdout
    assert "--output-root /tmp/multi_pipeline_test/platoon_smoke_pp" in result.stdout
    assert "--model-config-path /tmp/model.yaml" in result.stdout


def test_invalid_stage_and_anchor_method_fail() -> None:
    invalid_stage = _run({"STAGE": "invalid", "ANCHOR_METHOD": "dynamic"})
    invalid_anchor = _run({"STAGE": "anchors", "ANCHOR_METHOD": "invalid"})

    assert invalid_stage.returncode != 0
    assert "Unknown STAGE" in invalid_stage.stdout
    assert invalid_anchor.returncode != 0
    assert "Unsupported ANCHOR_METHOD" in invalid_anchor.stdout
