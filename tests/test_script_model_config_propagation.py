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


def test_grpo_compare_passes_fixed_route_and_max_steps_to_both_runs() -> None:
    output = _run_script(
        "scripts/run_grpo_test.sh",
        {
            "PYTHON_BIN": "echo",
            "BASELINE_CKPT": "/tmp/baseline.ckpt",
            "GRPO_CKPT": "/tmp/grpo.ckpt",
            "OUTPUT_BASE": "/tmp/grpo_compare_script_test",
            "SCENARIO_ID": "S2_free_cruise_curve",
            "LOCAL_ROUTE": "R2_entry_curve",
            "MAX_STEPS": "12",
            "EPISODES": "1",
        },
    )

    assert output.count("--scenario-id S2_free_cruise_curve") == 2
    assert output.count("--local-route R2_entry_curve") == 2
    assert output.count("--max-steps 12") == 2
    assert output.count("--random-traffic 0") == 2


def test_alternating_grpo_resolves_best_step_checkpoint_before_final(tmp_path) -> None:
    run_dir = tmp_path / "run_1"
    final_dir = run_dir / "checkpoints" / "final"
    low_dir = run_dir / "checkpoints" / "step_0001000_score_1.2500"
    high_dir = run_dir / "checkpoints" / "step_0002000_score_3.5000"
    for ckpt_dir in (final_dir, low_dir, high_dir):
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "full_platoon_grpo.ckpt").write_text(ckpt_dir.name)

    output = _run_script(
        "scripts/run_alternating_grpo_train.sh",
        {},
        "--resolve-ckpt",
        str(run_dir),
        "full_platoon_grpo.ckpt",
    )

    assert output.strip() == str(high_dir / "full_platoon_grpo.ckpt")
