from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _run_script(script_name: str, **env_overrides) -> str:
    env = os.environ.copy()
    env.update(env_overrides)
    completed = subprocess.run(
        ["bash", str(SCRIPTS_DIR / script_name)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_scripts_are_bash_syntax_valid():
    for script in sorted(SCRIPTS_DIR.glob("*.sh")):
        subprocess.run(["bash", "-n", str(script)], cwd=REPO_ROOT, check=True)


def test_run_diffusion_test_script_passes_explicit_render_flag():
    stdout = _run_script(
        "run_diffusion_test.sh",
        PYTHON_BIN="/bin/echo",
        CHECKPOINT_PATH="dummy.ckpt",
        SCENARIO_ID="S6_background_merge_in",
        RENDER="1",
        EPISODES="2",
    )

    assert "--scenario-id S6_background_merge_in" in stdout
    assert "--render 1" in stdout
    assert "--episodes 2" in stdout


def test_run_diffusion_test_script_supports_headless_mode():
    stdout = _run_script(
        "run_diffusion_test.sh",
        PYTHON_BIN="/bin/echo",
        CHECKPOINT_PATH="dummy.ckpt",
        RENDER="0",
    )

    assert "--render 0" in stdout


def test_run_diffusion_test_script_supports_periodic_camera_saves():
    stdout = _run_script(
        "run_diffusion_test.sh",
        PYTHON_BIN="/bin/echo",
        CHECKPOINT_PATH="dummy.ckpt",
        SAVE_CAMERA_INTERVAL="5",
        CAMERA_OUTPUT_DIR="/tmp/closed_loop_cameras",
    )

    assert "--save-camera-interval 5" in stdout
    assert "--camera-output-dir /tmp/closed_loop_cameras" in stdout


def test_run_dataset_collect_script_uses_single_output_root():
    stdout = _run_script(
        "run_dataset_collect.sh",
        PYTHON_BIN="/bin/echo",
        OUTPUT_ROOT="/tmp/datasets",
    )

    assert "--output-root /tmp/datasets" in stdout
    assert "metadrive_datasets/metadrive_datasets" not in stdout


def test_run_dataset_collect_script_supports_idm_expert():
    stdout = _run_script(
        "run_dataset_collect.sh",
        PYTHON_BIN="/bin/echo",
        EXPERT_TYPE="idm",
    )

    assert "--expert-type idm" in stdout
    assert "metadrive_idm_500_seed10" in stdout


def test_run_dataset_collect_script_supports_trajectory_visualization():
    stdout = _run_script(
        "run_dataset_collect.sh",
        PYTHON_BIN="/bin/echo",
        TRAJECTORY_VISUALIZATION_ENABLED="1",
        TRAJECTORY_VISUALIZATION_FRONT_MARGIN="30",
        TRAJECTORY_VISUALIZATION_LATERAL_MARGIN="12",
    )

    assert "--trajectory-visualization-enabled 1" in stdout
    assert "--trajectory-visualization-front-margin 30" in stdout
    assert "--trajectory-visualization-lateral-margin 12" in stdout


def test_run_abstract_anchors_script_supports_trajectory_key():
    stdout = _run_script(
        "run_abstract_anchors.sh",
        PYTHON_BIN="/bin/echo",
        EXPERT_NAME="ppo",
        TRAJECTORY_KEY="trajectory_raw",
    )

    assert "metadrive_ppo" in stdout
    assert "--trajectory-key trajectory_raw" in stdout
    assert "metadrive_anchors_ppo.npy" in stdout


def test_run_diffusion_preprocess_and_train_defaults_match_readme_contract():
    preprocess_stdout = _run_script("run_diffusion_preprocess.sh", PYTHON_BIN="/bin/echo")
    train_stdout = _run_script("run_diffusion_train.sh", PYTHON_BIN="/bin/echo")

    assert "--output-format dir" in preprocess_stdout
    assert "metadrive_ppo_preprocessed_small_dir" in preprocess_stdout
    assert "metadrive_ppo_preprocessed_small_dir" in train_stdout
    assert "--dataset-format auto" in train_stdout


def test_run_diffusion_convert_camera_layout_is_legacy_but_invocable():
    stdout = _run_script("run_diffusion_convert_camera_layout.sh", PYTHON_BIN="/bin/echo")

    assert "convert_camera_layout_dataset" in stdout
    assert "--input-root" in stdout
    assert "--output-root" in stdout


def test_run_diffusion_open_loop_eval_script_invocable():
    stdout = _run_script(
        "run_diffusion_open_loop_eval.sh",
        PYTHON_BIN="/bin/echo",
        CHECKPOINT_PATH="dummy.ckpt",
    )

    assert "eval_transfuser_open_loop" in stdout
    assert "--split val" in stdout
    assert "--mode-focus 7" in stdout
    assert "--save-trajectory-plots 1" in stdout
    assert "--save-csv 1" in stdout


def test_test_transfuser_policy_wrapper_delegates_to_main_script():
    stdout = _run_script(
        "test_transfuser_policy.sh",
        PYTHON_BIN="/bin/echo",
        CHECKPOINT_PATH="dummy.ckpt",
    )

    assert "test_transfuser_policy" in stdout
