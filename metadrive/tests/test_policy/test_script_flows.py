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


def test_preview_scenario_script_supports_mode_generate():
    stdout = _run_script(
        "preview_scenario.sh",
        PYTHON_BIN="/bin/echo",
        OUTPUT_ROOT="/tmp/scenario_preview",
        MANIFEST_PATH="/tmp/does_not_exist.json",
        SCENARIO_ID="S8_ego_exit_to_ramp",
        NUM_EPISODES="1",
        MODE_GENERATE="1",
        MODE_FRAME_LIMIT="5",
    )

    assert "--mode-generate-enabled true" in stdout
    assert "--mode-generate-frame-limit 5" in stdout


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


def test_run_data_pipeline_script_supports_k_means_anchor_method():
    stdout = _run_script(
        "run_data_pipeline.sh",
        PYTHON_BIN="/bin/echo",
        STAGE="anchors",
        ANCHOR_METHOD="k_means",
        OUTPUT_ROOT="/tmp/datasets",
        DATASET_NAME="metaIDM_test",
    )

    assert "ANCHOR_METHOD=k_means" in stdout
    assert "abstract_anchors_default" in stdout
    assert "--dataset-root /tmp/datasets/metaIDM_test" in stdout


def test_run_data_pipeline_script_supports_dynamic_anchor_method():
    stdout = _run_script(
        "run_data_pipeline.sh",
        PYTHON_BIN="/bin/echo",
        STAGE="anchors",
        ANCHOR_METHOD="dynamic",
        OUTPUT_ROOT="/tmp/datasets",
        DATASET_NAME="metaIDM_test",
    )

    assert "ANCHOR_METHOD=dynamic" in stdout
    assert "Dynamic anchors use mode-generator trajectories from the dataset/live context." in stdout
    assert "Skipping static anchor extraction; no anchors.npy will be generated in dynamic mode." in stdout
    assert "abstract_anchors_default" not in stdout


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


def test_run_diffusion_train_script_skips_plan_anchor_in_dynamic_mode():
    stdout = _run_script(
        "run_diffusion_train.sh",
        PYTHON_BIN="/bin/echo",
        ANCHOR_METHOD="dynamic",
        TRAJECTORY_REG_DECODER_TYPE="gru",
    )

    assert "--anchor-method dynamic" in stdout
    assert "--plan-anchor-path" not in stdout
    assert "--trajectory-reg-decoder-type gru" in stdout


def test_run_diffusion_train_script_keeps_plan_anchor_in_k_means_mode():
    stdout = _run_script(
        "run_diffusion_train.sh",
        PYTHON_BIN="/bin/echo",
        ANCHOR_METHOD="k_means",
    )

    assert "--anchor-method k_means" in stdout
    assert "--plan-anchor-path metadrive/exp_dataset/anchors.npy" in stdout


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
    assert "--save-trajectory-plots 1" in stdout
    assert "--save-csv 1" in stdout


def test_run_diffusion_open_loop_eval_skips_plan_anchor_in_dynamic_mode():
    stdout = _run_script(
        "run_diffusion_open_loop_eval.sh",
        PYTHON_BIN="/bin/echo",
        CHECKPOINT_PATH="dummy.ckpt",
        ANCHOR_METHOD="dynamic",
        TRAJECTORY_REG_DECODER_TYPE="gru",
    )

    assert "--anchor-method dynamic" in stdout
    assert "--plan-anchor-path" not in stdout
    assert "--trajectory-reg-decoder-type gru" in stdout


def test_run_diffusion_test_script_supports_gru_decoder_type():
    stdout = _run_script(
        "run_diffusion_test.sh",
        PYTHON_BIN="/bin/echo",
        CHECKPOINT_PATH="dummy.ckpt",
        TRAJECTORY_REG_DECODER_TYPE="gru",
    )

    assert "--trajectory-reg-decoder-type gru" in stdout


def test_diffusion_scripts_share_visible_reg_decoder_comment_template():
    expected_comment = '# REG decoder: set TRAJECTORY_REG_DECODER_TYPE to "mlp" or "gru".'
    for script_name in (
        "run_diffusion_train.sh",
        "run_diffusion_open_loop_eval.sh",
        "run_diffusion_test.sh",
    ):
        content = (SCRIPTS_DIR / script_name).read_text(encoding="utf-8")
        assert expected_comment in content


def test_run_diffusion_open_loop_eval_keeps_plan_anchor_in_k_means_mode():
    stdout = _run_script(
        "run_diffusion_open_loop_eval.sh",
        PYTHON_BIN="/bin/echo",
        CHECKPOINT_PATH="dummy.ckpt",
        ANCHOR_METHOD="k_means",
    )

    assert "--anchor-method k_means" in stdout
    assert "--plan-anchor-path" in stdout


def test_test_transfuser_policy_wrapper_delegates_to_main_script():
    stdout = _run_script(
        "test_transfuser_policy.sh",
        PYTHON_BIN="/bin/echo",
        CHECKPOINT_PATH="dummy.ckpt",
    )

    assert "test_transfuser_policy" in stdout
