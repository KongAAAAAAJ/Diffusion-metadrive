from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_bev_joint_grpo.sh"
DEFAULT_CONFIG = ROOT / "configs" / "train" / "bev_joint_grpo.yaml"
DEFAULT_ARTIFACT_ROOT = Path(
    "/media/kong/Elements_SE/Diffusion_Data/outputs/"
    "bev_diffusion_stage1/run_3/grpo_open"
)
REAL_PYTHON = Path("/home/kong/anaconda3/envs/meta_drive/bin/python")
FORBIDDEN_ENV = (
    "PIPELINE_STAGE",
    "ALLOW_FAILED_CALIBRATION_DIAGNOSTIC",
    "DEVELOPMENT_REPORT",
    "CALIBRATION_REPORT",
    "MAX_OPTIMIZER_STEPS",
    "VARIANT",
    "RUN_MODE",
    "SOURCE_CHECKPOINT",
    "MAX_ROLLOUT_GROUPS",
    "LOG_ROOT",
)


def _fake_python(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$CALL_LOG"
if [[ -n "${FAKE_RUN_MARKER:-}" ]]; then
  printf '%s:begin\\n' "$FAKE_RUN_MARKER"
  : > "$FAKE_READY_DIR/$FAKE_RUN_MARKER"
  while [[ ! -e "$FAKE_RELEASE_FILE" ]]; do
    sleep 0.01
  done
  printf '%s:end\\n' "$FAKE_RUN_MARKER"
  printf '%s:error\\n' "$FAKE_RUN_MARKER" >&2
fi
exit "${FAKE_EXIT_CODE:-0}"
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _environment(tmp_path: Path) -> dict[str, str]:
    artifact_root = tmp_path / "artifacts"
    environment = {
        **os.environ,
        "PYTHON_BIN": str(_fake_python(tmp_path / "fake-python")),
        "ARTIFACT_ROOT": str(artifact_root),
        "OUTPUT_ROOT": str(artifact_root),
        "CALL_LOG": str(tmp_path / "calls.log"),
    }
    environment.pop("CONFIG", None)
    for name in FORBIDDEN_ENV:
        environment.pop(name, None)
    return environment


def _training_logs(output_root: Path) -> list[Path]:
    return sorted(
        output_root.glob("run_*/training.log"),
        key=lambda path: int(path.parent.name.removeprefix("run_")),
    )


def _wait_for_files(paths: tuple[Path, ...], *, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not all(path.is_file() for path in paths):
        if time.monotonic() >= deadline:
            missing = [str(path) for path in paths if not path.is_file()]
            raise AssertionError(f"timed out waiting for fake launchers: {missing}")
        time.sleep(0.01)


def test_launcher_allocates_run_and_forwards_only_config_and_run_dir(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = Path(environment["CALL_LOG"]).read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert "-m train.train_bev_joint_grpo_online" in calls[0]
    assert f"--config {DEFAULT_CONFIG}" in calls[0]
    run_dir = Path(environment["OUTPUT_ROOT"]) / "run_1"
    assert f"--run-dir {run_dir}" in calls[0]
    assert "--output-root" not in calls[0]
    for removed_option in (
        "--variant",
        "--run-mode",
        "--source-checkpoint",
        "--max-rollout-groups",
    ):
        assert removed_option not in calls[0]
    assert "calibrat" not in calls[0]
    training_log = run_dir / "training.log"
    assert training_log.is_file()
    log_text = training_log.read_text(encoding="utf-8")
    assert f"[GRPO] config={DEFAULT_CONFIG}" in log_text
    assert f"[GRPO] run_dir={run_dir}" in log_text
    assert "[GRPO] python_exit_status=0 tee_exit_status=0" in log_text
    assert not (
        Path(environment["OUTPUT_ROOT"])
        / "logs"
        / "grpo-open-training.log"
    ).exists()
    assert "application=stage2_grpo_open_application_v3" in result.stdout
    assert "reward_domain=tau_d" in result.stdout


@pytest.mark.parametrize("name", FORBIDDEN_ENV)
def test_launcher_rejects_non_yaml_training_configuration_variables(
    tmp_path: Path,
    name: str,
) -> None:
    environment = _environment(tmp_path)
    environment[name] = "legacy-value"

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "no longer accepted" in result.stderr
    assert not Path(environment["CALL_LOG"]).exists()


def test_launcher_rejects_removed_cli_budget_option(tmp_path: Path) -> None:
    environment = _environment(tmp_path)

    result = subprocess.run(
        ["bash", str(LAUNCHER), "--max-rollout-groups", "1"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "launcher arguments are not accepted" in result.stderr
    assert not Path(environment["CALL_LOG"]).exists()


def test_custom_config_path_is_forwarded_without_parameter_extraction(
    tmp_path: Path,
) -> None:
    custom_config = tmp_path / "custom-grpo.yaml"
    custom_config.write_text("custom: true\n", encoding="utf-8")
    environment = _environment(tmp_path)
    environment["CONFIG"] = str(custom_config)

    subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    call = Path(environment["CALL_LOG"]).read_text(encoding="utf-8")
    assert f"--config {custom_config}" in call
    assert f"--run-dir {environment['OUTPUT_ROOT']}/run_1" in call
    assert "--max-rollout-groups" not in call


def test_launcher_allocates_after_highest_numeric_existing_run(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    output_root = Path(environment["OUTPUT_ROOT"])
    (output_root / "run_4").mkdir(parents=True)

    subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output_root / "run_5" / "training.log").is_file()
    call = Path(environment["CALL_LOG"]).read_text(encoding="utf-8")
    assert f"--run-dir {output_root / 'run_5'}" in call


def test_concurrent_launches_use_distinct_logs_and_preserve_exit_statuses(
    tmp_path: Path,
) -> None:
    base_environment = _environment(tmp_path)
    output_root = Path(base_environment["OUTPUT_ROOT"])
    ready_dir = tmp_path / "ready"
    ready_dir.mkdir()
    release_file = tmp_path / "release"

    failing_environment = {
        **base_environment,
        "CALL_LOG": str(tmp_path / "failing-calls.log"),
        "FAKE_RUN_MARKER": "failing",
        "FAKE_EXIT_CODE": "23",
        "FAKE_READY_DIR": str(ready_dir),
        "FAKE_RELEASE_FILE": str(release_file),
    }
    successful_environment = {
        **base_environment,
        "CALL_LOG": str(tmp_path / "successful-calls.log"),
        "FAKE_RUN_MARKER": "successful",
        "FAKE_EXIT_CODE": "0",
        "FAKE_READY_DIR": str(ready_dir),
        "FAKE_RELEASE_FILE": str(release_file),
    }
    failing = subprocess.Popen(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=failing_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    successful = subprocess.Popen(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=successful_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_files((ready_dir / "failing", ready_dir / "successful"))
        release_file.touch()
        failing_stdout, failing_stderr = failing.communicate(timeout=10)
        successful_stdout, successful_stderr = successful.communicate(timeout=10)
    finally:
        release_file.touch(exist_ok=True)
        for process in (failing, successful):
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    assert failing.returncode == 23
    assert successful.returncode == 0
    assert failing_stderr == ""
    assert successful_stderr == ""
    assert "failing:error" in failing_stdout
    assert "successful:error" in successful_stdout

    logs = _training_logs(output_root)
    assert [path.parent.name for path in logs] == ["run_1", "run_2"]
    log_texts = [path.read_text(encoding="utf-8") for path in logs]
    failing_log = next(text for text in log_texts if "failing:begin" in text)
    successful_log = next(text for text in log_texts if "successful:begin" in text)
    assert "failing:end" in failing_log
    assert "failing:error" in failing_log
    assert "successful" not in failing_log
    assert "[GRPO] python_exit_status=23 tee_exit_status=0" in failing_log
    assert "successful:end" in successful_log
    assert "successful:error" in successful_log
    assert "failing" not in successful_log
    assert "[GRPO] python_exit_status=0 tee_exit_status=0" in successful_log


def test_real_python_missing_source_fails_inside_reserved_run_without_training(
    tmp_path: Path,
) -> None:
    assert REAL_PYTHON.is_file()
    missing_checkpoint = tmp_path / "missing-stage1.ckpt"
    config = tmp_path / "missing-source.yaml"
    config.write_text(
        "\n".join(
            (
                "run:",
                "  variant: A",
                "  run_mode: smoke",
                f"  source_checkpoint: {missing_checkpoint}",
                "online:",
                "  device: cpu",
                "  total_rollout_groups: 1",
                "",
            )
        ),
        encoding="utf-8",
    )
    environment = _environment(tmp_path)
    environment["PYTHON_BIN"] = str(REAL_PYTHON)
    environment["CONFIG"] = str(config)

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    run_dir = Path(environment["OUTPUT_ROOT"]) / "run_1"
    training_log = run_dir / "training.log"
    log_text = training_log.read_text(encoding="utf-8")
    assert str(missing_checkpoint) in log_text
    assert "unable to inspect the Stage 1 source checkpoint" in log_text
    assert "python_exit_status=1 tee_exit_status=0" in log_text
    assert {path.name for path in run_dir.iterdir()} == {
        "checkpoints",
        "training.log",
    }
    assert not any((run_dir / "checkpoints").iterdir())


def test_launcher_uses_static_default_artifact_root() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert (
        f'ARTIFACT_ROOT="${{ARTIFACT_ROOT:-{DEFAULT_ARTIFACT_ROOT}}}"' in source
    )
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-$ARTIFACT_ROOT}"' in source
    assert "LOG_ROOT=" not in source
    assert "grpo-open-training.log" not in source
    assert 'RUN_LOG="$RUN_DIR/training.log"' in source
    assert '--run-dir "$RUN_DIR"' in source
    assert "--output-root" not in source
    assert "SOURCE_CHECKPOINT_DIR" not in source
    assert "SOURCE_RUN_ROOT" not in source
    assert "bev_joint_grpo_open_tau_d_v1" not in source
    assert "training_v1" not in source
    assert "logs_v1" not in source
    assert "calibrate_bev_joint_reward.py" not in source
    assert "stage2_grpo_open_application_v3" in source
    assert "stage2_grpo_open_application_v2" not in source
    assert "stage2_grpo_open_application_v1" not in source


def test_default_yaml_freezes_same_mode_sampling_parameters() -> None:
    payload = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    assert payload["online"]["trajectories_per_mode"] == 48
    assert payload["online"]["max_sampling_attempts_per_state"] == 3
    assert payload["online"]["max_sampling_attempts_multiplier"] == 3
