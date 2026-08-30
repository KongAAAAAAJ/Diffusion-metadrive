from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_bev_joint_grpo.sh"
DEFAULT_CONFIG = ROOT / "configs" / "train" / "bev_joint_grpo.yaml"
DEFAULT_ARTIFACT_ROOT = Path(
    "/media/kong/Elements_SE/Diffusion_Data/outputs/"
    "bev_diffusion_stage1/run_3/grpo_open"
)
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
)


def _fake_python(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$CALL_LOG"
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _environment(tmp_path: Path) -> dict[str, str]:
    environment = {
        **os.environ,
        "PYTHON_BIN": str(_fake_python(tmp_path / "fake-python")),
        "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "CALL_LOG": str(tmp_path / "calls.log"),
    }
    for name in FORBIDDEN_ENV:
        environment.pop(name, None)
    return environment


def test_launcher_forwards_only_yaml_config_and_output_root(tmp_path: Path) -> None:
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
    assert f"--output-root {environment['ARTIFACT_ROOT']}" in calls[0]
    for removed_option in (
        "--variant",
        "--run-mode",
        "--source-checkpoint",
        "--max-rollout-groups",
    ):
        assert removed_option not in calls[0]
    assert "calibrat" not in calls[0]
    assert (
        Path(environment["ARTIFACT_ROOT"])
        / "logs"
        / "grpo-open-training.log"
    ).is_file()
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
    assert "--max-rollout-groups" not in call


def test_launcher_uses_static_default_artifact_root() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert (
        f'ARTIFACT_ROOT="${{ARTIFACT_ROOT:-{DEFAULT_ARTIFACT_ROOT}}}"'
        in source
    )
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-$ARTIFACT_ROOT}"' in source
    assert 'LOG_ROOT="${LOG_ROOT:-${ARTIFACT_ROOT}/logs}"' in source
    assert "SOURCE_CHECKPOINT_DIR" not in source
    assert "SOURCE_RUN_ROOT" not in source
    assert "bev_joint_grpo_open_tau_d_v1" not in source
    assert "training_v1" not in source
    assert "logs_v1" not in source
    assert "calibrate_bev_joint_reward.py" not in source
