from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_bev_joint_grpo.sh"
LEGACY_ENV = (
    "PIPELINE_STAGE",
    "ALLOW_FAILED_CALIBRATION_DIAGNOSTIC",
    "DEVELOPMENT_REPORT",
    "CALIBRATION_REPORT",
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
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"stage1")
    environment = {
        **os.environ,
        "PYTHON_BIN": str(_fake_python(tmp_path / "fake-python")),
        "SOURCE_CHECKPOINT": str(checkpoint),
        "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "CALL_LOG": str(tmp_path / "calls.log"),
        "MAX_OPTIMIZER_STEPS": "37",
    }
    for name in LEGACY_ENV:
        environment.pop(name, None)
    return environment


def test_launcher_directly_starts_bounded_tau_d_training(tmp_path: Path) -> None:
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
    assert "--variant A" in calls[0]
    assert "--run-mode smoke" in calls[0]
    assert "--max-optimizer-steps 37" in calls[0]
    assert "calibrat" not in calls[0]
    assert "training_v1" in calls[0]
    assert "reward_domain=tau_d" in result.stdout


def test_launcher_rejects_legacy_calibration_and_pipeline_variables(
    tmp_path: Path,
) -> None:
    for name in LEGACY_ENV:
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


def test_formal_mode_is_rejected_before_training(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    environment["RUN_MODE"] = "formal"
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "requires RUN_MODE=smoke" in result.stderr
    assert not Path(environment["CALL_LOG"]).exists()


def test_launcher_defaults_to_isolated_tau_d_roots() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "bev_joint_grpo_open_tau_d_v1/stage1_run_1" in source
    assert 'DEFAULT_OUTPUT_ROOT="${ARTIFACT_ROOT}/training_v1"' in source
    assert 'DEFAULT_LOG_ROOT="${ARTIFACT_ROOT}/logs_v1"' in source
    assert "calibrate_bev_joint_reward.py" not in source
