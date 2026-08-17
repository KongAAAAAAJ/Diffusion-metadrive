from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_bev_joint_grpo.sh"


def _fake_python(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$CALL_LOG"
output=""
phase=""
while (( "$#" )); do
  case "$1" in
    --output)
      output="$2"
      shift 2
      ;;
    --phase)
      phase="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -n "$output" ]]; then
  mkdir -p "$(dirname "$output")"
  printf '{"phase":"%s"}\\n' "$phase" > "$output"
fi
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _environment(tmp_path: Path) -> dict[str, str]:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"stage1")
    call_log = tmp_path / "calls.log"
    return {
        **os.environ,
        "PYTHON_BIN": str(_fake_python(tmp_path / "fake-python")),
        "SOURCE_CHECKPOINT": str(checkpoint),
        "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "CALL_LOG": str(call_log),
        "MAX_OPTIMIZER_STEPS": "37",
    }


def test_all_runs_development_holdout_then_bounded_training(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = Path(environment["CALL_LOG"]).read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3
    assert "calibrate_bev_joint_reward.py" in calls[0]
    assert "--phase development" in calls[0]
    assert "calibrate_bev_joint_reward.py" in calls[1]
    assert "--phase holdout" in calls[1]
    assert "--tracking-envelope-report" in calls[1]
    assert "-m train.train_bev_joint_grpo_online" in calls[2]
    assert "--variant A" in calls[2]
    assert "--run-mode smoke" in calls[2]
    assert "--max-optimizer-steps 37" in calls[2]


def test_formal_mode_is_rejected_before_any_process_starts(tmp_path: Path) -> None:
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


def test_train_stage_requires_holdout_report(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    environment["PIPELINE_STAGE"] = "train"
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "holdout calibration report does not exist" in result.stderr
    assert not Path(environment["CALL_LOG"]).exists()


def test_explicit_bypass_runs_only_training_with_auditable_flag(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    environment["PIPELINE_STAGE"] = "train"
    environment["ALLOW_FAILED_CALIBRATION_DIAGNOSTIC"] = "1"
    calibration_report = (
        Path(environment["ARTIFACT_ROOT"]) / "calibration" / "A-holdout.json"
    )
    calibration_report.parent.mkdir(parents=True)
    calibration_report.write_text('{"passed":false}\n', encoding="utf-8")

    subprocess.run(
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
    assert "--allow-failed-calibration-diagnostic" in calls[0]
    assert "training_calibration_bypass" in calls[0]


def test_bypass_rejects_calibration_or_all_pipeline(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    environment["ALLOW_FAILED_CALIBRATION_DIAGNOSTIC"] = "1"
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "requires PIPELINE_STAGE=train" in result.stderr
    assert not Path(environment["CALL_LOG"]).exists()
