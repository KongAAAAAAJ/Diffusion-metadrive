from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_stage1_a_evaluation.sh"


def _fake_python(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$CALL_LOG"
output=""
while (( "$#" )); do
  if [[ "$1" == "--output" ]]; then
    output="$2"
    shift 2
  else
    shift
  fi
done
if [[ -n "$output" ]]; then
  mkdir -p "$(dirname "$output")"
  printf '{}\\n' > "$output"
fi
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _environment(tmp_path: Path) -> dict[str, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"stage1-a")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    return {
        **os.environ,
        "PYTHON_BIN": str(_fake_python(tmp_path / "fake-python")),
        "RUN_ROOT": str(tmp_path / "run_1"),
        "CHECKPOINT_PATH": str(checkpoint),
        "EXPECTED_CHECKPOINT_SHA256": digest,
        "DATASET_ROOT": str(dataset_root),
        "OUTPUT_ROOT": str(tmp_path / "evaluation"),
        "CALL_LOG": str(tmp_path / "calls.log"),
        "DEVICE": "cpu",
        "MAX_STEPS": "37",
        "REPEATS": "2",
    }


def test_all_writes_single_model_manifest_and_runs_both_evaluations(
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

    output_root = Path(environment["OUTPUT_ROOT"])
    manifest = json.loads((output_root / "manifest_v2.json").read_text())
    assert manifest == {
        "format": "bev_model_evaluation_manifest_v2",
        "models": [
            {
                "id": "stage1_a",
                "kind": "stage1",
                "variant": "A",
                "checkpoint": environment["CHECKPOINT_PATH"],
                "checkpoint_sha256": environment["EXPECTED_CHECKPOINT_SHA256"],
            }
        ],
        "comparisons": [],
    }

    calls = Path(environment["CALL_LOG"]).read_text().splitlines()
    assert len(calls) == 2
    assert "scripts/validate_bev_stage1_ab.py" in calls[0]
    assert f"--dataset-root {environment['DATASET_ROOT']}" in calls[0]
    assert "--device cpu" in calls[0]
    assert "--open-loop-num-samples 1500" in calls[0]
    assert "--save-visualizations" in calls[0]
    assert "--topdown-screen-size 800" in calls[0]
    assert "--topdown-film-size 3000" in calls[0]
    assert "-m evaluation.bev_four_model_evaluator" in calls[1]
    assert "--run-mode diagnostic" in calls[1]
    assert "--max-steps 37" in calls[1]
    assert "--repeats 2" in calls[1]
    assert "--video-fps 10" in calls[1]
    assert "--visualization-interval 1" in calls[1]
    assert "--save-visualizations" in calls[1]
    assert "--topdown-screen-size 800" in calls[1]
    assert "--topdown-film-size 3000" in calls[1]


def test_stage_selector_and_checkpoint_hash_are_strict(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    environment["EVAL_STAGE"] = "open_s1"
    subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = Path(environment["CALL_LOG"]).read_text().splitlines()
    assert len(calls) == 1
    assert "validate_bev_stage1_ab.py" in calls[0]

    environment = _environment(tmp_path / "bad_hash")
    environment["EXPECTED_CHECKPOINT_SHA256"] = "0" * 64
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "checkpoint SHA256 mismatch" in result.stderr
    assert not Path(environment["CALL_LOG"]).exists()
