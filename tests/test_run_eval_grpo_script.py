from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_eval_grpo.sh"
STAGE1_RUN3_ROOT = (
    "/media/kong/Elements_SE/Diffusion_Data/outputs/bev_diffusion_stage1/run_3"
)
GRPO_RUN4_ROOT = f"{STAGE1_RUN3_ROOT}/grpo_open/run_4"


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


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    stage1 = tmp_path / "stage1" / "checkpoints" / "best.pt"
    grpo = tmp_path / "grpo" / "checkpoints" / "best.pt"
    stage1.parent.mkdir(parents=True)
    grpo.parent.mkdir(parents=True)
    stage1.write_bytes(b"stage1-run3")
    grpo.write_bytes(b"grpo-run4")
    environment = {
        **os.environ,
        "PYTHON_BIN": str(_fake_python(tmp_path / "fake-python")),
        "STAGE1_CHECKPOINT_PATH": str(stage1),
        "GRPO_CHECKPOINT_PATH": str(grpo),
        "OUTPUT_ROOT": str(tmp_path / "evaluation"),
        "CALL_LOG": str(tmp_path / "calls.log"),
        "DEVICE": "cpu",
    }
    return environment, stage1, grpo


def test_launcher_defaults_to_stage1_run3_and_grpo_run4() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert f'STAGE1_RUN_ROOT="${{STAGE1_RUN_ROOT:-{STAGE1_RUN3_ROOT}}}"' in source
    assert (
        'GRPO_RUN_ROOT="${GRPO_RUN_ROOT:-${STAGE1_RUN_ROOT}/grpo_open/run_4}"'
        in source
    )
    assert GRPO_RUN4_ROOT.endswith("bev_diffusion_stage1/run_3/grpo_open/run_4")
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-${STAGE1_RUN_ROOT}/evaluation/compare}"' in source
    assert 'MAX_STEPS="${MAX_STEPS:-200}"' in source
    assert 'SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-1}"' in source


def test_launcher_defaults_all_comparison_outputs_beneath_stage1_run(
    tmp_path: Path,
) -> None:
    environment, stage1, _ = _environment(tmp_path)
    environment["STAGE1_RUN_ROOT"] = str(stage1.parents[1])
    environment.pop("OUTPUT_ROOT")

    subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    output_root = stage1.parents[1] / "evaluation" / "compare"
    assert (output_root / "manifest_v2.json").is_file()
    assert (output_root / "s5_s9_closed_loop.json").is_file()
    assert (output_root / "logs" / "s5_s9_closed_loop.log").is_file()
    assert (output_root / "artifacts").is_dir()
    call = Path(environment["CALL_LOG"]).read_text(encoding="utf-8")
    assert f"--output {output_root / 's5_s9_closed_loop.json'}" in call
    assert f"--artifact-root {output_root / 'artifacts'}" in call


def test_launcher_writes_comparison_manifest_and_calls_evaluator_once(
    tmp_path: Path,
) -> None:
    environment, stage1, grpo = _environment(tmp_path)
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
    stage1_sha = hashlib.sha256(stage1.read_bytes()).hexdigest()
    grpo_sha = hashlib.sha256(grpo.read_bytes()).hexdigest()
    assert manifest == {
        "format": "bev_model_evaluation_manifest_v2",
        "models": [
            {
                "id": "stage1_a",
                "kind": "stage1",
                "variant": "A",
                "checkpoint": str(stage1),
                "checkpoint_sha256": stage1_sha,
            },
            {
                "id": "grpo_open",
                "kind": "grpo",
                "variant": "A",
                "reward_domain": "tau_d",
                "checkpoint": str(grpo),
                "checkpoint_sha256": grpo_sha,
                "source_checkpoint": str(stage1),
                "source_checkpoint_sha256": stage1_sha,
            },
        ],
        "comparisons": [
            {
                "id": "open_vs_stage1",
                "baseline": "stage1_a",
                "candidate": "grpo_open",
            }
        ],
    }

    calls = Path(environment["CALL_LOG"]).read_text().splitlines()
    assert len(calls) == 1
    call = calls[0]
    assert "-m evaluation.bev_four_model_evaluator" in call
    assert "--run-mode diagnostic" in call
    assert "--max-steps 200" in call
    assert "--repeats 1" in call
    assert "--save-visualizations" in call
    assert f"--output {output_root / 's5_s9_closed_loop.json'}" in call


def test_launcher_overrides_runtime_and_rejects_bad_inputs(tmp_path: Path) -> None:
    environment, _, _ = _environment(tmp_path / "override")
    environment.update(
        {
            "MAX_STEPS": "37",
            "REPEATS": "2",
            "SAVE_VISUALIZATIONS": "0",
        }
    )
    subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    call = Path(environment["CALL_LOG"]).read_text()
    assert "--max-steps 37" in call
    assert "--repeats 2" in call
    assert "--save-visualizations" not in call

    bad_hash, _, _ = _environment(tmp_path / "bad_hash")
    bad_hash["EXPECTED_GRPO_CHECKPOINT_SHA256"] = "0" * 64
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=bad_hash,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "GRPO checkpoint SHA256 mismatch" in result.stderr
    assert not Path(bad_hash["CALL_LOG"]).exists()

    invalid_steps, _, _ = _environment(tmp_path / "invalid_steps")
    invalid_steps["MAX_STEPS"] = "0"
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=invalid_steps,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "MAX_STEPS must be a positive integer" in result.stderr
    assert not Path(invalid_steps["CALL_LOG"]).exists()

    missing, _, _ = _environment(tmp_path / "missing")
    missing["GRPO_CHECKPOINT_PATH"] = str(tmp_path / "missing.pt")
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=missing,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "GRPO checkpoint does not exist" in result.stderr
    assert not Path(missing["CALL_LOG"]).exists()
