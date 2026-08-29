from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_eval_grpo.sh"
STAGE1_RUN3_ROOT = (
    "/media/kong/Elements_SE/Diffusion_Data/outputs/bev_diffusion_stage1/run_3"
)


def _fake_python(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-c" ]]; then
  exec "$REAL_PYTHON" "$@"
fi

module=""
if [[ "${1:-}" == "-m" ]]; then
  module="$2"
fi
printf '%s\n' "$*" >> "$CALL_LOG"

if [[ "$module" == "evaluation.bev_reward_comparison" ]]; then
  output=""
  while (( "$#" )); do
    if [[ "$1" == "--output" ]]; then
      output="$2"
      shift 2
    else
      shift
    fi
  done
  case "${FAKE_EVALUATOR_MODE:-pass}" in
    fatal)
      exit 7
      ;;
    incomplete)
      mkdir -p "$(dirname "$output")"
      printf 'model,total_reward\nstage1_a,0\n' > "$output"
      exit 0
      ;;
    pass)
      ;;
    *)
      exit 64
      ;;
  esac
  mkdir -p "$(dirname "$output")"
  printf '%s\n' \
    'model,scenario,route,seed,step,total_reward,progress_reward,formation_reward,gap_reward,ttc_reward,road_reward,comfort_reward,collision_reward,out_of_drivable_reward' \
    'stage1_a,S5_hard_brake_lead,R1_entry_straight,31,4,-0.2,0.4,-0.1,-0.2,-0.1,-0.1,-0.1,0.0,0.0' \
    'grpo_open,S5_hard_brake_lead,R1_entry_straight,31,4,-0.1,0.5,-0.1,-0.2,-0.1,-0.1,-0.1,0.0,0.0' \
    > "$output"
  exit 0
fi

if [[ "$module" == "evaluation.plot_grpo_reward_boxplots" ]]; then
  output_dir=""
  while (( "$#" )); do
    if [[ "$1" == "--output-dir" ]]; then
      output_dir="$2"
      shift 2
    else
      shift
    fi
  done
  if [[ "${FAKE_PLOT_EXIT:-0}" != "0" ]]; then
    exit "$FAKE_PLOT_EXIT"
  fi
  mkdir -p "$output_dir"
  printf 'fake png\n' > "$output_dir/total_progress_ttc_comfort_reward_boxplots.png"
  for name in formation gap road collision out_of_drivable; do
    printf 'fake png\n' > "$output_dir/${name}_reward_boxplot.png"
  done
  exit 0
fi

exit 64
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
        "REAL_PYTHON": sys.executable,
        "DEVICE": "cpu",
    }
    return environment, stage1, grpo


def test_launcher_defaults_to_reward_only_comparison() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert (
        f'STAGE1_RUN_ROOT="${{STAGE1_RUN_ROOT:-{STAGE1_RUN3_ROOT}}}"  # *'
        in source
    )
    assert (
        'GRPO_RUN_ROOT="${GRPO_RUN_ROOT:-${STAGE1_RUN_ROOT}/grpo_open/run_4}"  # *'
        in source
    )
    assert (
        'OUTPUT_ROOT="${OUTPUT_ROOT:-${STAGE1_RUN_ROOT}/evaluation/reward_compare}"'
        in source
    )
    assert 'REWARD_CSV="${REWARD_CSV:-${OUTPUT_ROOT}/step_rewards.csv}"' in source
    assert 'BOXPLOT_ROOT="${BOXPLOT_ROOT:-${OUTPUT_ROOT}/boxplots}"' in source
    assert 'MAX_STEPS="${MAX_STEPS:-800}"' in source
    assert "evaluation.bev_four_model_evaluator" not in source
    assert "evaluation.bev_comparison_charts" not in source


def test_launcher_writes_one_csv_and_six_boxplots(tmp_path: Path) -> None:
    environment, _, _ = _environment(tmp_path)

    subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    output_root = Path(environment["OUTPUT_ROOT"])
    assert (output_root / "step_rewards.csv").is_file()
    assert len(list((output_root / "boxplots").glob("*.png"))) == 6
    assert (
        output_root
        / "boxplots"
        / "total_progress_ttc_comfort_reward_boxplots.png"
    ).is_file()
    assert (output_root / "logs" / "reward_evaluation.log").is_file()
    assert (output_root / "logs" / "reward_boxplots.log").is_file()
    assert not (output_root / "manifest_v2.json").exists()
    assert not (output_root / "s5_s9_closed_loop.json").exists()
    assert not (output_root / "tables").exists()
    calls = Path(environment["CALL_LOG"]).read_text().splitlines()
    assert len(calls) == 2
    assert sum("-m evaluation.bev_reward_comparison" in call for call in calls) == 1
    assert sum(
        "-m evaluation.plot_grpo_reward_boxplots" in call for call in calls
    ) == 1
    evaluator_call = calls[0]
    assert "--max-steps 800" in evaluator_call
    assert "--device cpu" in evaluator_call
    assert f"--output {output_root / 'step_rewards.csv'}.attempt." in evaluator_call
    assert f"--input-csv {output_root / 'step_rewards.csv'}" in calls[1]


def test_launcher_defaults_outputs_beneath_stage1_run(tmp_path: Path) -> None:
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

    root = stage1.parents[1] / "evaluation" / "reward_compare"
    assert (root / "step_rewards.csv").is_file()
    assert len(list((root / "boxplots").glob("*.png"))) == 6


def test_launcher_overrides_steps_and_rejects_bad_inputs(tmp_path: Path) -> None:
    environment, _, _ = _environment(tmp_path / "override")
    environment["MAX_STEPS"] = "37"
    subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = Path(environment["CALL_LOG"]).read_text().splitlines()
    assert "--max-steps 37" in calls[0]

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


def test_launcher_preserves_previous_csv_when_evaluation_fails(
    tmp_path: Path,
) -> None:
    for mode in ("fatal", "incomplete"):
        environment, _, _ = _environment(tmp_path / mode)
        environment["FAKE_EVALUATOR_MODE"] = mode
        output_root = Path(environment["OUTPUT_ROOT"])
        output_root.mkdir(parents=True)
        reward_csv = output_root / "step_rewards.csv"
        previous = "previous complete reward csv\n"
        reward_csv.write_text(previous, encoding="utf-8")

        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert reward_csv.read_text(encoding="utf-8") == previous
        calls = Path(environment["CALL_LOG"]).read_text().splitlines()
        assert sum("-m evaluation.bev_reward_comparison" in call for call in calls) == 1
        assert not any("plot_grpo_reward_boxplots" in call for call in calls)


def test_launcher_keeps_csv_when_boxplot_generation_fails(tmp_path: Path) -> None:
    environment, _, _ = _environment(tmp_path)
    environment["FAKE_PLOT_EXIT"] = "9"

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 9
    assert (Path(environment["OUTPUT_ROOT"]) / "step_rewards.csv").is_file()
