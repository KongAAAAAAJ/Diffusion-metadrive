from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
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
if [[ "${1:-}" == "-c" ]]; then
  exec "$REAL_PYTHON" "$@"
fi

module=""
if [[ "${1:-}" == "-m" ]]; then
  module="$2"
fi
printf '%s\\n' "$*" >> "$CALL_LOG"

if [[ "$module" == "evaluation.bev_four_model_evaluator" ]]; then
  output=""
  repeats=1
  while (( "$#" )); do
    if [[ "$1" == "--output" ]]; then
      output="$2"
      shift 2
    elif [[ "$1" == "--repeats" ]]; then
      repeats="$2"
      shift 2
    else
      shift
    fi
  done
  case "${FAKE_EVALUATOR_MODE:-pass}" in
    fatal)
      exit 1
      ;;
    incomplete)
      mkdir -p "$(dirname "$output")"
      printf '{}\\n' > "$output"
      exit 0
      ;;
    gate_no_report)
      exit 2
      ;;
    gate_incomplete)
      mkdir -p "$(dirname "$output")"
      printf '{}\\n' > "$output"
      exit 2
      ;;
    pass)
      evaluation_status="completed"
      all_gates_passed="true"
      exit_code=0
      ;;
    gate_failure)
      evaluation_status="completed_with_gate_failure"
      all_gates_passed="false"
      exit_code=2
      ;;
    *)
      exit 64
      ;;
  esac
  report_format="bev_model_evaluation_v3"
  report_tail=('  "comparisons": {}')
  if (( repeats > 1 )); then
    report_format="bev_model_fixed_process_repeat_v3"
    report_tail=('  "comparisons": {},' '  "repeat_reports": []')
  fi
  mkdir -p "$(dirname "$output")"
  printf '%s\\n' \
    '{' \
    "  \\"format\\": \\"${report_format}\\"," \
    "  \\"evaluation_status\\": \\"${evaluation_status}\\"," \
    "  \\"all_gates_passed\\": ${all_gates_passed}," \
    '  "model_order": ["stage1_a", "grpo_open"],' \
    '  "models": {' \
    '    "stage1_a": {"episode_metrics": [], "by_scenario": {}, "overall": {}, "gate_results": {}},' \
    '    "grpo_open": {"episode_metrics": [], "by_scenario": {}, "overall": {}, "gate_results": {}}' \
    '  },' \
    "${report_tail[@]}" \
    '}' > "$output"
  exit "$exit_code"
fi

if [[ "$module" == "evaluation.bev_comparison_charts" ]]; then
  output_root=""
  while (( "$#" )); do
    if [[ "$1" == "--output-root" ]]; then
      output_root="$2"
      shift 2
    else
      shift
    fi
  done
  mkdir -p "$output_root/charts" "$output_root/tables"
  printf '{}\\n' > "$output_root/chart_index.json"
  exit "${FAKE_CHART_EXIT:-0}"
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


def test_launcher_defaults_to_stage1_run3_and_grpo_run4() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert f'STAGE1_RUN_ROOT="${{STAGE1_RUN_ROOT:-{STAGE1_RUN3_ROOT}}}"' in source
    assert (
        'GRPO_RUN_ROOT="${GRPO_RUN_ROOT:-${STAGE1_RUN_ROOT}/grpo_open/run_4}"'
        in source
    )
    assert GRPO_RUN4_ROOT.endswith("bev_diffusion_stage1/run_3/grpo_open/run_4")
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-${STAGE1_RUN_ROOT}/evaluation/compare}"' in source
    assert 'CHART_ROOT="${OUTPUT_ROOT}/charts"' in source
    assert 'MAX_STEPS="${MAX_STEPS:-800}"' in source
    assert 'SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-0}"' in source


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
    assert (output_root / "logs" / "comparison_charts.log").is_file()
    assert (output_root / "artifacts").is_dir()
    assert (output_root / "charts").is_dir()
    assert (output_root / "tables").is_dir()
    assert (output_root / "chart_index.json").is_file()
    calls = Path(environment["CALL_LOG"]).read_text(encoding="utf-8").splitlines()
    evaluator_call = next(
        call for call in calls if "-m evaluation.bev_four_model_evaluator" in call
    )
    chart_call = next(
        call for call in calls if "-m evaluation.bev_comparison_charts" in call
    )
    assert f"--output {output_root / 's5_s9_closed_loop.json'}.attempt." in evaluator_call
    assert f"--artifact-root {output_root / 'artifacts'}" in evaluator_call
    assert f"--report {output_root / 's5_s9_closed_loop.json'}" in chart_call
    assert f"--output-root {output_root}" in chart_call


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
    assert len(calls) == 2
    evaluator_calls = [
        call for call in calls if "-m evaluation.bev_four_model_evaluator" in call
    ]
    chart_calls = [
        call for call in calls if "-m evaluation.bev_comparison_charts" in call
    ]
    assert len(evaluator_calls) == 1
    assert len(chart_calls) == 1
    evaluator_call = evaluator_calls[0]
    assert "--run-mode diagnostic" in evaluator_call
    assert "--max-steps 800" in evaluator_call
    assert "--repeats 1" in evaluator_call
    assert "--save-visualizations" not in evaluator_call
    assert f"--output {output_root / 's5_s9_closed_loop.json'}.attempt." in evaluator_call
    assert f"--report {output_root / 's5_s9_closed_loop.json'}" in chart_calls[0]


def test_launcher_overrides_runtime_and_rejects_bad_inputs(tmp_path: Path) -> None:
    environment, _, _ = _environment(tmp_path / "override")
    environment.update(
        {
            "MAX_STEPS": "37",
            "REPEATS": "2",
            "SAVE_VISUALIZATIONS": "1",
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
    calls = Path(environment["CALL_LOG"]).read_text().splitlines()
    evaluator_call = next(
        call for call in calls if "-m evaluation.bev_four_model_evaluator" in call
    )
    assert "--max-steps 37" in evaluator_call
    assert "--repeats 2" in evaluator_call
    assert "--save-visualizations" in evaluator_call

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


def test_launcher_generates_charts_after_completed_gate_failure(
    tmp_path: Path,
) -> None:
    environment, _, _ = _environment(tmp_path)
    environment["FAKE_EVALUATOR_MODE"] = "gate_failure"

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    output_root = Path(environment["OUTPUT_ROOT"])
    assert result.returncode == 2
    assert (output_root / "s5_s9_closed_loop.json").is_file()
    assert (output_root / "chart_index.json").is_file()
    calls = Path(environment["CALL_LOG"]).read_text().splitlines()
    assert sum("-m evaluation.bev_four_model_evaluator" in call for call in calls) == 1
    assert sum("-m evaluation.bev_comparison_charts" in call for call in calls) == 1
    assert "at least one hard gate failed" in result.stderr


def test_launcher_does_not_chart_fatal_or_incomplete_evaluation(
    tmp_path: Path,
) -> None:
    for mode in ("fatal", "incomplete"):
        environment, _, _ = _environment(tmp_path / mode)
        environment["FAKE_EVALUATOR_MODE"] = mode
        output_root = Path(environment["OUTPUT_ROOT"])
        output_root.mkdir(parents=True)
        report_path = output_root / "s5_s9_closed_loop.json"
        previous_report = '{"format": "previous_complete_report"}\n'
        report_path.write_text(previous_report, encoding="utf-8")

        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert report_path.read_text(encoding="utf-8") == previous_report
        assert not (output_root / "chart_index.json").exists()
        calls = Path(environment["CALL_LOG"]).read_text().splitlines()
        assert sum("-m evaluation.bev_four_model_evaluator" in call for call in calls) == 1
        assert not any("-m evaluation.bev_comparison_charts" in call for call in calls)


def test_launcher_does_not_chart_gate_exit_without_complete_report(
    tmp_path: Path,
) -> None:
    for mode in ("gate_no_report", "gate_incomplete"):
        environment, _, _ = _environment(tmp_path / mode)
        environment["FAKE_EVALUATOR_MODE"] = mode

        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        output_root = Path(environment["OUTPUT_ROOT"])
        assert result.returncode != 0
        assert not (output_root / "s5_s9_closed_loop.json").exists()
        assert not (output_root / "chart_index.json").exists()
        calls = Path(environment["CALL_LOG"]).read_text().splitlines()
        assert sum("-m evaluation.bev_four_model_evaluator" in call for call in calls) == 1
        assert not any("-m evaluation.bev_comparison_charts" in call for call in calls)


def test_launcher_returns_nonzero_when_chart_generation_fails(tmp_path: Path) -> None:
    environment, _, _ = _environment(tmp_path)
    environment["FAKE_CHART_EXIT"] = "7"

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    output_root = Path(environment["OUTPUT_ROOT"])
    assert result.returncode == 7
    assert (output_root / "s5_s9_closed_loop.json").is_file()
    calls = Path(environment["CALL_LOG"]).read_text().splitlines()
    assert sum("-m evaluation.bev_four_model_evaluator" in call for call in calls) == 1
    assert sum("-m evaluation.bev_comparison_charts" in call for call in calls) == 1
    assert "chart generation failed" in result.stderr
