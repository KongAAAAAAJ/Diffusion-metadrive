#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
STAGE1_RUN_ROOT="${STAGE1_RUN_ROOT:-/media/kong/Elements_SE/Diffusion_Data/outputs/bev_diffusion_stage1/run_3}"
STAGE1_CHECKPOINT_PATH="${STAGE1_CHECKPOINT_PATH:-${STAGE1_RUN_ROOT}/checkpoints/best.pt}"
GRPO_RUN_ROOT="${GRPO_RUN_ROOT:-${STAGE1_RUN_ROOT}/grpo_open/run_4}"
GRPO_CHECKPOINT_PATH="${GRPO_CHECKPOINT_PATH:-${GRPO_RUN_ROOT}/checkpoints/best.pt}"
EXPECTED_STAGE1_CHECKPOINT_SHA256="${EXPECTED_STAGE1_CHECKPOINT_SHA256:-}"
EXPECTED_GRPO_CHECKPOINT_SHA256="${EXPECTED_GRPO_CHECKPOINT_SHA256:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${STAGE1_RUN_ROOT}/evaluation/compare}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_ROOT}/manifest_v2.json}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${OUTPUT_ROOT}/s5_s9_closed_loop.json}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${OUTPUT_ROOT}/artifacts}"
CHART_ROOT="${OUTPUT_ROOT}/charts"

DEVICE="${DEVICE:-cuda}"
MAX_STEPS="${MAX_STEPS:-800}"
REPEATS="${REPEATS:-1}"
SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-0}"
VIDEO_FPS="${VIDEO_FPS:-10}"
VISUALIZATION_INTERVAL="${VISUALIZATION_INTERVAL:-1}"
TOPDOWN_SCREEN_SIZE="${TOPDOWN_SCREEN_SIZE:-800}"
TOPDOWN_FILM_SIZE="${TOPDOWN_FILM_SIZE:-3000}"

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer" >&2
    exit 2
  fi
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${STAGE1_CHECKPOINT_PATH}" ]]; then
  echo "Stage1 checkpoint does not exist: ${STAGE1_CHECKPOINT_PATH}" >&2
  exit 2
fi
if [[ ! -f "${GRPO_CHECKPOINT_PATH}" ]]; then
  echo "GRPO checkpoint does not exist: ${GRPO_CHECKPOINT_PATH}" >&2
  exit 2
fi
if [[ "${DEVICE}" != "cuda" && "${DEVICE}" != "cpu" ]]; then
  echo "DEVICE must be cuda or cpu" >&2
  exit 2
fi
require_positive_integer "MAX_STEPS" "${MAX_STEPS}"
require_positive_integer "REPEATS" "${REPEATS}"
require_positive_integer "VIDEO_FPS" "${VIDEO_FPS}"
require_positive_integer "VISUALIZATION_INTERVAL" "${VISUALIZATION_INTERVAL}"
require_positive_integer "TOPDOWN_SCREEN_SIZE" "${TOPDOWN_SCREEN_SIZE}"
require_positive_integer "TOPDOWN_FILM_SIZE" "${TOPDOWN_FILM_SIZE}"
if [[ "${SAVE_VISUALIZATIONS}" != "0" && "${SAVE_VISUALIZATIONS}" != "1" ]]; then
  echo "SAVE_VISUALIZATIONS must be 0 or 1" >&2
  exit 2
fi

STAGE1_CHECKPOINT_SHA256="$(sha256sum -- "${STAGE1_CHECKPOINT_PATH}" | awk '{print $1}')"
GRPO_CHECKPOINT_SHA256="$(sha256sum -- "${GRPO_CHECKPOINT_PATH}" | awk '{print $1}')"
if [[ -n "${EXPECTED_STAGE1_CHECKPOINT_SHA256}" && "${STAGE1_CHECKPOINT_SHA256}" != "${EXPECTED_STAGE1_CHECKPOINT_SHA256}" ]]; then
  echo "Stage1 checkpoint SHA256 mismatch" >&2
  echo "expected=${EXPECTED_STAGE1_CHECKPOINT_SHA256}" >&2
  echo "observed=${STAGE1_CHECKPOINT_SHA256}" >&2
  exit 2
fi
if [[ -n "${EXPECTED_GRPO_CHECKPOINT_SHA256}" && "${GRPO_CHECKPOINT_SHA256}" != "${EXPECTED_GRPO_CHECKPOINT_SHA256}" ]]; then
  echo "GRPO checkpoint SHA256 mismatch" >&2
  echo "expected=${EXPECTED_GRPO_CHECKPOINT_SHA256}" >&2
  echo "observed=${GRPO_CHECKPOINT_SHA256}" >&2
  exit 2
fi

mkdir -p \
  "$(dirname "${MANIFEST_PATH}")" \
  "$(dirname "${EVAL_OUTPUT}")" \
  "${LOG_ROOT}" \
  "${ARTIFACT_ROOT}"
printf '%s\n' \
  '{' \
  '  "format": "bev_model_evaluation_manifest_v2",' \
  '  "models": [' \
  '    {' \
  '      "id": "stage1_a",' \
  '      "kind": "stage1",' \
  '      "variant": "A",' \
  "      \"checkpoint\": \"${STAGE1_CHECKPOINT_PATH}\"," \
  "      \"checkpoint_sha256\": \"${STAGE1_CHECKPOINT_SHA256}\"" \
  '    },' \
  '    {' \
  '      "id": "grpo_open",' \
  '      "kind": "grpo",' \
  '      "variant": "A",' \
  '      "reward_domain": "tau_d",' \
  "      \"checkpoint\": \"${GRPO_CHECKPOINT_PATH}\"," \
  "      \"checkpoint_sha256\": \"${GRPO_CHECKPOINT_SHA256}\"," \
  "      \"source_checkpoint\": \"${STAGE1_CHECKPOINT_PATH}\"," \
  "      \"source_checkpoint_sha256\": \"${STAGE1_CHECKPOINT_SHA256}\"" \
  '    }' \
  '  ],' \
  '  "comparisons": [' \
  '    {' \
  '      "id": "open_vs_stage1",' \
  '      "baseline": "stage1_a",' \
  '      "candidate": "grpo_open"' \
  '    }' \
  '  ]' \
  '}' > "${MANIFEST_PATH}"

visualization_args=()
if [[ "${SAVE_VISUALIZATIONS}" == "1" ]]; then
  visualization_args+=(--save-visualizations)
fi

echo "[grpo-evaluation] stage1_checkpoint=${STAGE1_CHECKPOINT_PATH}"
echo "[grpo-evaluation] stage1_checkpoint_sha256=${STAGE1_CHECKPOINT_SHA256}"
echo "[grpo-evaluation] grpo_checkpoint=${GRPO_CHECKPOINT_PATH}"
echo "[grpo-evaluation] grpo_checkpoint_sha256=${GRPO_CHECKPOINT_SHA256}"
echo "[grpo-evaluation] output_root=${OUTPUT_ROOT}"
echo "[grpo-evaluation] chart_root=${CHART_ROOT}"
echo "[grpo-evaluation] device=${DEVICE} max_steps=${MAX_STEPS} repeats=${REPEATS}"
echo "[grpo-evaluation] save_visualizations=${SAVE_VISUALIZATIONS} classification=diagnostic_only"

EVAL_ATTEMPT_OUTPUT="${EVAL_OUTPUT}.attempt.$$"
trap 'rm -f -- "${EVAL_ATTEMPT_OUTPUT}"' EXIT

set +e
"${PYTHON_BIN}" -m evaluation.bev_four_model_evaluator \
  --manifest "${MANIFEST_PATH}" \
  --output "${EVAL_ATTEMPT_OUTPUT}" \
  --run-mode diagnostic \
  --device "${DEVICE}" \
  --max-steps "${MAX_STEPS}" \
  --repeats "${REPEATS}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --video-fps "${VIDEO_FPS}" \
  --visualization-interval "${VISUALIZATION_INTERVAL}" \
  --topdown-screen-size "${TOPDOWN_SCREEN_SIZE}" \
  --topdown-film-size "${TOPDOWN_FILM_SIZE}" \
  "${visualization_args[@]}" \
  2>&1 | tee "${LOG_ROOT}/s5_s9_closed_loop.log"
evaluation_pipeline_status=("${PIPESTATUS[@]}")
set -e
evaluator_status="${evaluation_pipeline_status[0]}"
evaluator_log_status="${evaluation_pipeline_status[1]}"

if [[ "${evaluator_status}" != "0" && "${evaluator_status}" != "2" ]]; then
  echo "[grpo-evaluation] evaluator failed before completing the report (status=${evaluator_status}); charts were not generated" >&2
  exit "${evaluator_status}"
fi

if ! "${PYTHON_BIN}" -c '
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
evaluator_status = int(sys.argv[2])
formats = {"bev_model_evaluation_v3", "bev_model_fixed_process_repeat_v3"}
statuses = {"completed", "completed_with_gate_failure"}
expected_order = ["stage1_a", "grpo_open"]

if report.get("format") not in formats:
    raise SystemExit(1)
if report.get("evaluation_status") not in statuses:
    raise SystemExit(1)
if not isinstance(report.get("all_gates_passed"), bool):
    raise SystemExit(1)
if report.get("model_order") != expected_order:
    raise SystemExit(1)
models = report.get("models")
if not isinstance(models, dict) or set(models) != set(expected_order):
    raise SystemExit(1)
for model_id in expected_order:
    model = models[model_id]
    if not isinstance(model, dict):
        raise SystemExit(1)
    if not isinstance(model.get("episode_metrics"), list):
        raise SystemExit(1)
    if not isinstance(model.get("by_scenario"), dict):
        raise SystemExit(1)
    if not isinstance(model.get("overall"), dict):
        raise SystemExit(1)
    if not isinstance(model.get("gate_results"), dict):
        raise SystemExit(1)
if report["format"] == "bev_model_fixed_process_repeat_v3" and not isinstance(
    report.get("repeat_reports"), list
):
    raise SystemExit(1)
if evaluator_status == 0 and not (
    report["evaluation_status"] == "completed" and report["all_gates_passed"]
):
    raise SystemExit(1)
if evaluator_status == 2 and not (
    report["evaluation_status"] == "completed_with_gate_failure"
    and not report["all_gates_passed"]
):
    raise SystemExit(1)
' "${EVAL_ATTEMPT_OUTPUT}" "${evaluator_status}"; then
  echo "[grpo-evaluation] evaluator did not produce a complete v3 report; charts were not generated" >&2
  exit 1
fi

mv -f -- "${EVAL_ATTEMPT_OUTPUT}" "${EVAL_OUTPUT}"

set +e
"${PYTHON_BIN}" -m evaluation.bev_comparison_charts \
  --report "${EVAL_OUTPUT}" \
  --output-root "${OUTPUT_ROOT}" \
  2>&1 | tee "${LOG_ROOT}/comparison_charts.log"
chart_pipeline_status=("${PIPESTATUS[@]}")
set -e
chart_status="${chart_pipeline_status[0]}"
chart_log_status="${chart_pipeline_status[1]}"

if [[ "${chart_status}" != "0" ]]; then
  echo "[grpo-evaluation] chart generation failed (status=${chart_status})" >&2
  exit "${chart_status}"
fi
if [[ "${evaluator_log_status}" != "0" || "${chart_log_status}" != "0" ]]; then
  echo "[grpo-evaluation] writing evaluation logs failed" >&2
  exit 1
fi
if [[ "${evaluator_status}" == "2" ]]; then
  echo "[grpo-evaluation] evaluation completed and charts were generated, but at least one hard gate failed" >&2
  exit 2
fi
