#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
STAGE1_RUN_ROOT="${STAGE1_RUN_ROOT:-/media/kong/Elements_SE/Diffusion_Data/outputs/bev_diffusion_stage1/run_3}"  # *
STAGE1_CHECKPOINT_PATH="${STAGE1_CHECKPOINT_PATH:-${STAGE1_RUN_ROOT}/checkpoints/best.pt}"
GRPO_RUN_ROOT="${GRPO_RUN_ROOT:-${STAGE1_RUN_ROOT}/grpo_open/run_4}"  # *
GRPO_CHECKPOINT_PATH="${GRPO_CHECKPOINT_PATH:-${GRPO_RUN_ROOT}/checkpoints/best.pt}"
EXPECTED_STAGE1_CHECKPOINT_SHA256="${EXPECTED_STAGE1_CHECKPOINT_SHA256:-}"
EXPECTED_GRPO_CHECKPOINT_SHA256="${EXPECTED_GRPO_CHECKPOINT_SHA256:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${STAGE1_RUN_ROOT}/evaluation/reward_compare}"
REWARD_CSV="${REWARD_CSV:-${OUTPUT_ROOT}/step_rewards.csv}"
BOXPLOT_ROOT="${BOXPLOT_ROOT:-${OUTPUT_ROOT}/boxplots}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"

DEVICE="${DEVICE:-cuda}"
MAX_STEPS="${MAX_STEPS:-800}"

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

mkdir -p "$(dirname "${REWARD_CSV}")" "${BOXPLOT_ROOT}" "${LOG_ROOT}"

echo "[grpo-reward-evaluation] stage1_checkpoint=${STAGE1_CHECKPOINT_PATH}"
echo "[grpo-reward-evaluation] stage1_checkpoint_sha256=${STAGE1_CHECKPOINT_SHA256}"
echo "[grpo-reward-evaluation] grpo_checkpoint=${GRPO_CHECKPOINT_PATH}"
echo "[grpo-reward-evaluation] grpo_checkpoint_sha256=${GRPO_CHECKPOINT_SHA256}"
echo "[grpo-reward-evaluation] reward_csv=${REWARD_CSV}"
echo "[grpo-reward-evaluation] boxplot_root=${BOXPLOT_ROOT}"
echo "[grpo-reward-evaluation] device=${DEVICE} max_steps=${MAX_STEPS}"

REWARD_ATTEMPT="${REWARD_CSV}.attempt.$$"
trap 'rm -f -- "${REWARD_ATTEMPT}"' EXIT

set +e
"${PYTHON_BIN}" -m evaluation.bev_reward_comparison \
  --stage1-checkpoint "${STAGE1_CHECKPOINT_PATH}" \
  --grpo-checkpoint "${GRPO_CHECKPOINT_PATH}" \
  --output "${REWARD_ATTEMPT}" \
  --device "${DEVICE}" \
  --max-steps "${MAX_STEPS}" \
  2>&1 | tee "${LOG_ROOT}/reward_evaluation.log"
evaluation_pipeline_status=("${PIPESTATUS[@]}")
set -e
evaluation_status="${evaluation_pipeline_status[0]}"
evaluation_log_status="${evaluation_pipeline_status[1]}"

if [[ "${evaluation_status}" != "0" ]]; then
  echo "[grpo-reward-evaluation] reward evaluator failed (status=${evaluation_status}); boxplots were not generated" >&2
  exit "${evaluation_status}"
fi
if [[ "${evaluation_log_status}" != "0" ]]; then
  echo "[grpo-reward-evaluation] writing evaluator log failed" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c '
import csv
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
reward_columns = (
    "total_reward", "progress_reward", "formation_reward", "gap_reward",
    "ttc_reward", "road_reward", "comfort_reward", "collision_reward",
    "out_of_drivable_reward",
)
required = {
    "model", "scenario", "route", "seed", "step",
    *reward_columns,
}
with path.open("r", encoding="utf-8", newline="") as stream:
    reader = csv.DictReader(stream)
    if set(reader.fieldnames or ()) != required:
        raise SystemExit(1)
    rows = list(reader)
if not rows or {row["model"] for row in rows} != {"stage1_a", "grpo_open"}:
    raise SystemExit(1)
for row in rows:
    values = [float(row[name]) for name in reward_columns]
    if not all(map(math.isfinite, values)):
        raise SystemExit(1)
    if not math.isclose(values[0], sum(values[1:]), rel_tol=1e-6, abs_tol=1e-6):
        raise SystemExit(1)
' "${REWARD_ATTEMPT}"; then
  echo "[grpo-reward-evaluation] evaluator did not produce a complete reward CSV; boxplots were not generated" >&2
  exit 1
fi

mv -f -- "${REWARD_ATTEMPT}" "${REWARD_CSV}"

set +e
"${PYTHON_BIN}" -m evaluation.plot_grpo_reward_boxplots \
  --input-csv "${REWARD_CSV}" \
  --output-dir "${BOXPLOT_ROOT}" \
  2>&1 | tee "${LOG_ROOT}/reward_boxplots.log"
plot_pipeline_status=("${PIPESTATUS[@]}")
set -e
plot_status="${plot_pipeline_status[0]}"
plot_log_status="${plot_pipeline_status[1]}"

if [[ "${plot_status}" != "0" ]]; then
  echo "[grpo-reward-evaluation] reward boxplot generation failed (status=${plot_status})" >&2
  exit "${plot_status}"
fi
if [[ "${plot_log_status}" != "0" ]]; then
  echo "[grpo-reward-evaluation] writing boxplot log failed" >&2
  exit 1
fi

echo "[grpo-reward-evaluation] completed"
