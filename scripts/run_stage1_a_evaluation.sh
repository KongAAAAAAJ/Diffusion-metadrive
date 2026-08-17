#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
RUN_ROOT="${RUN_ROOT:-/media/kong/Elements_SE/Diffusion_Data/outputs/bev_diffusion_stage1/run_1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RUN_ROOT}/checkpoints/best.pt}"
EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:-ab87ea481fc9bb4b2cd4b74e71d1da284803ecb8a6ca0df972ac0f297d58acfc}"
DATASET_ROOT="${DATASET_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/bev_joint_risk_candidate_v3_formal50k_v1/platoon_joint_bev}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/evaluation/stage1_a_diagnostic}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_ROOT}/manifest_v2.json}"
OPEN_S1_OUTPUT="${OPEN_S1_OUTPUT:-${OUTPUT_ROOT}/open_loop_s1.json}"
CLOSED_OUTPUT="${CLOSED_OUTPUT:-${OUTPUT_ROOT}/s5_s9_closed_loop.json}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
DEVICE="${DEVICE:-cuda}"
MAX_STEPS="${MAX_STEPS:-100}"
REPEATS="${REPEATS:-1}"
EVAL_STAGE="${EVAL_STAGE:-all}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Stage1-A checkpoint does not exist: ${CHECKPOINT_PATH}" >&2
  exit 2
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Stage1-A dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 2
fi
if [[ "${DEVICE}" != "cuda" && "${DEVICE}" != "cpu" ]]; then
  echo "DEVICE must be cuda or cpu" >&2
  exit 2
fi
if [[ ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_STEPS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPEATS must be a positive integer" >&2
  exit 2
fi
if [[ "${EVAL_STAGE}" != "all" && "${EVAL_STAGE}" != "open_s1" && "${EVAL_STAGE}" != "s5_s9" ]]; then
  echo "EVAL_STAGE must be all, open_s1, or s5_s9" >&2
  exit 2
fi

CHECKPOINT_SHA256="$(sha256sum -- "${CHECKPOINT_PATH}" | awk '{print $1}')"
if [[ "${CHECKPOINT_SHA256}" != "${EXPECTED_CHECKPOINT_SHA256}" ]]; then
  echo "Stage1-A checkpoint SHA256 mismatch" >&2
  echo "expected=${EXPECTED_CHECKPOINT_SHA256}" >&2
  echo "observed=${CHECKPOINT_SHA256}" >&2
  exit 2
fi

mkdir -p "$(dirname "${MANIFEST_PATH}")" "$(dirname "${OPEN_S1_OUTPUT}")" "$(dirname "${CLOSED_OUTPUT}")" "${LOG_ROOT}"
printf '%s\n' \
  '{' \
  '  "format": "bev_model_evaluation_manifest_v2",' \
  '  "models": [' \
  '    {' \
  '      "id": "stage1_a",' \
  '      "kind": "stage1",' \
  '      "variant": "A",' \
  "      \"checkpoint\": \"${CHECKPOINT_PATH}\"," \
  "      \"checkpoint_sha256\": \"${CHECKPOINT_SHA256}\"" \
  '    }' \
  '  ],' \
  '  "comparisons": []' \
  '}' > "${MANIFEST_PATH}"

run_logged() {
  local name="$1"
  shift
  "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"
}

run_open_s1() {
  run_logged open_loop_s1 \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/validate_bev_stage1_ab.py" \
      --manifest "${MANIFEST_PATH}" \
      --dataset-root "${DATASET_ROOT}" \
      --device "${DEVICE}" \
      --output "${OPEN_S1_OUTPUT}"
}

run_s5_s9() {
  run_logged s5_s9_closed_loop \
    "${PYTHON_BIN}" -m evaluation.bev_four_model_evaluator \
      --manifest "${MANIFEST_PATH}" \
      --run-mode diagnostic \
      --device "${DEVICE}" \
      --max-steps "${MAX_STEPS}" \
      --repeats "${REPEATS}" \
      --output "${CLOSED_OUTPUT}"
}

echo "[stage1-a-evaluation] checkpoint=${CHECKPOINT_PATH}"
echo "[stage1-a-evaluation] checkpoint_sha256=${CHECKPOINT_SHA256}"
echo "[stage1-a-evaluation] dataset_root=${DATASET_ROOT}"
echo "[stage1-a-evaluation] output_root=${OUTPUT_ROOT}"
echo "[stage1-a-evaluation] eval_stage=${EVAL_STAGE} device=${DEVICE} max_steps=${MAX_STEPS} repeats=${REPEATS}"
echo "[stage1-a-evaluation] classification=diagnostic_only"

case "${EVAL_STAGE}" in
  all)
    run_open_s1
    run_s5_s9
    ;;
  open_s1)
    run_open_s1
    ;;
  s5_s9)
    run_s5_s9
    ;;
esac
