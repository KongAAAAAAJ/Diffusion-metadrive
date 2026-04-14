#!/usr/bin/env bash
set -euo pipefail

# Optional offline feature caching only. Training does not require this step by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
DATA_ROOT="${DATA_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets}"
DATASET_NAME="${DATASET_NAME:-metaIDM_test}"
INPUT_ROOT="${INPUT_ROOT:-${DATA_ROOT}/${DATASET_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/${DATASET_NAME}_pp}"
MODEL_SIZE="${MODEL_SIZE:-small}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-dir}"
OVERWRITE="${OVERWRITE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

"${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.preprocess_transfuser_dataset \
    --input-root "${INPUT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --output-format "${OUTPUT_FORMAT}" \
    --model-size "${MODEL_SIZE}" \
    --skip-existing "${SKIP_EXISTING}" \
    $([[ "${OVERWRITE}" == "1" ]] && printf '%s' "--overwrite")
