#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets}"
DATASET_NAME="${DATASET_NAME:-metaIDM_test}"
TRAJECTORY_KEY="${TRAJECTORY_KEY:-trajectory}"

DATASET_ROOT="${DATASET_ROOT:-${OUTPUT_ROOT}/${DATASET_NAME}}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/metadrive/exp_dataset/anchors.npy}"
NUM_ANCHORS="${NUM_ANCHORS:-20}"
SEED="${SEED:-0}"
FIGURE_PATH="${FIGURE_PATH:-${REPO_ROOT}/metadrive/exp_dataset/anchors.png}"
NO_SHOW="${NO_SHOW:-1}"

ARGS=(
    --dataset-root "${DATASET_ROOT}"
    --output-path "${OUTPUT_PATH}"
    --trajectory-key "${TRAJECTORY_KEY}"
    --num-anchors "${NUM_ANCHORS}"
    --seed "${SEED}"
    --figure-path "${FIGURE_PATH}"
)

if [[ "${NO_SHOW}" == "1" ]]; then
    ARGS+=(--no-show)
fi

"${PYTHON_BIN}" -m metadrive.exp_dataset.abstract_anchors_default "${ARGS[@]}"
