#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERT_NAME="${EXPERT_NAME:-idm}" # "ppo" or "idm"
TRAJECTORY_KEY="${TRAJECTORY_KEY:-trajectory}" # "trajectory" 规则修正后, "trajectory_raw" ppo直接输出轨迹

DEFAULT_DATASET_ROOT="/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metadrive_${EXPERT_NAME}"
DEFAULT_OUTPUT_PATH="metadrive/exp_dataset/metadrive_anchors_${EXPERT_NAME}.npy"
DEFAULT_FIGURE_PATH="metadrive/exp_dataset/metadrive_anchors_${EXPERT_NAME}.png"

DATASET_ROOT="${DATASET_ROOT:-${DEFAULT_DATASET_ROOT}}"
OUTPUT_PATH="${OUTPUT_PATH:-${DEFAULT_OUTPUT_PATH}}"
NUM_ANCHORS="${NUM_ANCHORS:-9}"
SEED="${SEED:-0}"
FIGURE_PATH="${FIGURE_PATH:-${DEFAULT_FIGURE_PATH}}"
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

"${PYTHON_BIN}" -m metadrive.exp_dataset.abstract_anchors "${ARGS[@]}"
