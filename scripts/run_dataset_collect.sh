#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET_SAMPLES="${TARGET_SAMPLES:-20000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets}"
EXPERT_TYPE="${EXPERT_TYPE:-idm}"
START_SEED="${START_SEED:-0}"
DATASET_NAME="${DATASET_NAME:-meta_data_${EXPERT_TYPE}}"

"${PYTHON_BIN}" -m metadrive.exp_dataset.collect_expert \
    --target-samples "${TARGET_SAMPLES}" \
    --output-root "${OUTPUT_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --expert-type "${EXPERT_TYPE}" \
    --start-seed "${START_SEED}"
