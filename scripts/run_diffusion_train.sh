#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_SIZE="${MODEL_SIZE:-small}"
EXPERT_NAME="${EXPERT_NAME:-idm}"
DATASET_ROOT="${DATASET_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metaData_idm_single_preprocessed}"
PLAN_ANCHOR_PATH="${PLAN_ANCHOR_PATH:-metadrive/exp_dataset/anchors.npy}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-auto}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
VAL_VIS_INTERVAL="${VAL_VIS_INTERVAL:-5}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
DATASET_FORMAT="${DATASET_FORMAT:-auto}"

"${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.train_transfuser \
    --model-size "${MODEL_SIZE}" \
    --dataset-root "${DATASET_ROOT}" \
    --plan-anchor-path "${PLAN_ANCHOR_PATH}" \
    --dataset-format "${DATASET_FORMAT}" \
    --num-workers "${NUM_WORKERS}" \
    --precision "${PRECISION}" \
    --check-val-every-n-epoch "${CHECK_VAL_EVERY_N_EPOCH}" \
    --val-visualization-interval "${VAL_VIS_INTERVAL}" \
    --max-epochs "${MAX_EPOCHS}" \
    --cache-shards-in-memory
