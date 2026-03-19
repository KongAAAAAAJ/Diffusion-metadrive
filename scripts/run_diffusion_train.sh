#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_SIZE="${MODEL_SIZE:-small}"
DATASET_ROOT="${DATASET_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/meta_data_ppo_pre_tiny}"
DATASET_FORMAT="${DATASET_FORMAT:-auto}"
PLAN_ANCHOR_PATH="${PLAN_ANCHOR_PATH:-metadrive/exp_dataset/metadrive_anchors.npy}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PRECISION="${PRECISION:-auto}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
VAL_VIS_INTERVAL="${VAL_VIS_INTERVAL:-5}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"

"${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.train_transfuser \
    --model-size "${MODEL_SIZE}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-format "${DATASET_FORMAT}" \
    --plan-anchor-path "${PLAN_ANCHOR_PATH}" \
    --num-workers "${NUM_WORKERS}" \
    --persistent-workers "${PERSISTENT_WORKERS}" \
    --prefetch-factor "${PREFETCH_FACTOR}" \
    --precision "${PRECISION}" \
    --check-val-every-n-epoch "${CHECK_VAL_EVERY_N_EPOCH}" \
    --val-visualization-interval "${VAL_VIS_INTERVAL}" \
    --max-epochs "${MAX_EPOCHS}"
