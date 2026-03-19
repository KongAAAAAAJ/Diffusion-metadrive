#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/checkpoints/diffusion-epoch=98.ckpt}"
DATASET_ROOT="${DATASET_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metadrive_ppo_preprocessed_small_dir}"
DATASET_FORMAT="${DATASET_FORMAT:-auto}"
SPLIT="${SPLIT:-val}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"
PLAN_ANCHOR_PATH="${PLAN_ANCHOR_PATH:-metadrive/exp_dataset/metadrive_anchors.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/open_loop_eval}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
SAVE_JSON="${SAVE_JSON:-1}"
SAVE_TRAJECTORY_PLOTS="${SAVE_TRAJECTORY_PLOTS:-1}"
SAVE_CSV="${SAVE_CSV:-1}"
OVERLAY_ALL_ANCHORS="${OVERLAY_ALL_ANCHORS:-1}"
MODE_FOCUS="${MODE_FOCUS:-7}"

"${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.eval_transfuser_open_loop \
    --checkpoint "${CHECKPOINT_PATH}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-format "${DATASET_FORMAT}" \
    --split "${SPLIT}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --plan-anchor-path "${PLAN_ANCHOR_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --save-images "${SAVE_IMAGES}" \
    --save-json "${SAVE_JSON}" \
    --save-trajectory-plots "${SAVE_TRAJECTORY_PLOTS}" \
    --save-csv "${SAVE_CSV}" \
    --overlay-all-anchors "${OVERLAY_ALL_ANCHORS}" \
    --mode-focus "${MODE_FOCUS}"
