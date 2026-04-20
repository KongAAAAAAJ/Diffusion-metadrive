#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
ANCHOR_METHOD="${ANCHOR_METHOD:-dynamic}"
TRAJECTORY_REG_DECODER_TYPE="${TRAJECTORY_REG_DECODER_TYPE:-gru}"  # mlp | gru
DATA_ROOT="${DATA_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets}"
DATASET_NAME="${DATASET_NAME:-metaIDM}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${DATA_ROOT}/${DATASET_NAME}_pp}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_12/checkpoints/diffusion-epoch=43.ckpt}"
PLAN_ANCHOR_PATH="${PLAN_ANCHOR_PATH:-${REPO_ROOT}/metadrive/exp_dataset/anchors.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/eval/open}"

if [[ "${ANCHOR_METHOD}" == "k_means" ]]; then
    "${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.eval_transfuser_open_loop \
        --checkpoint "${CHECKPOINT_PATH}" \
        --dataset-root "${PREPROCESSED_ROOT}" \
        --dataset-format auto \
        --split val \
        --num-samples 500 \
        --batch-size 8 \
        --num-workers 0 \
        --device auto \
        --anchor-method k_means \
        --trajectory-reg-decoder-type "${TRAJECTORY_REG_DECODER_TYPE}" \
        --plan-anchor-path "${PLAN_ANCHOR_PATH}" \
        --output-dir "${OUTPUT_DIR}" \
        --save-images 1 \
        --save-json 1 \
        --save-trajectory-plots 1 \
        --save-csv 1 \
        --overlay-all-anchors 1
else
    "${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.eval_transfuser_open_loop \
        --checkpoint "${CHECKPOINT_PATH}" \
        --dataset-root "${PREPROCESSED_ROOT}" \
        --dataset-format auto \
        --split val \
        --num-samples 500 \
        --batch-size 8 \
        --num-workers 0 \
        --device auto \
        --anchor-method dynamic \
        --trajectory-reg-decoder-type "${TRAJECTORY_REG_DECODER_TYPE}" \
        --output-dir "${OUTPUT_DIR}" \
        --save-images 1 \
        --save-json 1 \
        --save-trajectory-plots 1 \
        --save-csv 1 \
        --overlay-all-anchors 1
fi
