#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${REPO_ROOT}/configs/diffusion/model.yaml}"
DATA_ROOT="${DATA_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets}"
DATASET_NAME="${DATASET_NAME:-metaIDM_0424}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${DATA_ROOT}/${DATASET_NAME}_pp}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_20/checkpoints/diffusion-epoch=25.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/eval/open}"

"${PYTHON_BIN}" -m models.diffusion.eval_transfuser_open_loop \
    --checkpoint "${CHECKPOINT_PATH}" \
    --model-config-path "${MODEL_CONFIG_PATH}" \
    --dataset-root "${PREPROCESSED_ROOT}" \
    --dataset-format auto \
    --split val \
    --num-samples 1500 \
    --batch-size 8 \
    --num-workers 0 \
    --device auto \
    --output-dir "${OUTPUT_DIR}" \
    --save-images 1 \
    --save-json 1 \
    --save-trajectory-plots 1 \
    --save-csv 1 \
    --overlay-all-anchors 1
