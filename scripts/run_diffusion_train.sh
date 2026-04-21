#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${REPO_ROOT}/configs/diffusion/model.yaml}"

"${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.train_transfuser \
  --model-config-path "${MODEL_CONFIG_PATH}" \
  --dataset-root /media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metaData_strong_idm_single_preprocessed \
  --dataset-format auto \
  --num-workers 8 \
  --precision auto \
  --check-val-every-n-epoch 1 \
  --val-visualization-interval 5 \
  --max-epochs 50 \
  --cache-shards-in-memory
