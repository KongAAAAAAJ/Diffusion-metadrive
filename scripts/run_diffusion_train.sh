#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
ANCHOR_METHOD="${ANCHOR_METHOD:-dynamic}"
TRAJECTORY_REG_DECODER_TYPE="${TRAJECTORY_REG_DECODER_TYPE:-gru}"  # mlp | gru

if [[ "${ANCHOR_METHOD}" == "k_means" ]]; then
  "${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.train_transfuser \
    --model-size small \
    --dataset-root /media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metaData_strong_idm_single_preprocessed \
    --anchor-method k_means \
    --trajectory-reg-decoder-type "${TRAJECTORY_REG_DECODER_TYPE}" \
    --plan-anchor-path metadrive/exp_dataset/anchors.npy \
    --dataset-format auto \
    --num-workers 8 \
    --precision auto \
    --check-val-every-n-epoch 1 \
    --val-visualization-interval 5 \
    --max-epochs 50 \
    --cache-shards-in-memory
else
  "${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.train_transfuser \
    --model-size small \
    --dataset-root /media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metaData_strong_idm_single_preprocessed \
    --anchor-method dynamic \
    --trajectory-reg-decoder-type "${TRAJECTORY_REG_DECODER_TYPE}" \
    --dataset-format auto \
    --num-workers 8 \
    --precision auto \
    --check-val-every-n-epoch 1 \
    --val-visualization-interval 5 \
    --max-epochs 50 \
    --cache-shards-in-memory
fi
