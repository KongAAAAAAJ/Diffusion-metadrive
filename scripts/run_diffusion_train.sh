#!/usr/bin/env bash
set -euo pipefail

/home/kong/anaconda3/envs/meta_drive/bin/python -m metadrive.policy.diffusion_policy.train_transfuser \
  --model-size small \
  --dataset-root /media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metaData_strong_idm_single_preprocessed \
  --plan-anchor-path metadrive/exp_dataset/anchors.npy \
  --dataset-format auto \
  --num-workers 8 \
  --precision auto \
  --check-val-every-n-epoch 1 \
  --val-visualization-interval 5 \
  --max-epochs 50 \
  --cache-shards-in-memory
