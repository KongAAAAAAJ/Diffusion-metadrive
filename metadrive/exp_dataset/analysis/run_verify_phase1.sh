#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"

"${PYTHON_BIN}" metadrive/exp_dataset/analysis/verify_phase1.py \
  --dataset-root /media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metaIDM_test \
  --output-dir /media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metaIDM_test/reports/phase1_verify \
  --max-traj-plots 500 \
  --heatmap-bins 200
