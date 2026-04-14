#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets}"
DATASET_NAME="${DATASET_NAME:-metaIDM}"
DATASET_ROOT="${DATASET_ROOT:-${OUTPUT_ROOT}/${DATASET_NAME}}"
VERIFY_OUTPUT_DIR="${VERIFY_OUTPUT_DIR:-${DATASET_ROOT}/reports/phase1_verify}"

"${PYTHON_BIN}" metadrive/exp_dataset/analysis/verify_phase1.py \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${VERIFY_OUTPUT_DIR}" \
  --max-traj-plots 50000 \
  --heatmap-bins 200
