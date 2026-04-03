#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/kong/anaconda3/envs/meta_drive/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets"
DATASET_NAME="metaIDM"
REPORT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/reports"
LOG_PATH="${REPORT_DIR}/${DATASET_NAME}.log"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${REPORT_DIR}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m metadrive.exp_dataset.collect_expert \
  --target-samples 200 \
  --output-root "${OUTPUT_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --save-video true \
  --expert-type idm \
  --start-seed 30 \
  --trajectory-correction-enabled true \
  --save-raw-trajectory true \
  --mode-classifier-version v1 \
  --centerline-attraction-strength 0.85 \
  --smoothing-strength 0.25 \
  --trajectory-visualization-enabled true \
  --trajectory-visualization-front-margin 25 \
  --trajectory-visualization-lateral-margin 10 \
  --traffic-density-min 0.08 \
  --traffic-density-max 0.12 \
  --use-hybrid-map true \
  --hybrid-map-sequence SSXCOCSS \
  --map-block-num 5 \
  --num-scenarios 1 \
  2>&1 | tee "${LOG_PATH}"
