#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/kong/anaconda3/envs/meta_drive/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets"
DATASET_NAME="metaIDM_test"
REPORT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/reports"
LOG_PATH="${REPORT_DIR}/${DATASET_NAME}.log"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${REPORT_DIR}"

"${PYTHON_BIN}" -m metadrive.exp_dataset.collect_expert \
    --target-samples 2000 \
    --output-root "${OUTPUT_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --save-videos true \
    --expert-type idm \
    --start-seed 3 \
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
    --local-route-weights '{"R1_entry_straight": 1.0, "R2_entry_curve": 1.0, "R3_mainline_straight": 1.0, "R4_mainline_transition": 1.0, "R5_ramp_curve": 1.0, "R6_exit_to_ramp": 1.0, "R7_merge_core": 1.0, "R8_narrow_channel": 1.0, "R9_post_split_curve": 1.0}' \
    --idm-variant-weights '{"default": 1.0}' \
    --use-hybrid-map true \
    --map-block-num 5 \
    --num-scenarios 1 \
    2>&1 | tee "${LOG_PATH}"
