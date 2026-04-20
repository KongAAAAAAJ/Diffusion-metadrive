#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/kong/anaconda3/envs/meta_drive/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets"
DATASET_NAME="metaIDM_test"
DATASET_ROOT="${OUTPUT_ROOT}/${DATASET_NAME}"
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
    --scenario-weights '{"S1_free_cruise_straight": 1.0, "S2_free_cruise_curve": 1.0, "S3_straight_following": 1.0, "S4_curve_following": 1.0, "S5_hard_brake_lead": 1.0, "S6_background_merge_in": 1.0, "S7_ego_merge_from_ramp": 1.0, "S8_ego_exit_to_ramp": 1.0, "S9_narrow_channel_negotiation": 1.0, "S10_straight_lane_change": 1.0, "S11_curve_lane_change": 1.0}' \
    --use-hybrid-map true \
    --map-block-num 5 \
    --num-scenarios 1 \
    2>&1 | tee "${LOG_PATH}"
