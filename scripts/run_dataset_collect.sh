#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET_SAMPLES="${TARGET_SAMPLES:-20000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets}"
EXPERT_TYPE="${EXPERT_TYPE:-ppo}"
START_SEED="${START_SEED:-10}"
DATASET_NAME="${DATASET_NAME:-metadrive_${EXPERT_TYPE}}"

TRAJECTORY_CORRECTION_ENABLED="${TRAJECTORY_CORRECTION_ENABLED:-1}"  # 开启轨迹修正
SAVE_RAW_TRAJECTORY="${SAVE_RAW_TRAJECTORY:-1}"  # 保存原始轨迹
MODE_CLASSIFIER_VERSION="${MODE_CLASSIFIER_VERSION:-v1}"  
CENTERLINE_ATTRACTION_STRENGTH="${CENTERLINE_ATTRACTION_STRENGTH:-0.85}"
SMOOTHING_STRENGTH="${SMOOTHING_STRENGTH:-0.25}"

TRAJECTORY_VISUALIZATION_ENABLED="${TRAJECTORY_VISUALIZATION_ENABLED:-0}"  # 开启/关闭轨迹可视化
TRAJECTORY_VISUALIZATION_FRONT_MARGIN="${TRAJECTORY_VISUALIZATION_FRONT_MARGIN:-25}"  # 前向可视化范围（单位：米）
TRAJECTORY_VISUALIZATION_LATERAL_MARGIN="${TRAJECTORY_VISUALIZATION_LATERAL_MARGIN:-10}"  # 横向可视化范围（单位：米）

"${PYTHON_BIN}" -m metadrive.exp_dataset.collect_expert \
    --target-samples "${TARGET_SAMPLES}" \
    --output-root "${OUTPUT_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --expert-type "${EXPERT_TYPE}" \
    --start-seed "${START_SEED}" \
    --trajectory-correction-enabled "${TRAJECTORY_CORRECTION_ENABLED}" \
    --save-raw-trajectory "${SAVE_RAW_TRAJECTORY}" \
    --mode-classifier-version "${MODE_CLASSIFIER_VERSION}" \
    --centerline-attraction-strength "${CENTERLINE_ATTRACTION_STRENGTH}" \
    --smoothing-strength "${SMOOTHING_STRENGTH}" \
    --trajectory-visualization-enabled "${TRAJECTORY_VISUALIZATION_ENABLED}" \
    --trajectory-visualization-front-margin "${TRAJECTORY_VISUALIZATION_FRONT_MARGIN}" \
    --trajectory-visualization-lateral-margin "${TRAJECTORY_VISUALIZATION_LATERAL_MARGIN}"
