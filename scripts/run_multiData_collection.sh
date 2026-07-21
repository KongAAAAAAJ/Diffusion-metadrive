#!/usr/bin/env bash
# run_multiData_collection.sh — platoon data pipeline: collect → anchors → preprocess
# Usage: STAGE=all|collect|anchors|preprocess bash scripts/run_multiData_collection.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
DATASET_CONFIG_PATH="${DATASET_CONFIG_PATH:-${REPO_ROOT}/configs/dataset/data_collect.yaml}"
STAGE="${STAGE:-all}"

exec "${PYTHON_BIN}" -m expert_dataset.run_multi_data_pipeline \
    --config "${DATASET_CONFIG_PATH}" \
    --stage "${STAGE}"
