#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/train/bev_diffusion_stage1_v2.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/outputs/bev_diffusion_stage1}"

ARGS=(
  --config "${CONFIG}"
  --output-root "${OUTPUT_ROOT}"
  --run-mode "${RUN_MODE:-formal}"
)

if [[ -n "${VARIANT:-}" ]]; then
  ARGS+=(--variant "${VARIANT}")
fi
if [[ -n "${DATASET_ROOT:-}" ]]; then
  ARGS+=(--dataset-root "${DATASET_ROOT}")
fi
if [[ -n "${DEVICE:-}" ]]; then
  ARGS+=(--device "${DEVICE}")
fi
if [[ -n "${MAX_OPTIMIZER_STEPS:-}" ]]; then
  ARGS+=(--max-optimizer-steps "${MAX_OPTIMIZER_STEPS}")
fi
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
exec "${PYTHON_BIN}" -m train.train_bev_diffusion_stage1 "${ARGS[@]}"
