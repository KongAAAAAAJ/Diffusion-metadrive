#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"

PRETRAINED_CKPT="${PRETRAINED_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_20/checkpoints/diffusion-epoch=25.ckpt}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/train/plan_cls_grpo.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/outputs/plan_cls_grpo}"
TOTAL_STEPS="${TOTAL_STEPS:-50000}"
NUM_AGENTS="${NUM_AGENTS:-3}"
PLANNER_DEVICE="${PLANNER_DEVICE:-cuda}"
SCENARIO_IDS="${SCENARIO_IDS:-}"

cd "${REPO_ROOT}"

exec "${PYTHON_BIN}" -m train.train_plan_cls_grpo \
    --config "${CONFIG}" \
    --pretrained-ckpt "${PRETRAINED_CKPT}" \
    --output-root "${OUTPUT_ROOT}" \
    --total-env-steps "${TOTAL_STEPS}" \
    --num-agents "${NUM_AGENTS}" \
    --planner-device "${PLANNER_DEVICE}" \
    ${SCENARIO_IDS:+--scenario-ids "${SCENARIO_IDS}"}
