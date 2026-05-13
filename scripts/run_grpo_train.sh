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
# SCENARIO_IDS="${SCENARIO_IDS:-S2_free_cruise_curve}"
# RESUME_GRPO_CKPT="${RESUME_GRPO_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/plan_cls_grpo/run_4/checkpoints/step_0045000_score_7.2916}"   # e.g. .../plan_cls_grpo/run_1/checkpoints/final/full_platoon_grpo.ckpt

cd "${REPO_ROOT}"

exec "${PYTHON_BIN}" -m train.train_plan_cls_grpo \
    --config "${CONFIG}" \
    --pretrained-ckpt "${PRETRAINED_CKPT}" \
    --output-root "${OUTPUT_ROOT}" \
    --total-env-steps "${TOTAL_STEPS}" \
    --num-agents "${NUM_AGENTS}" \
    --planner-device "${PLANNER_DEVICE}" \
    ${SCENARIO_IDS:+--scenario-ids "${SCENARIO_IDS}"} \
    ${RESUME_GRPO_CKPT:+--resume-grpo-ckpt "${RESUME_GRPO_CKPT}"}






# * run_8（中断）/run_9（失败）/run_10（有上升趋势）/run_11(失败)/run_12（失败）
# * 直接训S1+S2(失败)/课程学习训练S2（失败）/直接训S2（有上升趋势）/直接训S1+S2（失败）/直接训S1+S2+S3+S4（失败）
# * S1和S2都只包含1条local route

# ? Target point不对（推理时）