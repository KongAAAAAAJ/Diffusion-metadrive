#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"

PRETRAINED_CKPT="${PRETRAINED_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_20/checkpoints/diffusion-epoch=25.ckpt}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/train/selected_refine_grpo.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/outputs/selected_refine_grpo}"
TOTAL_STEPS="${TOTAL_STEPS:-100000}"
NUM_AGENTS="${NUM_AGENTS:-3}"
PLANNER_DEVICE="${PLANNER_DEVICE:-cuda}"
PLAN_CLS_GRPO_RUN_DIR="${PLAN_CLS_GRPO_RUN_DIR:-}"
CLS_GRPO_CKPT_DIR="${CLS_GRPO_CKPT_DIR:-}"
CLS_GRPO_FULL_CKPT="${CLS_GRPO_FULL_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/plan_cls_grpo/run_14/checkpoints/step_0025000_score_8.5445/full_platoon_grpo.ckpt}"   # e.g. .../plan_cls_grpo/run_1/checkpoints/final/full_platoon_grpo.ckpt

if [[ -z "${CLS_GRPO_CKPT_DIR}" && -z "${CLS_GRPO_FULL_CKPT}" && -n "${PLAN_CLS_GRPO_RUN_DIR}" ]]; then
    CLS_GRPO_CKPT_DIR="${PLAN_CLS_GRPO_RUN_DIR%/}/checkpoints/final"
fi

echo "[refine-grpo] single pretrained ckpt : ${PRETRAINED_CKPT}"
if [[ -n "${CLS_GRPO_CKPT_DIR}" ]]; then
    echo "[refine-grpo] cls GRPO ckpt dir   : ${CLS_GRPO_CKPT_DIR}"
elif [[ -n "${CLS_GRPO_FULL_CKPT}" ]]; then
    echo "[refine-grpo] cls GRPO full ckpt  : ${CLS_GRPO_FULL_CKPT}"
else
    echo "[refine-grpo] WARNING: no cls GRPO weights provided; refinement will use the classifier from PRETRAINED_CKPT."
fi

cd "${REPO_ROOT}"

cmd=(
    "${PYTHON_BIN}" -m train.train_selected_refine_grpo
    --config "${CONFIG}" \
    --pretrained-ckpt "${PRETRAINED_CKPT}" \
    --output-root "${OUTPUT_ROOT}" \
    --total-env-steps "${TOTAL_STEPS}" \
    --num-agents "${NUM_AGENTS}" \
    --planner-device "${PLANNER_DEVICE}"
)

if [[ -n "${CLS_GRPO_CKPT_DIR}" ]]; then
    cmd+=(--cls-grpo-ckpt-dir "${CLS_GRPO_CKPT_DIR}")
fi

if [[ -n "${CLS_GRPO_FULL_CKPT}" ]]; then
    cmd+=(--cls-grpo-full-ckpt "${CLS_GRPO_FULL_CKPT}")
fi

if [[ -n "${SCENARIO_IDS:-}" ]]; then
    cmd+=(--scenario-ids "${SCENARIO_IDS}")
fi

exec "${cmd[@]}"



# !!!!！
# * run_1: 在分类头run_13的基础上继续训练GRPO轨迹头（S1）    5万
# * run_2: 在分类头run_14的基础上继续训练GRPO轨迹头（S1+S2）    10万
# * run_3: 在分类头run_15的基础上继续训练GRPO轨迹头（S1+S2+S3+S4）    20万
