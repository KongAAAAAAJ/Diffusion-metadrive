#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"

# PRETRAINED_CKPT="${PRETRAINED_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_20/checkpoints/diffusion-epoch=25.ckpt}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/train/plan_cls_grpo.yaml}"
# OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/outputs/plan_cls_grpo}"
# TOTAL_STEPS="${TOTAL_STEPS:-200000}"
# NUM_AGENTS="${NUM_AGENTS:-3}"
# PLANNER_DEVICE="${PLANNER_DEVICE:-cuda}"
# SCENARIO_IDS="${SCENARIO_IDS:-S2_free_cruise_curve}"
# RESUME_GRPO_CKPT="${RESUME_GRPO_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/plan_cls_grpo/run_4/checkpoints/step_0045000_score_7.2916}"   # e.g. .../plan_cls_grpo/run_1/checkpoints/final/full_platoon_grpo.ckpt

cd "${REPO_ROOT}"

cmd=("${PYTHON_BIN}" -m train.train_plan_cls_grpo --config "${CONFIG}")
if [[ -n "${PRETRAINED_CKPT:-}" ]]; then
    cmd+=(--pretrained-ckpt "${PRETRAINED_CKPT}")
fi
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
    cmd+=(--output-root "${OUTPUT_ROOT}")
fi
if [[ -n "${TOTAL_STEPS:-}" ]]; then
    cmd+=(--total-env-steps "${TOTAL_STEPS}")
fi
if [[ -n "${NUM_AGENTS:-}" ]]; then
    cmd+=(--num-agents "${NUM_AGENTS}")
fi
if [[ -n "${PLANNER_DEVICE:-}" ]]; then
    cmd+=(--planner-device "${PLANNER_DEVICE}")
fi
if [[ -n "${SCENARIO_IDS:-}" ]]; then
    cmd+=(--scenario-ids "${SCENARIO_IDS}")
fi
if [[ -n "${RESUME_GRPO_CKPT:-}" ]]; then
    cmd+=(--resume-grpo-ckpt "${RESUME_GRPO_CKPT}")
fi

exec "${cmd[@]}"




# * run_4: S1场景训练（收敛）

# * run_8（中断）/run_9（失败）/run_10（有上升趋势）/run_11(失败)/run_12（失败）
# * 直接训S1+S2(失败)/课程学习训练S2（失败）/直接训S2（有上升趋势）/直接训S1+S2（失败）/直接训S1+S2+S3+S4（失败）
# * S1和S2都只包含1条local route

# ? Target point不对（推理时），更新语义决策偏好Preference point引导 --> reward平均水平从 6 提升至 10 左右,但是reward方差大,收敛特性不好
# * run_13: S1
# * run_14: S1+S2
# * run_15: S1+S2+S3+S4 

# ? 收敛的不好,奖励曲线方差大
# * 添加old log ratio clip限幅后：
# * run_16: S1    5万
# * run_17: S1+S2    10万


# !!!!!!
# * run_18: S1+S2    10万    use_mode_valid_mask:false
# * run_19: S1+S2    10万    use_mode_valid_mask:false, beta_kl=0.1
