#!/usr/bin/env bash
# =============================================================================
# run.sh — 两阶段训练流水线
#
# 阶段一：单车扩散 Policy 开环预训练（train_transfuser）
# 阶段二：CTDE mode-classification PPO 闭环微调
#
# 用法：
#   bash scripts/run_train.sh                # 完整流水线（阶段一 → 二）
#   STAGE=1 bash scripts/run_train.sh        # 仅阶段一预训练
#   STAGE=2 bash scripts/run_train.sh        # 仅阶段二（需预先设置 SINGLE_CKPT）
#
# 所有参数均可通过环境变量覆盖，无需修改本脚本。
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 路径与环境
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"

# ---------------------------------------------------------------------------
# 流程控制
# ---------------------------------------------------------------------------
STAGE="${STAGE:-2}"   # 1 | 2 | both

# ---------------------------------------------------------------------------
# 阶段一：单车扩散 Policy 开环预训练
# ---------------------------------------------------------------------------
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${REPO_ROOT}/configs/diffusion/model.yaml}"
EXPERT_NAME="${EXPERT_NAME:-idm}"
DATASET_ROOT="${DATASET_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metaIDM_pp}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion}"
STAGE1_MAX_EPOCHS="${STAGE1_MAX_EPOCHS:-40}"  # *
STAGE1_RESUME_CKPT="${STAGE1_RESUME_CKPT:-}"  # 若设置了预训练 checkpoint，则进入续训模式
STAGE1_RESUME_EPOCHS="${STAGE1_RESUME_EPOCHS:-10}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-16}"
STAGE1_NUM_WORKERS="${STAGE1_NUM_WORKERS:-8}"
STAGE1_LR="${STAGE1_LR:-1e-4}"
STAGE1_PRECISION="${STAGE1_PRECISION:-auto}"

# ---------------------------------------------------------------------------
# 阶段二：SB3 MaskablePPO mode-classification + 编队闭环微调
# ---------------------------------------------------------------------------
# 若已有预训练 checkpoint，设置 SINGLE_CKPT 可直接跳过阶段一
SINGLE_CKPT="${SINGLE_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_20/checkpoints/diffusion-epoch=25.ckpt}"
# SINGLE_CKPT="${SINGLE_CKPT:-}"

MODE_CLS_CONFIG="${MODE_CLS_CONFIG:-${REPO_ROOT}/configs/train/mode_cls_ppo.yaml}"
MODE_CLS_OUTPUT_ROOT="${MODE_CLS_OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/outputs/mode_cls_ppo}"
RL_NUM_AGENTS="${RL_NUM_AGENTS:-3}"
RL_RENDER="${RL_RENDER:-0}"
RL_STEPS="${RL_STEPS:-200000}"  # SB3 total env steps
MODE_CLS_PLANNER_DEVICE="${MODE_CLS_PLANNER_DEVICE:-cuda}"
# 逗号分隔的场景 ID；第一版默认限定 S1~S4
MODE_CLS_SCENARIO_IDS="${MODE_CLS_SCENARIO_IDS:-S1_free_cruise_straight,S2_free_cruise_curve,S3_straight_following,S4_curve_following}"

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_section() {
    echo ""
    echo -e "${YELLOW}================================================================${NC}"
    echo -e "${YELLOW}  $*${NC}"
    echo -e "${YELLOW}================================================================${NC}"
}

# 阶段一完成后，自动定位最新 run 目录中 epoch 最大的 checkpoint
find_best_ckpt() {
    local output_dir="$1"
    local latest_run
    latest_run="$(ls -d "${output_dir}"/run_* 2>/dev/null | sort -V | tail -1)"
    if [[ -z "${latest_run}" ]]; then
        log_error "未在 ${output_dir} 中找到任何 run 目录"; return 1
    fi
    local best_ckpt
    best_ckpt="$(ls "${latest_run}/checkpoints"/diffusion-epoch=*.ckpt 2>/dev/null | sort -V | tail -1)"
    if [[ -z "${best_ckpt}" ]]; then
        log_error "未在 ${latest_run}/checkpoints 中找到 checkpoint"; return 1
    fi
    echo "${best_ckpt}"
}

find_best_ckpt_in_run_dir() {
    local run_dir="$1"
    if [[ ! -d "${run_dir}" ]]; then
        log_error "run 目录不存在：${run_dir}"; return 1
    fi
    local best_ckpt
    best_ckpt="$(ls "${run_dir}/checkpoints"/diffusion-epoch=*.ckpt 2>/dev/null | sort -V | tail -1)"
    if [[ -z "${best_ckpt}" ]]; then
        log_error "未在 ${run_dir}/checkpoints 中找到 checkpoint"; return 1
    fi
    echo "${best_ckpt}"
}

find_latest_run_dir() {
    local output_dir="$1"
    local latest_run
    latest_run="$(ls -d "${output_dir}"/run_* 2>/dev/null | sort -V | tail -1)"
    if [[ -z "${latest_run}" ]]; then
        return 1
    fi
    echo "${latest_run}"
}

has_mode_cls_final_outputs() {
    local run_dir="$1"
    [[ -f "${run_dir}/summary.json" ]] \
        && [[ -f "${run_dir}/checkpoints/final/sb3_model.zip" ]] \
        && [[ -f "${run_dir}/checkpoints/final/plan_cls_branch_delta.pt" ]]
}

# ---------------------------------------------------------------------------
# 阶段一：开环预训练
# ---------------------------------------------------------------------------
run_stage1() {
    log_section "阶段一：单车扩散 Policy 开环预训练"

    if [[ ! -d "${DATASET_ROOT}" ]]; then
        log_error "数据集目录不存在：${DATASET_ROOT}"
        log_error "请先运行 bash scripts/run_dataset_collect.sh 采集数据"
        exit 1
    fi

    log_info "数据集   ：${DATASET_ROOT}"
    log_info "模型配置 ：${MODEL_CONFIG_PATH}"
    log_info "输出目录 ：${STAGE1_OUTPUT_DIR}"
    log_info "batch_size：${STAGE1_BATCH_SIZE}"

    local cmd=(
        "${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.train_transfuser
        --model-config-path "${MODEL_CONFIG_PATH}"
        --dataset-root "${DATASET_ROOT}"
        --output-dir "${STAGE1_OUTPUT_DIR}"
        --batch-size "${STAGE1_BATCH_SIZE}"
        --num-workers "${STAGE1_NUM_WORKERS}"
        --lr "${STAGE1_LR}"
        --precision "${STAGE1_PRECISION}"
        --check-val-every-n-epoch 1
        --cache-shards-in-memory
    )

    if [[ -n "${STAGE1_RESUME_CKPT}" ]]; then
        if [[ ! -f "${STAGE1_RESUME_CKPT}" ]]; then
            log_error "STAGE1_RESUME_CKPT 文件不存在：${STAGE1_RESUME_CKPT}"
            exit 1
        fi
        log_info "模式     ：resume"
        log_info "恢复 ckpt：${STAGE1_RESUME_CKPT}"
        log_info "追加 epochs：${STAGE1_RESUME_EPOCHS}"
        cmd+=(
            --resume-from-checkpoint "${STAGE1_RESUME_CKPT}"
            --resume-additional-epochs "${STAGE1_RESUME_EPOCHS}"
        )
        SINGLE_CKPT="${STAGE1_RESUME_CKPT}"
    else
        log_info "模式     ：fresh"
        log_info "Epochs   ：${STAGE1_MAX_EPOCHS}"
        cmd+=(--max-epochs "${STAGE1_MAX_EPOCHS}")
    fi

    "${cmd[@]}"

    if [[ -n "${STAGE1_RESUME_CKPT}" ]]; then
        SINGLE_CKPT="$(find_best_ckpt_in_run_dir "$(dirname "$(dirname "${STAGE1_RESUME_CKPT}")")")"
    else
        SINGLE_CKPT="$(find_best_ckpt "${STAGE1_OUTPUT_DIR}")"
    fi
    log_ok "阶段一完成。checkpoint：${SINGLE_CKPT}"
}

# ---------------------------------------------------------------------------
# 阶段二：闭环强化微调
# ---------------------------------------------------------------------------
run_stage2() {
    log_section "阶段二：SB3 MaskablePPO mode classification 训练"
    log_info "单车 checkpoint ：${SINGLE_CKPT}"
    log_info "训练配置        ：${MODE_CLS_CONFIG}"
    log_info "输出目录        ：${MODE_CLS_OUTPUT_ROOT}"
    log_info "编队车辆数      ：${RL_NUM_AGENTS}"
    log_info "渲染            ：${RL_RENDER}"
    log_info "Planner device  ：${MODE_CLS_PLANNER_DEVICE}"
    log_info "训练步数        ：${RL_STEPS}"
    log_info "场景            ：${MODE_CLS_SCENARIO_IDS}"
    log_info "包含            ：冻结候选轨迹生成 + SB3 MaskablePPO 微调 plan_cls_branch"

    local cmd=(
        "${PYTHON_BIN}" -m train.train_mode_cls_sb3
        --config "${MODE_CLS_CONFIG}"
        --total-env-steps "${RL_STEPS}"
        --pretrained-ckpt "${SINGLE_CKPT}"
        --output-root "${MODE_CLS_OUTPUT_ROOT}"
        --num-agents "${RL_NUM_AGENTS}"
        --use-render "${RL_RENDER}"
        --planner-device "${MODE_CLS_PLANNER_DEVICE}"
        --scenario-ids "${MODE_CLS_SCENARIO_IDS}"
    )
    local latest_run=""

    set +e
    "${cmd[@]}"
    local status=$?
    set -e
    if [[ "${status}" -eq 139 ]]; then
        local latest_run
        latest_run="$(find_latest_run_dir "${MODE_CLS_OUTPUT_ROOT}" || true)"
        if [[ -n "${latest_run}" ]] && has_mode_cls_final_outputs "${latest_run}"; then
            log_info "阶段二进程以 139 退出；检测到 final checkpoint 已落盘，按 MetaDrive/Panda3D 已知退出段错误处理。"
            log_info "实际输出目录：${latest_run}"
        else
            log_error "阶段二以 139 退出，但未检测到完整 final checkpoint；这不是可忽略的 Panda3D 退出问题。"
            exit 139
        fi
    elif [[ "${status}" -ne 0 ]]; then
        log_error "阶段二训练失败，退出码：${status}"
            exit "${status}"
    fi
    latest_run="$(find_latest_run_dir "${MODE_CLS_OUTPUT_ROOT}" || true)"

    log_ok "阶段二完成"
    if [[ -n "${latest_run}" ]]; then
        log_info "输出      → ${latest_run}"
        log_info "checkpoints → ${latest_run}/checkpoints/{step_*,final}"
        log_info "debug log   → ${latest_run}/mode_selection_debug.jsonl"
    else
        log_info "输出      → ${MODE_CLS_OUTPUT_ROOT}/run_x"
        log_info "checkpoints → ${MODE_CLS_OUTPUT_ROOT}/run_x/checkpoints/{step_*,final}"
        log_info "debug log   → ${MODE_CLS_OUTPUT_ROOT}/run_x/mode_selection_debug.jsonl"
    fi
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"

echo ""
log_info "========================================================"
log_info " 编队扩散 Policy 两阶段训练  STAGE=${STAGE}"
log_info "========================================================"

if [[ "${STAGE}" == "1" || "${STAGE}" == "both" ]]; then
    run_stage1
fi

if [[ "${STAGE}" == "2" || "${STAGE}" == "both" ]]; then
    if [[ -z "${SINGLE_CKPT}" ]]; then
        log_error "SINGLE_CKPT 未设置，无法启动阶段二"
        log_error "请设置：SINGLE_CKPT=/path/to/diffusion-epoch=XX.ckpt bash scripts/run.sh"
        exit 1
    fi
    if [[ ! -f "${SINGLE_CKPT}" ]]; then
        log_error "SINGLE_CKPT 文件不存在：${SINGLE_CKPT}"; exit 1
    fi
    run_stage2
fi

echo ""
log_ok "========================================================"
log_ok " 两阶段训练全部完成"
log_ok "========================================================"
