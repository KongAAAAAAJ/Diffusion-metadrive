#!/usr/bin/env bash
# =============================================================================
# run.sh — 两阶段训练流水线
#
# 阶段一：单车扩散 Policy 开环预训练（train_transfuser）
# 阶段二：编队意图 selector 强化训练（RLlib MAPPO）
#
# 用法：
#   bash scripts/run.sh                      # 完整流水线（阶段一 → 二）
#   STAGE=1 bash scripts/run.sh              # 仅阶段一预训练
#   STAGE=2 bash scripts/run.sh              # 仅阶段二（需预先设置 SINGLE_CKPT）
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
STAGE="${STAGE:-1}"   # 1 | 2 | both

# ---------------------------------------------------------------------------
# 阶段一：单车扩散 Policy 开环预训练
# ---------------------------------------------------------------------------
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${REPO_ROOT}/configs/diffusion/model.yaml}"
EXPERT_NAME="${EXPERT_NAME:-idm}"
DATASET_ROOT="${DATASET_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metaIDM_pp}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion}"
STAGE1_MAX_EPOCHS="${STAGE1_MAX_EPOCHS:-50}"  # *
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-16}"
STAGE1_NUM_WORKERS="${STAGE1_NUM_WORKERS:-8}"
STAGE1_LR="${STAGE1_LR:-1e-4}"
STAGE1_PRECISION="${STAGE1_PRECISION:-auto}"

# ---------------------------------------------------------------------------
# 阶段二：编队闭环强化微调
# ---------------------------------------------------------------------------
# 若已有预训练 checkpoint，设置 SINGLE_CKPT 可直接跳过阶段一
# SINGLE_CKPT="${SINGLE_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_9/checkpoints/diffusion-epoch=42.ckpt}"
SINGLE_CKPT="${SINGLE_CKPT:-}"

RL_CONFIG="${RL_CONFIG:-${REPO_ROOT}/configs/train/selector.yaml}"
RL_NUM_AGENTS="${RL_NUM_AGENTS:-3}"
RL_RENDER="${RL_RENDER:-0}"
RL_STEPS="${RL_STEPS:-20000}"  # total env steps
RL_CKPT_DIR="${RL_CKPT_DIR:-}" 
RL_LOG_DIR="${RL_LOG_DIR:-}"

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
    log_info "Epochs   ：${STAGE1_MAX_EPOCHS}，batch_size=${STAGE1_BATCH_SIZE}"

    "${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.train_transfuser \
        --model-config-path       "${MODEL_CONFIG_PATH}" \
        --dataset-root            "${DATASET_ROOT}" \
        --output-dir              "${STAGE1_OUTPUT_DIR}" \
        --max-epochs              "${STAGE1_MAX_EPOCHS}" \
        --batch-size              "${STAGE1_BATCH_SIZE}" \
        --num-workers             "${STAGE1_NUM_WORKERS}" \
        --lr                      "${STAGE1_LR}" \
        --precision               "${STAGE1_PRECISION}" \
        --check-val-every-n-epoch 1 \
        --cache-shards-in-memory

    SINGLE_CKPT="$(find_best_ckpt "${STAGE1_OUTPUT_DIR}")"
    log_ok "阶段一完成。checkpoint：${SINGLE_CKPT}"
}

# ---------------------------------------------------------------------------
# 阶段二：闭环强化微调
# ---------------------------------------------------------------------------
run_stage2() {
    log_section "阶段二：编队意图 selector 强化训练（MAPPO，${RL_STEPS} env steps）"
    log_info "单车 checkpoint ：${SINGLE_CKPT}"
    log_info "训练配置        ：${RL_CONFIG}"
    log_info "编队车辆数      ：${RL_NUM_AGENTS}"
    log_info "包含            ：冻结 planner + shared selector actor + centralized critic"

    local cmd=(
        "${PYTHON_BIN}" -m train.train_selector
        --config "${RL_CONFIG}"
        --total-env-steps "${RL_STEPS}"
        --pretrained-ckpt "${SINGLE_CKPT}"
    )
    if [[ -n "${RL_CKPT_DIR}" ]]; then
        cmd+=(--output-root "$(dirname "$(dirname "${RL_CKPT_DIR}")")")
    elif [[ -n "${RL_LOG_DIR}" ]]; then
        cmd+=(--output-root "$(dirname "$(dirname "${RL_LOG_DIR}")")")
    fi

    "${cmd[@]}"

    log_ok "阶段二完成"
    if [[ -n "${RL_CKPT_DIR}" ]]; then
        log_info "checkpoints → ${RL_CKPT_DIR}"
    else
        log_info "checkpoints → /media/kong/Elements_SE/Diffusion_Data/outputs/selector/run_x/checkpoints"
    fi
    if [[ -n "${RL_LOG_DIR}" ]]; then
        log_info "TensorBoard → tensorboard --logdir ${RL_LOG_DIR}"
    else
        log_info "TensorBoard → tensorboard --logdir /media/kong/Elements_SE/Diffusion_Data/outputs/selector/run_x/tb"
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
