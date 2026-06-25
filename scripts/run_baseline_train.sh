#!/usr/bin/env bash
# =============================================================================
# run_train.sh — 单车扩散 Policy 开环预训练
#
# 单车扩散 Policy 开环预训练（train_transfuser）
#
# 用法：
#   bash scripts/run_train.sh
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
# 单车扩散 Policy 开环预训练
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

run_pretrain() {
    log_section "单车扩散 Policy 开环预训练"

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
        "${PYTHON_BIN}" -m models.diffusion.train_transfuser
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
    else
        log_info "模式     ：fresh"
        log_info "Epochs   ：${STAGE1_MAX_EPOCHS}"
        cmd+=(--max-epochs "${STAGE1_MAX_EPOCHS}")
    fi

    "${cmd[@]}"

    if [[ -n "${STAGE1_RESUME_CKPT}" ]]; then
        local best_ckpt
        best_ckpt="$(find_best_ckpt_in_run_dir "$(dirname "$(dirname "${STAGE1_RESUME_CKPT}")")")"
    else
        local best_ckpt
        best_ckpt="$(find_best_ckpt "${STAGE1_OUTPUT_DIR}")"
    fi
    log_ok "预训练完成。checkpoint：${best_ckpt}"
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"

echo ""
log_info "========================================================"
log_info " 单车扩散 Policy 开环预训练"
log_info "========================================================"

run_pretrain

echo ""
log_ok "========================================================"
log_ok " 预训练完成"
log_ok "========================================================"
