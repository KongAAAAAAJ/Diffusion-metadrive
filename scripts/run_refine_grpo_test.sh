#!/usr/bin/env bash
# Test the selected-refine-GRPO checkpoint configured by YAML with an execution flow strictly aligned
# with train/train_refine_grpo.py (ModeSelectionSB3Env + rollout_multimodal_refinement).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ── 路径配置 ──────────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
REFINE_TRAIN_CONFIG_PATH="${REFINE_TRAIN_CONFIG_PATH:-${REPO_ROOT}/configs/train/refine_grpo.yaml}"

# ── 公共测试函数 ──────────────────────────────────────────────────────────────
run_test() {
    echo ""
    echo "================================================================"
    echo "  Running: YAML-configured checkpoint"
    echo "  config : ${REFINE_TRAIN_CONFIG_PATH}"
    echo "================================================================"

    local cmd=(
        "${PYTHON_BIN}" models/diffusion/test_refine_grpo.py
        --refine-train-config-path      "${REFINE_TRAIN_CONFIG_PATH}"
    )
    "${cmd[@]}"
}

# ── 执行 ──────────────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"
run_test
