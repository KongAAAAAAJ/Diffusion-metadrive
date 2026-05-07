#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
MODE_CLS_CONFIG="${MODE_CLS_CONFIG:-${REPO_ROOT}/configs/train/ppo.yaml}"
SINGLE_CKPT="${SINGLE_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_20/checkpoints/diffusion-epoch=25.ckpt}"
MODE_CLS_OUTPUT_ROOT="${MODE_CLS_OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/outputs/mode_cls_ppo}"
PPO_POLICY_SOURCE="${PPO_POLICY_SOURCE:-pretrained}"  # ppo | pretrained
PPO_CKPT="${PPO_CKPT:-}"
PPO_RUN_DIR="${PPO_RUN_DIR:-}"
PPO_TEST_OUTPUT_DIR="${PPO_TEST_OUTPUT_DIR:-}"
PPO_TEST_EPISODES="${PPO_TEST_EPISODES:-1}"
PPO_TEST_MAX_STEPS="${PPO_TEST_MAX_STEPS:-100}"
PPO_TEST_DETERMINISTIC="${PPO_TEST_DETERMINISTIC:-1}"
SAVE_TOPDOWN_FRAMES="${SAVE_TOPDOWN_FRAMES:-1}"
SAVE_TRAJECTORY_PLOT="${SAVE_TRAJECTORY_PLOT:-1}"
SAVE_2D_VIDEO="${SAVE_2D_VIDEO:-1}"
SAVE_TERMINAL_FRAME="${SAVE_TERMINAL_FRAME:-1}"
VIDEO_FPS="${VIDEO_FPS:-10}"
MODE_CLS_PLANNER_DEVICE="${MODE_CLS_PLANNER_DEVICE:-cuda}"
RL_NUM_AGENTS="${RL_NUM_AGENTS:-1}"
RL_RENDER="${RL_RENDER:-0}"
MODE_CLS_SCENARIO_IDS="${MODE_CLS_SCENARIO_IDS:-S1_free_cruise_straight,S2_free_cruise_curve,S3_straight_following,S4_curve_following}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_section() {
    echo ""
    echo -e "${YELLOW}================================================================${NC}"
    echo -e "${YELLOW}  $*${NC}"
    echo -e "${YELLOW}================================================================${NC}"
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

if [[ "${PPO_POLICY_SOURCE}" != "ppo" && "${PPO_POLICY_SOURCE}" != "pretrained" ]]; then
    log_error "PPO_POLICY_SOURCE 只支持 ppo 或 pretrained，当前为：${PPO_POLICY_SOURCE}"
    exit 1
fi

if [[ "${PPO_POLICY_SOURCE}" == "ppo" && -z "${PPO_RUN_DIR}" ]]; then
    PPO_RUN_DIR="$(find_latest_run_dir "${MODE_CLS_OUTPUT_ROOT}" || true)"
fi
if [[ "${PPO_POLICY_SOURCE}" == "ppo" && -z "${PPO_CKPT}" && -z "${PPO_RUN_DIR}" ]]; then
    log_error "未找到 PPO_RUN_DIR，也未设置 PPO_CKPT。请先训练或显式设置 PPO_CKPT=/path/to/sb3_model.zip。"
    exit 1
fi
if [[ "${PPO_POLICY_SOURCE}" == "ppo" && -n "${PPO_CKPT}" && ! -f "${PPO_CKPT}" ]]; then
    log_error "PPO_CKPT 不存在：${PPO_CKPT}"
    exit 1
fi
if [[ "${PPO_POLICY_SOURCE}" == "ppo" && -z "${PPO_CKPT}" && ! -f "${PPO_RUN_DIR}/checkpoints/final/sb3_model.zip" ]]; then
    log_error "未找到 final PPO checkpoint：${PPO_RUN_DIR}/checkpoints/final/sb3_model.zip"
    exit 1
fi
if [[ ! -f "${SINGLE_CKPT}" ]]; then
    log_error "SINGLE_CKPT 不存在：${SINGLE_CKPT}"
    exit 1
fi

log_section "SB3 MaskablePPO 多车闭环测试"
log_info "训练配置      ：${MODE_CLS_CONFIG}"
log_info "单车 checkpoint：${SINGLE_CKPT}"
log_info "策略来源      ：${PPO_POLICY_SOURCE}"
if [[ "${PPO_POLICY_SOURCE}" == "ppo" ]]; then
    default_output_dir="${PPO_RUN_DIR}/test"
    log_info "PPO run       ：${PPO_RUN_DIR:-<explicit ckpt>}"
    log_info "PPO ckpt      ：${PPO_CKPT:-${PPO_RUN_DIR}/checkpoints/final/sb3_model.zip}"
else
    default_output_dir="${MODE_CLS_OUTPUT_ROOT}/pretrained_test"
    log_info "PPO run       ：<skip>"
    log_info "PPO ckpt      ：<skip>"
fi
log_info "输出目录      ：${PPO_TEST_OUTPUT_DIR:-${default_output_dir}}"
log_info "episodes      ：${PPO_TEST_EPISODES}"
log_info "max_steps     ：${PPO_TEST_MAX_STEPS}"
log_info "deterministic ：${PPO_TEST_DETERMINISTIC}"
log_info "save topdown  ：${SAVE_TOPDOWN_FRAMES}"
log_info "save traj plot：${SAVE_TRAJECTORY_PLOT}"
log_info "save video    ：${SAVE_2D_VIDEO}"
log_info "save terminal ：${SAVE_TERMINAL_FRAME}"
log_info "Planner device：${MODE_CLS_PLANNER_DEVICE}"
log_info "场景          ：${MODE_CLS_SCENARIO_IDS}"

cmd=(
    "${PYTHON_BIN}" -m train.test_mode_cls_sb3
    --config "${MODE_CLS_CONFIG}"
    --pretrained-ckpt "${SINGLE_CKPT}"
    --policy-source "${PPO_POLICY_SOURCE}"
    --episodes "${PPO_TEST_EPISODES}"
    --max-steps "${PPO_TEST_MAX_STEPS}"
    --deterministic "${PPO_TEST_DETERMINISTIC}"
    --save-topdown-frames "${SAVE_TOPDOWN_FRAMES}"
    --save-trajectory-plot "${SAVE_TRAJECTORY_PLOT}"
    --save-video "${SAVE_2D_VIDEO}"
    --save-terminal-frame "${SAVE_TERMINAL_FRAME}"
    --video-fps "${VIDEO_FPS}"
    --num-agents "${RL_NUM_AGENTS}"
    --use-render "${RL_RENDER}"
    --planner-device "${MODE_CLS_PLANNER_DEVICE}"
    --scenario-ids "${MODE_CLS_SCENARIO_IDS}"
)
if [[ "${PPO_POLICY_SOURCE}" == "ppo" ]]; then
    if [[ -n "${PPO_CKPT}" ]]; then
        cmd+=(--ppo-ckpt "${PPO_CKPT}")
    else
        cmd+=(--ppo-run-dir "${PPO_RUN_DIR}")
    fi
fi
if [[ -n "${PPO_TEST_OUTPUT_DIR}" ]]; then
    cmd+=(--output-dir "${PPO_TEST_OUTPUT_DIR}")
fi

set +e
"${cmd[@]}"
status=$?
set -e
if [[ "${status}" -eq 139 ]]; then
    output_dir="${PPO_TEST_OUTPUT_DIR:-${default_output_dir}}"
    if [[ -f "${output_dir}/test_summary.json" ]] && [[ -f "${output_dir}/ppo_test_debug.jsonl" ]]; then
        log_info "测试进程以 139 退出；检测到测试结果已落盘，按 MetaDrive/Panda3D 已知退出段错误处理。"
    else
        log_error "测试以 139 退出，但未检测到完整测试结果。"
        exit 139
    fi
elif [[ "${status}" -ne 0 ]]; then
    log_error "PPO 闭环测试失败，退出码：${status}"
    exit "${status}"
fi

log_ok "PPO 闭环测试完成"
