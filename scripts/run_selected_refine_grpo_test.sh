#!/usr/bin/env bash
# Test a selected-refine-GRPO checkpoint with an execution flow strictly aligned
# with train/train_selected_refine_grpo.py (ModeSelectionSB3Env + rollout_multimodal_refinement).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ── 路径配置 ──────────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
REFINE_TRAIN_CONFIG_PATH="${REFINE_TRAIN_CONFIG_PATH:-${REPO_ROOT}/configs/train/selected_refine_grpo.yaml}"

# 单车 baseline / 微调前
BASELINE_CKPT="${BASELINE_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_20/checkpoints/diffusion-epoch=25.ckpt}"

# 分类头 / 轨迹头 GRPO
GRPO_CKPT="${GRPO_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/selected_refine_grpo/run_21/checkpoints/step_0045000_score_0.7949/full_platoon_refine_grpo.ckpt}"

if [[ -z "${GRPO_CKPT}" ]]; then
    echo "[ERROR] GRPO_CKPT is not set." >&2; exit 1
fi

# ── 输出目录 ──────────────────────────────────────────────────────────────────
RUN_TAG="${RUN_TAG:-0523}"
OUTPUT_BASE="${OUTPUT_BASE:-/media/kong/Elements_SE/Diffusion_Data/outputs/selected_refine_grpo_compare}"
BASELINE_OUTPUT_DIR="${OUTPUT_BASE}/${RUN_TAG}/baseline"
GRPO_OUTPUT_DIR="${OUTPUT_BASE}/${RUN_TAG}/grpo"

# ── 场景 / 运行参数 ───────────────────────────────────────────────────────────
SCENARIO_ID="${SCENARIO_ID:-S1_free_cruise_straight}"
LOCAL_ROUTE="${LOCAL_ROUTE:-}"
EPISODES="${EPISODES:-3}"
START_SEED="${START_SEED:-10}"
NUM_SCENARIOS="${EPISODES}"
TRAFFIC_DENSITY="${TRAFFIC_DENSITY:-0.04}"
RANDOM_TRAFFIC="${RANDOM_TRAFFIC:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
DEVICE="${DEVICE:-cuda}"

# RUN_MODE: "both" | "baseline" | "grpo"
RUN_MODE="${RUN_MODE:-both}"

# ── 公共测试函数 ──────────────────────────────────────────────────────────────
run_test() {
    local label="$1" ckpt="$2" outdir="$3"
    echo ""
    echo "================================================================"
    echo "  Running: ${label}"
    echo "  ckpt   : ${ckpt}"
    echo "  outdir : ${outdir}"
    echo "  scenario=${SCENARIO_ID} local_route=${LOCAL_ROUTE:-<auto>} start_seed=${START_SEED}"
    echo "  episodes=${EPISODES} traffic_density=${TRAFFIC_DENSITY} max_steps=${MAX_STEPS}"
    echo "================================================================"

    local cmd=(
        "${PYTHON_BIN}" metadrive/policy/diffusion_policy/test_selected_refine_grpo.py
        --checkpoint                    "${ckpt}"
        --refine-train-config-path      "${REFINE_TRAIN_CONFIG_PATH}"
        --scenario-id                   "${SCENARIO_ID}"
        --episodes                      "${EPISODES}"
        --start-seed                    "${START_SEED}"
        --num-scenarios                 "${NUM_SCENARIOS}"
        --traffic-density               "${TRAFFIC_DENSITY}"
        --random-traffic                "${RANDOM_TRAFFIC}"
        --max-steps                     "${MAX_STEPS}"
        --device                        "${DEVICE}"
        --output-dir                    "${outdir}"
        --save-combined-traj-frames     "${SAVE_COMBINED_TRAJ_FRAMES:-1}"
        --save-traj-plots               "${SAVE_TRAJ_PLOTS:-1}"
        --save-trajectory-data          "${SAVE_TRAJECTORY_DATA:-1}"
        --render                        0
    )
    [[ -n "${LOCAL_ROUTE}" ]] && cmd+=(--local-route "${LOCAL_ROUTE}")
    "${cmd[@]}"
}

# ── 执行 ──────────────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"

case "${RUN_MODE}" in
    both)
        run_test "BASELINE (pretrained)"  "${BASELINE_CKPT}" "${BASELINE_OUTPUT_DIR}"
        run_test "GRPO    (fine-tuned)"   "${GRPO_CKPT}"     "${GRPO_OUTPUT_DIR}"
        ;;
    baseline)
        run_test "BASELINE (pretrained)"  "${BASELINE_CKPT}" "${BASELINE_OUTPUT_DIR}"
        ;;
    grpo)
        run_test "GRPO    (fine-tuned)"   "${GRPO_CKPT}"     "${GRPO_OUTPUT_DIR}"
        ;;
    *)
        echo "[ERROR] Unknown RUN_MODE='${RUN_MODE}'. Use: both | baseline | grpo" >&2; exit 1
        ;;
esac

# ── 汇总对比（仅 both 模式）────────────────────────────────────────────────────
[[ "${RUN_MODE}" != "both" ]] && exit 0

echo ""
echo "================================================================"
echo "  Comparison"
echo "================================================================"

"${PYTHON_BIN}" - <<PYEOF
import json, pathlib, re, sys

_SUMMARY_FNAME = "random_action_reward_summary.json"
_RUN_DIR_RE = re.compile(r"^run_(\d+)$")

def load(p):
    root = pathlib.Path(p)
    if (root / _SUMMARY_FNAME).exists():
        return json.loads((root / _SUMMARY_FNAME).read_text())
    runs = [
        (int(m.group(1)), d) for d in root.iterdir()
        if d.is_dir() and (m := _RUN_DIR_RE.match(d.name)) and (d / _SUMMARY_FNAME).exists()
    ]
    if not runs:
        sys.exit(f"[ERROR] no complete run found under: {root}")
    return json.loads((max(runs)[1] / _SUMMARY_FNAME).read_text())

b = load("${BASELINE_OUTPUT_DIR}")
g = load("${GRPO_OUTPUT_DIR}")
episodes = int("${EPISODES}")

def pct(base, new):
    return "  n/a" if base == 0 else f"{(new - base) / abs(base) * 100:+.1f}%"

b_pdms = float(b.get("episode_pdms_reward_per_step_mean", 0)); g_pdms = float(g.get("episode_pdms_reward_per_step_mean", 0))
b_succ = float(b.get("success_rate", 0));                      g_succ = float(g.get("success_rate", 0))
b_cr   = float(b.get("crash_rate", 0));                        g_cr   = float(g.get("crash_rate", 0))
b_oor  = float(b.get("out_of_road_rate", 0));                  g_oor  = float(g.get("out_of_road_rate", 0))
b_len  = b.get("num_records", 0) / max(1, episodes);           g_len  = g.get("num_records", 0) / max(1, episodes)

w = [24, 10, 10, 8]
fmt = "{:<{w0}}{:>{w1}}{:>{w2}}{:>{w3}}"
print(fmt.format("metric", "BASELINE", "GRPO", "delta", w0=w[0], w1=w[1], w2=w[2], w3=w[3]))
print("-" * sum(w))
for row in [
    ("pdms_reward_per_step",  f"{b_pdms:.4f}", f"{g_pdms:.4f}", pct(b_pdms, g_pdms)),
    ("success_rate",          f"{b_succ:.3f}", f"{g_succ:.3f}", pct(b_succ, g_succ)),
    ("crash_rate",            f"{b_cr:.3f}",   f"{g_cr:.3f}",   pct(b_cr,   g_cr)),
    ("out_of_road_rate",      f"{b_oor:.3f}",  f"{g_oor:.3f}",  pct(b_oor,  g_oor)),
    ("mean_ep_length",        f"{b_len:.1f}",  f"{g_len:.1f}",  pct(b_len,  g_len)),
]:
    print(fmt.format(*row, w0=w[0], w1=w[1], w2=w[2], w3=w[3]))
print()
print(f"episodes=${EPISODES}  scenario=${SCENARIO_ID}  local_route=${LOCAL_ROUTE:-<auto>}  seed=${START_SEED}")
PYEOF
