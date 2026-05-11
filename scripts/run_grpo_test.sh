#!/usr/bin/env bash
# Compare baseline diffusion planner vs GRPO-fine-tuned planner on the same scenarios.
# Both runs use the same seed / episodes / scenario so results are directly comparable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${REPO_ROOT}/configs/diffusion/model.yaml}"

BASELINE_CKPT="${BASELINE_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_20/checkpoints/diffusion-epoch=25.ckpt}"
GRPO_CKPT="${GRPO_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/plan_cls_grpo/run_1/checkpoints/step_0040000_score_8.8577/full_platoon_grpo.ckpt}"   # e.g. .../plan_cls_grpo/run_1/checkpoints/final/full_platoon_grpo.ckpt

OUTPUT_BASE="${OUTPUT_BASE:-/media/kong/Elements_SE/Diffusion_Data/outputs/grpo_compare}"
BASELINE_OUTPUT_DIR="${OUTPUT_BASE}/baseline"
GRPO_OUTPUT_DIR="${OUTPUT_BASE}/grpo"

NUM_AGENTS="${NUM_AGENTS:-3}"
SCENARIO_ID="${SCENARIO_ID:-S1_free_cruise_straight}"
EPISODES="${EPISODES:-3}"
START_SEED="${START_SEED:-0}"
TRAFFIC_DENSITY="${TRAFFIC_DENSITY:-0.04}"
CONTROLLER_TYPE="${CONTROLLER_TYPE:-stabilized}"
DEVICE="${DEVICE:-cuda}"

if [[ -z "${GRPO_CKPT}" ]]; then
    echo "[ERROR] GRPO_CKPT is not set. Usage:" >&2
    echo "  GRPO_CKPT=/path/to/full_platoon_grpo.ckpt bash scripts/run_grpo_compare.sh" >&2
    exit 1
fi

run_test() {
    local label="$1"
    local ckpt="$2"
    local outdir="$3"
    local use_relation_encoder="${4:-1}"   # 0: skip relation_encoder; 1: use it (default)
    echo ""
    echo "================================================================"
    echo "  Running: ${label}"
    echo "  ckpt   : ${ckpt}"
    echo "  outdir : ${outdir}"
    echo "  use_relation_encoder=${use_relation_encoder}"
    echo "================================================================"
    "${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.test_transfuser_policy \
        --checkpoint "${ckpt}" \
        --model-config-path "${MODEL_CONFIG_PATH}" \
        --num-agents "${NUM_AGENTS}" \
        --scenario-id "${SCENARIO_ID}" \
        --episodes "${EPISODES}" \
        --start-seed "${START_SEED}" \
        --traffic-density "${TRAFFIC_DENSITY}" \
        --controller-type "${CONTROLLER_TYPE}" \
        --device "${DEVICE}" \
        --ppo-actor-ckpt "" \
        --use-relation-encoder "${use_relation_encoder}" \
        --output-dir "${outdir}" \
        --save-trajectory-plot 1 \
        --save-combined-traj-frames 1 \
        --render 0
}

cd "${REPO_ROOT}"
# Baseline: disable relation_encoder so each vehicle runs the pure single-vehicle diffusion planner
run_test "BASELINE (pretrained diffusion planner)" "${BASELINE_CKPT}" "${BASELINE_OUTPUT_DIR}" 0
# GRPO ckpt was trained with relation_encoder enabled
run_test "GRPO (fine-tuned cls_branch)"            "${GRPO_CKPT}"     "${GRPO_OUTPUT_DIR}"     1

echo ""
echo "================================================================"
echo "  Comparison"
echo "================================================================"

"${PYTHON_BIN}" - <<PYEOF
import json, pathlib, sys

def load(p):
    f = pathlib.Path(p) / "random_action_reward_summary.json"
    if not f.exists():
        sys.exit(f"[ERROR] summary not found: {f}")
    return json.loads(f.read_text())

b = load("${BASELINE_OUTPUT_DIR}")
g = load("${GRPO_OUTPUT_DIR}")

n = b.get("episodes", 1)

def pct(base, new):
    if base == 0:
        return "  n/a"
    return f"{(new - base) / abs(base) * 100:+.1f}%"

def rps(d):
    rps_list = d.get("reward_per_step") or d.get("summary", {}).get("reward_per_step", [])
    return sum(rps_list) / len(rps_list) if rps_list else 0.0

def rate(d, key):
    return d.get(key, 0) / n

b_rps  = rps(b);           g_rps  = rps(g)
b_succ = rate(b,"success"); g_succ = rate(g,"success")
b_cr   = rate(b,"crash");   g_cr   = rate(g,"crash")
b_oor  = rate(b,"out_of_road"); g_oor = rate(g,"out_of_road")
b_len_list = b.get("episode_length") or b.get("summary", {}).get("episode_length", [])
g_len_list = g.get("episode_length") or g.get("summary", {}).get("episode_length", [])
b_len  = sum(b_len_list)/len(b_len_list) if b_len_list else 0
g_len  = sum(g_len_list)/len(g_len_list) if g_len_list else 0

rows = [
    ("reward_per_step",   f"{b_rps:.4f}",  f"{g_rps:.4f}",  pct(b_rps, g_rps)),
    ("success_rate",      f"{b_succ:.3f}", f"{g_succ:.3f}", pct(b_succ, g_succ)),
    ("crash_rate",        f"{b_cr:.3f}",   f"{g_cr:.3f}",   pct(b_cr, g_cr)),
    ("out_of_road_rate",  f"{b_oor:.3f}",  f"{g_oor:.3f}",  pct(b_oor, g_oor)),
    ("mean_ep_length",    f"{b_len:.1f}",  f"{g_len:.1f}",  pct(b_len, g_len)),
]

w = [22, 10, 10, 8]
fmt = "{:<{w0}}{:>{w1}}{:>{w2}}{:>{w3}}"
print(fmt.format("metric", "BASELINE", "GRPO", "delta",
                 w0=w[0], w1=w[1], w2=w[2], w3=w[3]))
print("-" * sum(w))
for row in rows:
    print(fmt.format(*row, w0=w[0], w1=w[1], w2=w[2], w3=w[3]))
print()
print(f"episodes={n}  scenario=${SCENARIO_ID}  seed=${START_SEED}")
PYEOF
