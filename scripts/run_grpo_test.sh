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
GRPO_CKPT="${GRPO_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/plan_cls_grpo/run_9/checkpoints/step_0040000_score_7.1248/full_platoon_grpo.ckpt}"   # e.g. .../plan_cls_grpo/run_1/checkpoints/final/full_platoon_grpo.ckpt

# RUN_TAG: set to a unique label when running multiple comparisons simultaneously,
# e.g. RUN_TAG=run9 in one terminal and RUN_TAG=run10 in another.
RUN_TAG="${RUN_TAG:-}"
OUTPUT_BASE="${OUTPUT_BASE:-/media/kong/Elements_SE/Diffusion_Data/outputs/grpo_compare}"
if [[ -n "${RUN_TAG}" ]]; then
    BASELINE_OUTPUT_DIR="${OUTPUT_BASE}/${RUN_TAG}/baseline"
    GRPO_OUTPUT_DIR="${OUTPUT_BASE}/${RUN_TAG}/grpo"
else
    BASELINE_OUTPUT_DIR="${OUTPUT_BASE}/baseline"
    GRPO_OUTPUT_DIR="${OUTPUT_BASE}/grpo"
fi

NUM_AGENTS="${NUM_AGENTS:-3}"
SCENARIO_ID="${SCENARIO_ID:-S2_free_cruise_curve}"
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
import json, pathlib, re, sys
from typing import Optional

_RUN_DIR_RE = re.compile(r"^run_(\d+)$")

_SUMMARY_FNAME = "random_action_reward_summary.json"

def _latest_complete_run_dir(root: pathlib.Path) -> Optional[pathlib.Path]:
    """Return the highest-numbered run_N/ dir that contains the summary file."""
    runs = [
        (int(m.group(1)), d)
        for d in root.iterdir()
        if d.is_dir() and (m := _RUN_DIR_RE.match(d.name))
        and (d / _SUMMARY_FNAME).exists()
    ]
    return max(runs, key=lambda x: x[0])[1] if runs else None

def load(p):
    root = pathlib.Path(p)
    # Test script creates run_N/ subdirs; fall back to root for backwards compat.
    candidate = root / _SUMMARY_FNAME
    if not candidate.exists():
        run_dir = _latest_complete_run_dir(root)
        if run_dir is None:
            sys.exit(f"[ERROR] no complete run found under: {root}")
        candidate = run_dir / _SUMMARY_FNAME
    return json.loads(candidate.read_text())

b = load("${BASELINE_OUTPUT_DIR}")
g = load("${GRPO_OUTPUT_DIR}")

episodes = int("${EPISODES}")

def pct(base, new):
    if base == 0:
        return "  n/a"
    return f"{(new - base) / abs(base) * 100:+.1f}%"

def get_rps(d):
    return float(d.get("episode_env_reward_per_step_mean", 0.0))

def get_rate(d, key):
    return float(d.get(key, 0.0))

def get_ep_len(d):
    # num_records is total steps across all episodes
    return d.get("num_records", 0) / max(1, episodes)

b_rps  = get_rps(b);                      g_rps  = get_rps(g)
b_succ = get_rate(b, "success_rate");      g_succ = get_rate(g, "success_rate")
b_cr   = get_rate(b, "crash_rate");        g_cr   = get_rate(g, "crash_rate")
b_oor  = get_rate(b, "out_of_road_rate");  g_oor  = get_rate(g, "out_of_road_rate")
b_len  = get_ep_len(b);                    g_len  = get_ep_len(g)

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
print(f"episodes={episodes}  scenario=${SCENARIO_ID}  seed=${START_SEED}")
PYEOF
