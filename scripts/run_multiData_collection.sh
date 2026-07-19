#!/usr/bin/env bash
# run_multiData_collection.sh — platoon data pipeline: collect → anchors → preprocess
# Usage: STAGE=all|collect|anchors|preprocess [VAR=val ...] bash scripts/run_multiData_collection.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${REPO_ROOT}/configs/diffusion/model.yaml}"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-${REPO_ROOT}/configs/train/refine_grpo.yaml}"

resolve_model_config_anchor_method() {
    "${PYTHON_BIN}" - "${MODEL_CONFIG_PATH}" <<'PY'
import sys
from models.diffusion.transfuser_config import load_diffusion_model_config

config = load_diffusion_model_config(sys.argv[1])
print(config.get("anchor_method", "dynamic"))
PY
}

# ── Stage selection and shared paths ────────────────────────────────────────
STAGE="${STAGE:-all}"  # all | collect | anchors | preprocess
ANCHOR_METHOD="${ANCHOR_METHOD:-$(resolve_model_config_anchor_method)}"  # dynamic | k_means
DATASET_NAME="${DATASET_NAME:-platoon_rule_expert}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets}"

COLLECT_OUTPUT="${OUTPUT_ROOT}/${DATASET_NAME}"
PREPROCESS_OUTPUT="${PREPROCESS_OUTPUT:-${COLLECT_OUTPUT}_pp}"
ANCHORS_OUTPUT="${ANCHORS_OUTPUT:-${REPO_ROOT}/expert_dataset/platoon_anchors.npy}"
ANCHORS_FIGURE="${ANCHORS_FIGURE:-${REPO_ROOT}/expert_dataset/platoon_anchors.png}"

# ── Multi-agent collection parameters ──────────────────────────────────────
TARGET_SAMPLES="${TARGET_SAMPLES:-40000}"
START_SEED="${START_SEED:-10}"
MAX_EPISODES="${MAX_EPISODES:-0}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-0}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-2048}"
RESUME="${RESUME:-1}"
NUM_AGENTS="${NUM_AGENTS:-3}"
SCENARIO_WEIGHTS="${SCENARIO_WEIGHTS:-}"
if [[ -z "${SCENARIO_WEIGHTS}" ]]; then
    SCENARIO_WEIGHTS='{"S5_hard_brake_lead":1.0,"S6_background_merge_in":1.0}'
fi
TRAJECTORY_VISUALIZATION_ENABLED="${TRAJECTORY_VISUALIZATION_ENABLED:-0}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"

# ── Anchor parameters ───────────────────────────────────────────────────────
TRAJECTORY_KEY="${TRAJECTORY_KEY:-trajectory}"
NUM_ANCHORS="${NUM_ANCHORS:-20}"
ANCHOR_SEED="${ANCHOR_SEED:-0}"
ANCHOR_SPLIT="${ANCHOR_SPLIT:-all}"
ANCHOR_MAX_TRAJECTORIES="${ANCHOR_MAX_TRAJECTORIES:-100000}"

# ── Preprocess parameters ───────────────────────────────────────────────────
OUTPUT_FORMAT="${OUTPUT_FORMAT:-dir}"

run_collect() {
    local log_path="${OUTPUT_ROOT}/collect_multi_command_${DATASET_NAME}.log"
    echo "=== [1/3] Multi-agent Expert Collection ==="
    mkdir -p "${OUTPUT_ROOT}"

    "${PYTHON_BIN}" -m expert_dataset.collect_multi_experts \
        --output-root "${OUTPUT_ROOT}" \
        --dataset-name "${DATASET_NAME}" \
        --model-config-path "${MODEL_CONFIG_PATH}" \
        --train-config-path "${TRAIN_CONFIG_PATH}" \
        --target-samples "${TARGET_SAMPLES}" \
        --start-seed "${START_SEED}" \
        --max-episodes "${MAX_EPISODES}" \
        --max-episode-steps "${MAX_EPISODE_STEPS}" \
        --samples-per-shard "${SAMPLES_PER_SHARD}" \
        --resume "${RESUME}" \
        --num-agents "${NUM_AGENTS}" \
        --scenario-weights "${SCENARIO_WEIGHTS}" \
        --trajectory-visualization-enabled "${TRAJECTORY_VISUALIZATION_ENABLED}" \
        --save-videos "${SAVE_VIDEOS}" \
        2>&1 | tee -a "${log_path}"

    echo "=== [1/3] Done. Raw dataset: ${COLLECT_OUTPUT} ==="
}

run_anchors() {
    echo "=== [2/3] Platoon Anchor Preparation (method=${ANCHOR_METHOD}) ==="
    case "${ANCHOR_METHOD}" in
        k_means)
            "${PYTHON_BIN}" -m expert_dataset.abstract_anchors_default \
                --dataset-root "${COLLECT_OUTPUT}" \
                --output-path "${ANCHORS_OUTPUT}" \
                --split "${ANCHOR_SPLIT}" \
                --trajectory-key "${TRAJECTORY_KEY}" \
                --num-anchors "${NUM_ANCHORS}" \
                --max-trajectories "${ANCHOR_MAX_TRAJECTORIES}" \
                --seed "${ANCHOR_SEED}" \
                --figure-path "${ANCHORS_FIGURE}" \
                --no-show
            echo "=== [2/3] Done. Anchors: ${ANCHORS_OUTPUT} ==="
            ;;
        dynamic)
            echo "Dynamic anchors are generated from the mode-generator context."
            echo "Skipping static K-Means extraction; no platoon_anchors.npy will be generated."
            echo "=== [2/3] Done. Static anchor extraction skipped ==="
            ;;
        *)
            echo "Unsupported ANCHOR_METHOD: ${ANCHOR_METHOD}. Use: k_means | dynamic" >&2
            exit 1
            ;;
    esac
}

run_preprocess() {
    echo "=== [3/3] Multi-agent Diffusion Preprocess ==="
    "${PYTHON_BIN}" -m models.diffusion.preprocess_transfuser_dataset \
        --input-root "${COLLECT_OUTPUT}" \
        --output-root "${PREPROCESS_OUTPUT}" \
        --output-format "${OUTPUT_FORMAT}" \
        --model-config-path "${MODEL_CONFIG_PATH}"
    echo "=== [3/3] Done. Preprocessed dataset: ${PREPROCESS_OUTPUT} ==="
}

echo "Multi-agent pipeline config:"
echo "  STAGE=${STAGE} ANCHOR_METHOD=${ANCHOR_METHOD}"
echo "  agents=${NUM_AGENTS} target_samples=${TARGET_SAMPLES} resume=${RESUME}"
echo "  scenario_weights=${SCENARIO_WEIGHTS}"
echo "  model_config=${MODEL_CONFIG_PATH}"
echo "  train_config=${TRAIN_CONFIG_PATH}"
echo "  collect_output=${COLLECT_OUTPUT}"
echo "  anchors_output=${ANCHORS_OUTPUT}"
echo "  preprocess_output=${PREPROCESS_OUTPUT}"
echo ""

case "${STAGE}" in
    all)        run_collect; run_anchors; run_preprocess ;;
    collect)    run_collect ;;
    anchors)    run_anchors ;;
    preprocess) run_preprocess ;;
    *)
        echo "Unknown STAGE='${STAGE}'. Use: all | collect | anchors | preprocess" >&2
        exit 1
        ;;
esac

echo "=== Multi-agent pipeline complete ==="
