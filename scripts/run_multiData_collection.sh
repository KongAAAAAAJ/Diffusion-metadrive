#!/usr/bin/env bash
# run_multiData_collection.sh — platoon data pipeline: collect → anchors → preprocess
# Usage: STAGE=all|collect|anchors|preprocess [VAR=val ...] bash scripts/run_multiData_collection.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${REPO_ROOT}/configs/diffusion/model.yaml}"
DATASET_CONFIG_PATH="${DATASET_CONFIG_PATH:-${REPO_ROOT}/configs/dataset/data_collect.yaml}"

STAGE="${STAGE:-all}"  # all | collect | anchors | preprocess
if [[ -z "${ANCHOR_METHOD:-}" ]]; then
    ANCHOR_METHOD="$("${PYTHON_BIN}" -c \
        'import sys; from models.diffusion.transfuser_config import load_diffusion_model_config; print(load_diffusion_model_config(sys.argv[1]).get("anchor_method", "dynamic"))' \
        "${MODEL_CONFIG_PATH}")"
fi

case "${STAGE}" in
    all|collect|anchors|preprocess) ;;
    *)
        echo "Unknown STAGE='${STAGE}'. Use: all | collect | anchors | preprocess" >&2
        exit 1
        ;;
esac

DATASET_NAME="${DATASET_NAME:-platoon_rule_expert}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets}"

COLLECT_OUTPUT="${OUTPUT_ROOT}/${DATASET_NAME}"
PREPROCESS_OUTPUT="${PREPROCESS_OUTPUT:-${COLLECT_OUTPUT}_pp}"
ANCHORS_OUTPUT="${ANCHORS_OUTPUT:-${REPO_ROOT}/expert_dataset/platoon_anchors.npy}"
ANCHORS_FIGURE="${ANCHORS_FIGURE:-${REPO_ROOT}/expert_dataset/platoon_anchors.png}"

TARGET_SAMPLES="${TARGET_SAMPLES:-40000}"
START_SEED="${START_SEED:-10}"
MAX_EPISODES="${MAX_EPISODES:-0}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-0}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-2048}"
RESUME="${RESUME:-1}"
NUM_AGENTS="${NUM_AGENTS:-3}"
TRAJECTORY_VISUALIZATION_ENABLED="${TRAJECTORY_VISUALIZATION_ENABLED:-0}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"

TRAJECTORY_KEY="${TRAJECTORY_KEY:-trajectory}"
NUM_ANCHORS="${NUM_ANCHORS:-20}"
ANCHOR_SEED="${ANCHOR_SEED:-0}"
ANCHOR_SPLIT="${ANCHOR_SPLIT:-all}"
ANCHOR_MAX_TRAJECTORIES="${ANCHOR_MAX_TRAJECTORIES:-100000}"

OUTPUT_FORMAT="${OUTPUT_FORMAT:-dir}"

echo "Multi-agent pipeline config:"
echo "  STAGE=${STAGE} ANCHOR_METHOD=${ANCHOR_METHOD}"
echo "  agents=${NUM_AGENTS} target_samples=${TARGET_SAMPLES} resume=${RESUME}"
echo "  model_config=${MODEL_CONFIG_PATH}"
echo "  dataset_config=${DATASET_CONFIG_PATH}"
echo "  collect_output=${COLLECT_OUTPUT}"
echo "  anchors_output=${ANCHORS_OUTPUT}"
echo "  preprocess_output=${PREPROCESS_OUTPUT}"
echo ""

if [[ "${STAGE}" == "all" || "${STAGE}" == "collect" ]]; then
    echo "=== [1/3] Multi-agent Expert Collection ==="
    mkdir -p "${OUTPUT_ROOT}"
    "${PYTHON_BIN}" -m expert_dataset.collect_multi_experts \
        --output-root "${OUTPUT_ROOT}" \
        --dataset-name "${DATASET_NAME}" \
        --model-config-path "${MODEL_CONFIG_PATH}" \
        --dataset-config-path "${DATASET_CONFIG_PATH}" \
        --target-samples "${TARGET_SAMPLES}" \
        --start-seed "${START_SEED}" \
        --max-episodes "${MAX_EPISODES}" \
        --max-episode-steps "${MAX_EPISODE_STEPS}" \
        --samples-per-shard "${SAMPLES_PER_SHARD}" \
        --resume "${RESUME}" \
        --num-agents "${NUM_AGENTS}" \
        --trajectory-visualization-enabled "${TRAJECTORY_VISUALIZATION_ENABLED}" \
        --save-videos "${SAVE_VIDEOS}" \
        2>&1 | tee -a "${OUTPUT_ROOT}/collect_multi_command_${DATASET_NAME}.log"
    echo "=== [1/3] Done: ${COLLECT_OUTPUT} ==="
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "anchors" ]]; then
    echo "=== [2/3] Anchor Preparation (${ANCHOR_METHOD}) ==="
    if [[ "${ANCHOR_METHOD}" == "k_means" ]]; then
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
        echo "=== [2/3] Done: ${ANCHORS_OUTPUT} ==="
    elif [[ "${ANCHOR_METHOD}" == "dynamic" ]]; then
        echo "Dynamic anchors enabled. Skipping static K-Means extraction."
    else
        echo "Unsupported ANCHOR_METHOD: ${ANCHOR_METHOD}. Use: k_means | dynamic" >&2
        exit 1
    fi
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "preprocess" ]]; then
    echo "=== [3/3] Diffusion Preprocess ==="
    "${PYTHON_BIN}" -m models.diffusion.preprocess_transfuser_dataset \
        --input-root "${COLLECT_OUTPUT}" \
        --output-root "${PREPROCESS_OUTPUT}" \
        --output-format "${OUTPUT_FORMAT}" \
        --model-config-path "${MODEL_CONFIG_PATH}"
    echo "=== [3/3] Done: ${PREPROCESS_OUTPUT} ==="
fi

echo "=== Multi-agent pipeline complete ==="
