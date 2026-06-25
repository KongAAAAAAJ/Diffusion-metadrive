#!/usr/bin/env bash
# run_data_pipeline.sh — unified data pipeline: collect → anchors → preprocess
# Usage: STAGE=all|collect|anchors|preprocess [VAR=val ...] bash scripts/run_data_pipeline.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${REPO_ROOT}/configs/diffusion/model.yaml}"

resolve_model_config_anchor_method() {
    "${PYTHON_BIN}" - "${MODEL_CONFIG_PATH}" <<'PY'
import sys
from models.diffusion.transfuser_config import load_diffusion_model_config

config = load_diffusion_model_config(sys.argv[1])
print(config.get("anchor_method", "dynamic"))
PY
}

# ── Stage selection ─────────────────────────────────────────────────────────
STAGE="${STAGE:-all}"  # all | collect | anchors | preprocess

# *── Shared paths ────────────────────────────────────────────────────────────
EXPERT_TYPE="${EXPERT_TYPE:-idm}"
COLLECTION_MODE="${COLLECTION_MODE:-single}"  # 选择地图模式：single | fixed_hybrid | random_road | phase2_plan
ANCHOR_METHOD="${ANCHOR_METHOD:-$(resolve_model_config_anchor_method)}"  # k_means | dynamic
DATASET_NAME="${DATASET_NAME:-metaIDM}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets}"
# SCENARIO_WEIGHTS=${SCENARIO_WEIGHTS:-'{
#     "S1_free_cruise_straight": 1.0,
#     "S2_free_cruise_curve": 2.0,
#     "S3_straight_following": 3.0,
#     "S4_curve_following": 3.0}'}
SCENARIO_WEIGHTS=${SCENARIO_WEIGHTS:-'{
    "S1_free_cruise_straight": 1.0,
    "S2_free_cruise_curve": 1.0,
    "S3_straight_following": 2.0,
    "S4_curve_following": 2.0,
    "S5_hard_brake_lead": 1.0,
    "S6_background_merge_in": 1.0,
    "S7_ego_merge_from_ramp": 1.0,
    "S8_ego_exit_to_ramp": 1.0,
    "S9_narrow_channel_negotiation": 1.0,
    "S10_straight_lane_change": 2.0,
    "S11_curve_lane_change": 2.0}'}
IDM_VARIANT_WEIGHTS=${IDM_VARIANT_WEIGHTS:-'{"default": 1.0}'}

# Derived paths (auto-chained between stages)
COLLECT_OUTPUT="${OUTPUT_ROOT}/${DATASET_NAME}"
ANCHORS_OUTPUT="${ANCHORS_OUTPUT:-${REPO_ROOT}/expert_dataset/anchors.npy}"
ANCHORS_FIGURE="${ANCHORS_FIGURE:-${REPO_ROOT}/expert_dataset/anchors.png}"
PREPROCESS_OUTPUT="${PREPROCESS_OUTPUT:-${COLLECT_OUTPUT}_pp}"

# Collection parameters
TARGET_SAMPLES="${TARGET_SAMPLES:-40000}"  # *
START_SEED="${START_SEED:-10}" # *
SEED_LIST="${SEED_LIST:-10,11,12,13,14,15,16,17,18,19}"
LOW_TRAFFIC_DENSITY="${LOW_TRAFFIC_DENSITY:-0.08}"
HIGH_TRAFFIC_DENSITY="${HIGH_TRAFFIC_DENSITY:-0.12}"
USE_HYBRID_MAP="${USE_HYBRID_MAP:-1}"
MAP_BLOCK_NUM="${MAP_BLOCK_NUM:-5}"
NUM_SCENARIOS="${NUM_SCENARIOS:-1}"
TRAJECTORY_CORRECTION_ENABLED="${TRAJECTORY_CORRECTION_ENABLED:-1}"
SAVE_RAW_TRAJECTORY="${SAVE_RAW_TRAJECTORY:-1}"
MODE_CLASSIFIER_VERSION="${MODE_CLASSIFIER_VERSION:-v1}"
CENTERLINE_ATTRACTION_STRENGTH="${CENTERLINE_ATTRACTION_STRENGTH:-0.85}"
SMOOTHING_STRENGTH="${SMOOTHING_STRENGTH:-0.25}"
TRAJECTORY_VISUALIZATION_ENABLED="${TRAJECTORY_VISUALIZATION_ENABLED:-1}"
TRAJECTORY_VISUALIZATION_FRONT_MARGIN="${TRAJECTORY_VISUALIZATION_FRONT_MARGIN:-25}"
TRAJECTORY_VISUALIZATION_LATERAL_MARGIN="${TRAJECTORY_VISUALIZATION_LATERAL_MARGIN:-10}"

# Anchor parameters
TRAJECTORY_KEY="${TRAJECTORY_KEY:-trajectory}"
NUM_ANCHORS="${NUM_ANCHORS:-20}"  # *
ANCHOR_SEED="${ANCHOR_SEED:-0}"

# Preprocess parameters
OUTPUT_FORMAT="${OUTPUT_FORMAT:-dir}"

# ── Stage 1: Data Collection ────────────────────────────────────────────────
run_collect_once() {
    local use_hybrid_map="$1"
    local map_block_num="$2"
    local num_scenarios="$3"
    local seed="$4"
    local density_min="$5"
    local density_max="$6"
    local log_path="${OUTPUT_ROOT}/collect_command_${DATASET_NAME}.log"

    "${PYTHON_BIN}" -m expert_dataset.collect_expert \
        --target-samples "${TARGET_SAMPLES}" \
        --output-root "${OUTPUT_ROOT}" \
        --dataset-name "${DATASET_NAME}" \
        --model-config-path "${MODEL_CONFIG_PATH}" \
        --expert-type "${EXPERT_TYPE}" \
        --start-seed "${seed}" \
        --trajectory-correction-enabled "${TRAJECTORY_CORRECTION_ENABLED}" \
        --save-raw-trajectory "${SAVE_RAW_TRAJECTORY}" \
        --mode-classifier-version "${MODE_CLASSIFIER_VERSION}" \
        --centerline-attraction-strength "${CENTERLINE_ATTRACTION_STRENGTH}" \
        --smoothing-strength "${SMOOTHING_STRENGTH}" \
        --trajectory-visualization-enabled "${TRAJECTORY_VISUALIZATION_ENABLED}" \
        --trajectory-visualization-front-margin "${TRAJECTORY_VISUALIZATION_FRONT_MARGIN}" \
        --trajectory-visualization-lateral-margin "${TRAJECTORY_VISUALIZATION_LATERAL_MARGIN}" \
        --traffic-density-min "${density_min}" \
        --traffic-density-max "${density_max}" \
        --scenario-weights "${SCENARIO_WEIGHTS}" \
        --idm-variant-weights "${IDM_VARIANT_WEIGHTS}" \
        --use-hybrid-map "${use_hybrid_map}" \
        --map-block-num "${map_block_num}" \
        --num-scenarios "${num_scenarios}" \
        2>&1 | tee "${log_path}"
}

run_collect() {
    echo "=== [1/3] Data Collection (mode=${COLLECTION_MODE}) ==="
    mkdir -p "${OUTPUT_ROOT}"

    IFS=',' read -r -a SEEDS <<< "${SEED_LIST}"

    case "${COLLECTION_MODE}" in
        single)
            run_collect_once "${USE_HYBRID_MAP}" "${MAP_BLOCK_NUM}" \
                "${NUM_SCENARIOS}" "${START_SEED}" "${LOW_TRAFFIC_DENSITY}" "${HIGH_TRAFFIC_DENSITY}"
            ;;
        fixed_hybrid|phase2_plan)
            for seed in "${SEEDS[@]}"; do
                run_collect_once "1" "5" "1" "${seed}" "${LOW_TRAFFIC_DENSITY}" "${LOW_TRAFFIC_DENSITY}"
                run_collect_once "1" "5" "1" "${seed}" "${HIGH_TRAFFIC_DENSITY}" "${HIGH_TRAFFIC_DENSITY}"
                run_collect_once "1" "5" "1" "${seed}" "${LOW_TRAFFIC_DENSITY}" "${HIGH_TRAFFIC_DENSITY}"
                run_collect_once "1" "5" "1" "${seed}" "${HIGH_TRAFFIC_DENSITY}" "${HIGH_TRAFFIC_DENSITY}"
            done
            [[ "${COLLECTION_MODE}" == "fixed_hybrid" ]] && return 0
            ;;&
        random_road|phase2_plan)
            for seed in "${SEEDS[@]}"; do
                run_collect_once "0" "${MAP_BLOCK_NUM}" "10" "${seed}" "${LOW_TRAFFIC_DENSITY}" "${HIGH_TRAFFIC_DENSITY}"
            done
            ;;
        *)
            echo "Unsupported COLLECTION_MODE: ${COLLECTION_MODE}" >&2; exit 1 ;;
    esac
    echo "=== [1/3] Done. Output: ${COLLECT_OUTPUT} ==="
}

# ── Stage 2: Anchor Extraction ──────────────────────────────────────────────
run_anchors() {
    echo "=== [2/3] Anchor Preparation (method=${ANCHOR_METHOD}) ==="

    case "${ANCHOR_METHOD}" in
        k_means)
            "${PYTHON_BIN}" -m expert_dataset.abstract_anchors_default \
                --dataset-root "${COLLECT_OUTPUT}" \
                --output-path "${ANCHORS_OUTPUT}" \
                --trajectory-key "${TRAJECTORY_KEY}" \
                --num-anchors "${NUM_ANCHORS}" \
                --seed "${ANCHOR_SEED}" \
                --figure-path "${ANCHORS_FIGURE}" \
                --no-show
            ;;
        dynamic)
            echo "Dynamic anchors use mode-generator trajectories from the dataset/live context."
            echo "Skipping static anchor extraction; no anchors.npy will be generated in dynamic mode."
            ;;
        *)
            echo "Unsupported ANCHOR_METHOD: ${ANCHOR_METHOD}. Use: k_means | dynamic" >&2
            exit 1
            ;;
    esac
    echo "=== [2/3] Done. Anchors: ${ANCHORS_OUTPUT} ==="
}

# ── Stage 3: Diffusion Preprocess ───────────────────────────────────────────
run_preprocess() {
    echo "=== [3/3] Diffusion Preprocess ==="
    "${PYTHON_BIN}" -m models.diffusion.preprocess_transfuser_dataset \
        --input-root "${COLLECT_OUTPUT}" \
        --output-root "${PREPROCESS_OUTPUT}" \
        --output-format "${OUTPUT_FORMAT}" \
        --model-config-path "${MODEL_CONFIG_PATH}"
    echo "=== [3/3] Done. Preprocessed: ${PREPROCESS_OUTPUT} ==="
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
echo "Pipeline config: STAGE=${STAGE} EXPERT_TYPE=${EXPERT_TYPE} COLLECTION_MODE=${COLLECTION_MODE} ANCHOR_METHOD=${ANCHOR_METHOD}"
echo "  model_config=${MODEL_CONFIG_PATH}"
echo "  collect_output=${COLLECT_OUTPUT}"
echo "  anchors_output=${ANCHORS_OUTPUT}"
echo "  preprocess_output=${PREPROCESS_OUTPUT}"
echo ""

case "${STAGE}" in
    all)       run_collect; run_anchors; run_preprocess ;;
    collect)   run_collect ;;
    anchors)   run_anchors ;;
    preprocess) run_preprocess ;;
    *)
        echo "Unknown STAGE='${STAGE}'. Use: all | collect | anchors | preprocess" >&2
        exit 1 ;;
esac

echo "=== Pipeline complete ==="
