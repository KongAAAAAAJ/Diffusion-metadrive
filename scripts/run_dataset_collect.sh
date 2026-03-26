#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TARGET_SAMPLES="${TARGET_SAMPLES:-5000}" # *
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets}"

EXPERT_TYPE="${EXPERT_TYPE:-idm}" # *
START_SEED="${START_SEED:-10}" # *
SEED_LIST="${SEED_LIST:-10,11,12,13,14,15,16,17,18,19}"

COLLECTION_MODE="${COLLECTION_MODE:-single}"  # *single | fixed_hybrid | random_road | phase2_plan

DATASET_NAME="${DATASET_NAME:-1_metaData_${EXPERT_TYPE}_${COLLECTION_MODE}}"

LOW_TRAFFIC_DENSITY="${LOW_TRAFFIC_DENSITY:-0.08}" # *
HIGH_TRAFFIC_DENSITY="${HIGH_TRAFFIC_DENSITY:-0.12}"

USE_HYBRID_MAP="${USE_HYBRID_MAP:-1}"
HYBRID_MAP_SEQUENCE="${HYBRID_MAP_SEQUENCE:-SSXCOCSS}"  # used in single

MAP_BLOCK_NUM="${MAP_BLOCK_NUM:-5}"
NUM_SCENARIOS="${NUM_SCENARIOS:-1}"

TRAJECTORY_CORRECTION_ENABLED="${TRAJECTORY_CORRECTION_ENABLED:-1}"  # 开启轨迹修正，only in ppo
SAVE_RAW_TRAJECTORY="${SAVE_RAW_TRAJECTORY:-1}"  # 保存原始轨迹
MODE_CLASSIFIER_VERSION="${MODE_CLASSIFIER_VERSION:-v1}"  
CENTERLINE_ATTRACTION_STRENGTH="${CENTERLINE_ATTRACTION_STRENGTH:-0.85}"
SMOOTHING_STRENGTH="${SMOOTHING_STRENGTH:-0.25}"

TRAJECTORY_VISUALIZATION_ENABLED="${TRAJECTORY_VISUALIZATION_ENABLED:-1}"  # 开启/关闭轨迹可视化
TRAJECTORY_VISUALIZATION_FRONT_MARGIN="${TRAJECTORY_VISUALIZATION_FRONT_MARGIN:-25}"  # 前向可视化范围（单位：米）
TRAJECTORY_VISUALIZATION_LATERAL_MARGIN="${TRAJECTORY_VISUALIZATION_LATERAL_MARGIN:-10}"  # 横向可视化范围（单位：米）

LOG_PATH="${OUTPUT_ROOT}/collect_command.log"
mkdir -p "${OUTPUT_ROOT}"

run_collect_job() {
    local mode="$1"
    local label="$2"
    local use_hybrid_map="$3"
    local hybrid_map_sequence="$4"
    local map_block_num="$5"
    local num_scenarios="$6"
    local seed="$7"
    local density_min="$8"
    local density_max="$9"

    local dataset_name="${DATASET_NAME}"
    local log_path="${OUTPUT_ROOT}/collect_command_${DATASET_NAME}.log"
    local cmd=(
        "${PYTHON_BIN}" -m metadrive.exp_dataset.collect_expert
        --target-samples "${TARGET_SAMPLES}"
        --output-root "${OUTPUT_ROOT}"
        --dataset-name "${dataset_name}"
        --expert-type "${EXPERT_TYPE}"
        --start-seed "${seed}"
        --trajectory-correction-enabled "${TRAJECTORY_CORRECTION_ENABLED}"
        --save-raw-trajectory "${SAVE_RAW_TRAJECTORY}"
        --mode-classifier-version "${MODE_CLASSIFIER_VERSION}"
        --centerline-attraction-strength "${CENTERLINE_ATTRACTION_STRENGTH}"
        --smoothing-strength "${SMOOTHING_STRENGTH}"
        --trajectory-visualization-enabled "${TRAJECTORY_VISUALIZATION_ENABLED}"
        --trajectory-visualization-front-margin "${TRAJECTORY_VISUALIZATION_FRONT_MARGIN}"
        --trajectory-visualization-lateral-margin "${TRAJECTORY_VISUALIZATION_LATERAL_MARGIN}"
        --traffic-density-min "${density_min}"
        --traffic-density-max "${density_max}"
        --use-hybrid-map "${use_hybrid_map}"
        --hybrid-map-sequence "${hybrid_map_sequence}"
        --map-block-num "${map_block_num}"
        --num-scenarios "${num_scenarios}"
    )

    echo "mode=${mode} label=${label} seed=${seed} density=[${density_min},${density_max}] use_hybrid_map=${use_hybrid_map} hybrid_map_sequence=${hybrid_map_sequence} map_block_num=${map_block_num} num_scenarios=${num_scenarios}"

    "${cmd[@]}" 2>&1 | tee "${log_path}"
}

IFS=',' read -r -a SEEDS <<< "${SEED_LIST}"

case "${COLLECTION_MODE}" in
    single)
        run_collect_job "single" "${DATASET_NAME}" "${USE_HYBRID_MAP}" "${HYBRID_MAP_SEQUENCE}" "${MAP_BLOCK_NUM}" "${NUM_SCENARIOS}" "${START_SEED}" "${LOW_TRAFFIC_DENSITY}" "${HIGH_TRAFFIC_DENSITY}"
        ;;
    fixed_hybrid|phase2_plan)
        # straight road structure: straight-only hybrid sequence for cruising coverage.
        for seed in "${SEEDS[@]}"; do
            run_collect_job "fixed_hybrid" "straight" "1" "SSSSS" "5" "1" "${seed}" "${LOW_TRAFFIC_DENSITY}" "${LOW_TRAFFIC_DENSITY}"
        done
        # curve road structure: curve-dominant hybrid sequence for turning coverage.
        for seed in "${SEEDS[@]}"; do
            run_collect_job "fixed_hybrid" "curve" "1" "CCSCC" "5" "1" "${seed}" "${HIGH_TRAFFIC_DENSITY}" "${HIGH_TRAFFIC_DENSITY}"
        done
        # intersection road structure: multiple X blocks for intersection interactions.
        for seed in "${SEEDS[@]}"; do
            run_collect_job "fixed_hybrid" "intersection" "1" "SXSXS" "5" "1" "${seed}" "${LOW_TRAFFIC_DENSITY}" "${HIGH_TRAFFIC_DENSITY}"
        done
        # roundabout road structure: O blocks for entering and exiting roundabouts.
        for seed in "${SEEDS[@]}"; do
            run_collect_job "fixed_hybrid" "roundabout" "1" "SOSSO" "5" "1" "${seed}" "${HIGH_TRAFFIC_DENSITY}" "${HIGH_TRAFFIC_DENSITY}"
        done
        [[ "${COLLECTION_MODE}" == "fixed_hybrid" ]] && exit 0
        ;;&
    random_road|phase2_plan)
        # mixed random road structure: random block maps covering straight / curve / intersection / roundabout by seed sampling.
        for seed in "${SEEDS[@]}"; do
            run_collect_job "random_road" "random_blocks" "0" "SSXCOCSS" "${MAP_BLOCK_NUM}" "10" "${seed}" "${LOW_TRAFFIC_DENSITY}" "${HIGH_TRAFFIC_DENSITY}"
        done
        ;;
    *)
        echo "Unsupported COLLECTION_MODE: ${COLLECTION_MODE}" >&2
        exit 1
        ;;
esac
