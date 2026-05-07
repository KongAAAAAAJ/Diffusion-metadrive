#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_20/checkpoints/diffusion-epoch=25.ckpt}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-configs/diffusion/model.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/outputs/single_step_manual_test}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

NUM_AGENTS="${NUM_AGENTS:-3}"
MANUAL_MODES="${MANUAL_MODES:-10,10,15}"  # 0 = no manual mode; 1 = manual mode for all agents; 2 = manual mode for one random agent
TARGET_POINT_POLICY="${TARGET_POINT_POLICY:-selected_endpoint}"  # expert = use expert-defined target point; default = use original target point (e.g. route endpoint)
SCENARIO_ID="${SCENARIO_ID:-S1_free_cruise_straight}"
LOCAL_ROUTE="${LOCAL_ROUTE:-}"
START_SEED="${START_SEED:-0}"
TRAFFIC_DENSITY="${TRAFFIC_DENSITY:-0.06}"
MAX_STEPS="${MAX_STEPS:-1}"
USE_RENDER="${USE_RENDER:-0}"
IMAGE_ON_CUDA="${IMAGE_ON_CUDA:-0}"
SAVE_TRAJECTORY_PLOT="${SAVE_TRAJECTORY_PLOT:-1}"
LOOKAHEAD_INDEX="${LOOKAHEAD_INDEX:-2}"
TARGET_SPEED_KM_H="${TARGET_SPEED_KM_H:-30.0}"
CONTROLLER_TYPE="${CONTROLLER_TYPE:-stabilized}"
PLANNER_DEVICE="${PLANNER_DEVICE:-auto}"

cmd=(
  "${PYTHON_BIN}" -m train.single_step_test
  --checkpoint "${CHECKPOINT_PATH}"
  --model-config-path "${MODEL_CONFIG_PATH}"
  --output-root "${OUTPUT_ROOT}"
  --manual-modes "${MANUAL_MODES}"
  --target-point-policy "${TARGET_POINT_POLICY}"
  --scenario-id "${SCENARIO_ID}"
  --num-agents "${NUM_AGENTS}"
  --max-steps "${MAX_STEPS}"
  --start-seed "${START_SEED}"
  --traffic-density "${TRAFFIC_DENSITY}"
  --use-render "${USE_RENDER}"
  --image-on-cuda "${IMAGE_ON_CUDA}"
  --save-trajectory-plot "${SAVE_TRAJECTORY_PLOT}"
  --lookahead-index "${LOOKAHEAD_INDEX}"
  --target-speed-km-h "${TARGET_SPEED_KM_H}"
  --controller-type "${CONTROLLER_TYPE}"
  --planner-device "${PLANNER_DEVICE}"
)

if [[ -n "${LOCAL_ROUTE}" ]]; then
  cmd+=(--local-route "${LOCAL_ROUTE}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi

echo "[single_step] command: ${cmd[*]}"
"${cmd[@]}"
