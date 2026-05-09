#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_20/checkpoints/diffusion-epoch=25.ckpt}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${REPO_ROOT}/configs/diffusion/model.yaml}"
SCENARIO_ID="S2_free_cruise_curve"
NUM_AGENTS="${NUM_AGENTS:-3}"
EPISODES="${EPISODES:-5}"
START_SEED="${START_SEED:-0}"
TRAFFIC_DENSITY="${TRAFFIC_DENSITY:-0.06}"
RENDER="${RENDER:-0}"
CONTROLLER_TYPE="${CONTROLLER_TYPE:-stabilized}"
PRINT_TRAJECTORY_DEBUG="${PRINT_TRAJECTORY_DEBUG:-1}"
DEBUG_COORDINATE_AUDIT="${DEBUG_COORDINATE_AUDIT:-1}"
IMAGE_ON_CUDA="${IMAGE_ON_CUDA:-0}"
SAVE_CAMERA_INTERVAL="${SAVE_CAMERA_INTERVAL:-0}"
CAMERA_OUTPUT_DIR="${CAMERA_OUTPUT_DIR:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/eval/closed/cameras}"
SAVE_3D_VIDEO="${SAVE_3D_VIDEO:-0}"
SAVE_2D_VIDEO="${SAVE_2D_VIDEO:-0}"
SAVE_TRAJECTORY_PLOT="${SAVE_TRAJECTORY_PLOT:-1}"
SAVE_COMBINED_TRAJ_FRAMES="${SAVE_COMBINED_TRAJ_FRAMES:-1}"   # combined coarse+multimode+selected per step
SAVE_STEP_IMAGES="${SAVE_STEP_IMAGES:-0}"
STEP_IMAGE_INTERVAL="${STEP_IMAGE_INTERVAL:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/eval/closed}"
VIDEO_FPS="${VIDEO_FPS:-10}"
TOPDOWN_CAMERA_HEIGHT="${TOPDOWN_CAMERA_HEIGHT:-80.0}"
SHOW_TOPOLOGY_POLYLINE="${SHOW_TOPOLOGY_POLYLINE:-1}"
LOCAL_ROUTE="${LOCAL_ROUTE:-}"      # e.g. R1_entry_straight; empty = default random selection
MAX_STEPS="${MAX_STEPS:-0}"         # steps per episode limit; 0 = use env horizon
MODE_CLS_OUTPUT_ROOT="${MODE_CLS_OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/outputs/ppo}"

# PPO_ACTOR_CKPT="${PPO_ACTOR_CKPT:-}"  # pretrained diffusion planner argmax (no PPO)
# PPO_ACTOR_CKPT="${PPO_ACTOR_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/ppo/run_9/checkpoints/step_00020480_score_2678.9948/sb3_model.zip}"
PPO_ACTOR_CKPT="${PPO_ACTOR_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/ppo/run_10/checkpoints/step_00030720_score_2399.1966/sb3_model.zip}"


PPO_RUN_DIR="${PPO_RUN_DIR:-}"
PPO_DETERMINISTIC="${PPO_DETERMINISTIC:-1}"

find_latest_run_dir() {
  local output_dir="$1"
  local latest_run
  latest_run="$(ls -d "${output_dir}"/run_* 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -z "${latest_run}" ]]; then
    return 1
  fi
  echo "${latest_run}"
}

if [[ "${NUM_AGENTS}" -gt 1 ]]; then
  # Auto-resolve from PPO_RUN_DIR if PPO_ACTOR_CKPT is not explicitly set
  if [[ -z "${PPO_ACTOR_CKPT}" && -n "${PPO_RUN_DIR}" ]]; then
    PPO_ACTOR_CKPT="${PPO_RUN_DIR}/checkpoints/final/sb3_model.zip"
  fi
  # If PPO_ACTOR_CKPT is now non-empty but the file is missing → hard error
  if [[ -n "${PPO_ACTOR_CKPT}" && ! -f "${PPO_ACTOR_CKPT}" ]]; then
    echo "[ERROR] PPO_ACTOR_CKPT file not found: ${PPO_ACTOR_CKPT}" >&2
    exit 1
  fi
  # Empty PPO_ACTOR_CKPT → pretrained diffusion planner (no PPO actor)
  if [[ -z "${PPO_ACTOR_CKPT}" ]]; then
    echo "[run_diffusion_test] NUM_AGENTS=${NUM_AGENTS}: using pretrained diffusion planner (no PPO actor)"
  fi
fi

CMD=(
  "${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.test_transfuser_policy
  --checkpoint "${CHECKPOINT_PATH}"
  --model-config-path "${MODEL_CONFIG_PATH}"
  --scenario-id "${SCENARIO_ID}"
  --num-agents "${NUM_AGENTS}"
  --episodes "${EPISODES}"
  --start-seed "${START_SEED}"
  --traffic-density "${TRAFFIC_DENSITY}"
  --render "${RENDER}"
  --controller-type "${CONTROLLER_TYPE}"
  --print-trajectory-debug "${PRINT_TRAJECTORY_DEBUG}"
  --debug-coordinate-audit "${DEBUG_COORDINATE_AUDIT}"
  --image-on-cuda "${IMAGE_ON_CUDA}"
  --save-camera-interval "${SAVE_CAMERA_INTERVAL}"
  --camera-output-dir "${CAMERA_OUTPUT_DIR}"
  --save-3d-video "${SAVE_3D_VIDEO}"
  --save-2d-video "${SAVE_2D_VIDEO}"
  --save-trajectory-plot "${SAVE_TRAJECTORY_PLOT}"
  --save-combined-traj-frames "${SAVE_COMBINED_TRAJ_FRAMES}"
  --save-step-images "${SAVE_STEP_IMAGES}"
  --step-image-interval "${STEP_IMAGE_INTERVAL}"
  --output-dir "${OUTPUT_DIR}"
  --video-fps "${VIDEO_FPS}"
  --topdown-camera-height "${TOPDOWN_CAMERA_HEIGHT}"
  --show-topology-polyline "${SHOW_TOPOLOGY_POLYLINE}"
  --max-steps "${MAX_STEPS}"
)
if [[ "${NUM_AGENTS}" -gt 1 ]]; then
  CMD+=(--ppo-actor-ckpt "${PPO_ACTOR_CKPT}" --ppo-deterministic "${PPO_DETERMINISTIC}")
  if [[ -n "${PPO_RUN_DIR}" ]]; then
    CMD+=(--ppo-run-dir "${PPO_RUN_DIR}")
  fi
fi
if [[ -n "${LOCAL_ROUTE}" ]]; then
  CMD+=(--local-route "${LOCAL_ROUTE}")
fi
echo "[run_diffusion_test] scenario=${SCENARIO_ID} local_route=${LOCAL_ROUTE:-<auto>} num_agents=${NUM_AGENTS}"
if [[ "${NUM_AGENTS}" -gt 1 ]]; then
  echo "[run_diffusion_test] frozen_planner_ckpt=${CHECKPOINT_PATH}"
  echo "[run_diffusion_test] ppo_actor_ckpt=${PPO_ACTOR_CKPT}"
fi
"${CMD[@]}"
