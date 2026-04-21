#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_13/checkpoints/diffusion-epoch=25.ckpt}"
TRAJECTORY_REG_DECODER_TYPE="${TRAJECTORY_REG_DECODER_TYPE:-gru}"  # mlp | gru
SCENARIO_ID="${SCENARIO_ID:-S4_curve_following}"
EPISODES="${EPISODES:-5}"
RENDER="${RENDER:-0}"
CONTROLLER_TYPE="${CONTROLLER_TYPE:-stabilized}"
PRINT_TRAJECTORY_DEBUG="${PRINT_TRAJECTORY_DEBUG:-1}"
IMAGE_ON_CUDA="${IMAGE_ON_CUDA:-0}"
SAVE_CAMERA_INTERVAL="${SAVE_CAMERA_INTERVAL:-0}"
CAMERA_OUTPUT_DIR="${CAMERA_OUTPUT_DIR:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/eval/closed/cameras}"
SAVE_3D_VIDEO="${SAVE_3D_VIDEO:-0}"
SAVE_2D_VIDEO="${SAVE_2D_VIDEO:-1}"
SAVE_TRAJECTORY_PLOT="${SAVE_TRAJECTORY_PLOT:-1}"
SAVE_STEP_IMAGES="${SAVE_STEP_IMAGES:-1}"
STEP_IMAGE_INTERVAL="${STEP_IMAGE_INTERVAL:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/eval/closed}"
VIDEO_FPS="${VIDEO_FPS:-10}"
TOPDOWN_CAMERA_HEIGHT="${TOPDOWN_CAMERA_HEIGHT:-80.0}"
SHOW_TOPOLOGY_POLYLINE="${SHOW_TOPOLOGY_POLYLINE:-1}"

"${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.test_transfuser_policy \
  --checkpoint "${CHECKPOINT_PATH}" \
  --scenario-id "${SCENARIO_ID}" \
  --episodes "${EPISODES}" \
  --render "${RENDER}" \
  --controller-type "${CONTROLLER_TYPE}" \
  --print-trajectory-debug "${PRINT_TRAJECTORY_DEBUG}" \
  --image-on-cuda "${IMAGE_ON_CUDA}" \
  --save-camera-interval "${SAVE_CAMERA_INTERVAL}" \
  --camera-output-dir "${CAMERA_OUTPUT_DIR}" \
  --save-3d-video "${SAVE_3D_VIDEO}" \
  --save-2d-video "${SAVE_2D_VIDEO}" \
  --save-trajectory-plot "${SAVE_TRAJECTORY_PLOT}" \
  --save-step-images "${SAVE_STEP_IMAGES}" \
  --step-image-interval "${STEP_IMAGE_INTERVAL}" \
  --output-dir "${OUTPUT_DIR}" \
  --video-fps "${VIDEO_FPS}" \
  --topdown-camera-height "${TOPDOWN_CAMERA_HEIGHT}" \
  --show-topology-polyline "${SHOW_TOPOLOGY_POLYLINE}" \
  --trajectory-reg-decoder-type "${TRAJECTORY_REG_DECODER_TYPE}"
