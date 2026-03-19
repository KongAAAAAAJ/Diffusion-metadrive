#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/checkpoints/run_1/diffusion-epoch=98.ckpt}"
EPISODES="${EPISODES:-1}"
RENDER="${RENDER:-1}"
CONTROLLER_TYPE="${CONTROLLER_TYPE:-stabilized}"
PRINT_TRAJECTORY_DEBUG="${PRINT_TRAJECTORY_DEBUG:-1}"
IMAGE_ON_CUDA="${IMAGE_ON_CUDA:-0}"
SAVE_CAMERA_INTERVAL="${SAVE_CAMERA_INTERVAL:-0}"
CAMERA_OUTPUT_DIR="${CAMERA_OUTPUT_DIR:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/closed_loop/cameras}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CMD=(
"${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.test_transfuser_policy
    --checkpoint "${CHECKPOINT_PATH}"
    --episodes "${EPISODES}"
    --render "${RENDER}"
    --controller-type "${CONTROLLER_TYPE}"
    --print-trajectory-debug "${PRINT_TRAJECTORY_DEBUG}"
    --image-on-cuda "${IMAGE_ON_CUDA}"
    --save-camera-interval "${SAVE_CAMERA_INTERVAL}"
    --camera-output-dir "${CAMERA_OUTPUT_DIR}"
)

"${CMD[@]}"
