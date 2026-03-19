#!/usr/bin/env bash
set -euo pipefail

# Legacy migration helper only.
# Use this script only for old CHW-format camera datasets. New collect_expert output is already HWC.

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_ROOT="${INPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metadrive_ppo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metadrive_ppo_hwc}"

"${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.convert_camera_layout_dataset \
    --input-root "${INPUT_ROOT}" \
    --output-root "${OUTPUT_ROOT}"
