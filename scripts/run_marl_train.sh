#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
LOG_DIR="${LOG_DIR:-}"

cmd=(
    "${PYTHON_BIN}" -m train.train_platoon_rl
    --mode platoon-closedloop
    --num-agents 3
    --steps 300
    --render 0
    --ckpt /media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_4/checkpoints/diffusion-epoch=04.ckpt
)

if [[ -n "${CHECKPOINT_DIR}" ]]; then
    cmd+=(--checkpoint-dir "${CHECKPOINT_DIR}")
fi
if [[ -n "${LOG_DIR}" ]]; then
    cmd+=(--log-dir "${LOG_DIR}")
fi

"${cmd[@]}"
