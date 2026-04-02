#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"

RL_CHECKPOINT="${RL_CHECKPOINT:-latest}"
RL_CHECKPOINT_ROOT="${RL_CHECKPOINT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/outputs/selector}"
RL_CONFIG="${RL_CONFIG:-configs/train/selector.yaml}"
SINGLE_CKPT="${SINGLE_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_2/checkpoints/diffusion-epoch=78.ckpt}"
EPISODES="${EPISODES:-1}"
RENDER="${RENDER:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
MAX_STEPS="${MAX_STEPS:-1000}"
EXPLORE="${EXPLORE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/kong/Elements_SE/Diffusion_Data/outputs/selector_test}"
DEVICE="${DEVICE:-auto}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

CMD=(
  "$PYTHON_BIN" scripts/test_platoon_rl_ckpt.py
  --checkpoint "$RL_CHECKPOINT"
  --checkpoint-root "$RL_CHECKPOINT_ROOT"
  --config "$RL_CONFIG"
  --episodes "$EPISODES"
  --render "$RENDER"
  --save-video "$SAVE_VIDEO"
  --device "$DEVICE"
  --explore "$EXPLORE"
)

if [[ -n "$SINGLE_CKPT" ]]; then
  CMD+=(--single-ckpt "$SINGLE_CKPT")
fi

if [[ -n "$MAX_STEPS" ]]; then
  CMD+=(--max-steps "$MAX_STEPS")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(--output-dir "$OUTPUT_DIR")
fi

printf '[INFO] Testing RL checkpoint: %s\n' "$RL_CHECKPOINT"
printf '[INFO] Config: %s\n' "$RL_CONFIG"

"${CMD[@]}"
