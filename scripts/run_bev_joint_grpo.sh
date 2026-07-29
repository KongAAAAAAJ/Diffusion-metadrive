#!/usr/bin/env bash
set -euo pipefail

VARIANT="${VARIANT:-A}"
RUN_MODE="${RUN_MODE:-smoke}"
CONFIG="${CONFIG:-configs/train/bev_joint_grpo.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?SOURCE_CHECKPOINT is required}"
CALIBRATION_REPORT="${CALIBRATION_REPORT:?CALIBRATION_REPORT is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

ARGS=(
  -m train.train_bev_joint_grpo_online
  --config "$CONFIG"
  --variant "$VARIANT"
  --run-mode "$RUN_MODE"
  --source-checkpoint "$SOURCE_CHECKPOINT"
  --calibration-report "$CALIBRATION_REPORT"
  --output-root "$OUTPUT_ROOT"
)

if [[ "$RUN_MODE" == "smoke" ]]; then
  MAX_OPTIMIZER_STEPS="${MAX_OPTIMIZER_STEPS:-20}"
  ARGS+=(--max-optimizer-steps "$MAX_OPTIMIZER_STEPS")
fi

exec /home/kong/anaconda3/envs/meta_drive/bin/python "${ARGS[@]}"
