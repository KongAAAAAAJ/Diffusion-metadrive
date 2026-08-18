#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
VARIANT="${VARIANT:-A}"
RUN_MODE="${RUN_MODE:-smoke}"
PIPELINE_STAGE="${PIPELINE_STAGE:-train}"
ALLOW_FAILED_CALIBRATION_DIAGNOSTIC="${ALLOW_FAILED_CALIBRATION_DIAGNOSTIC:-0}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/train/bev_joint_grpo.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/media/kong/Elements_SE/Diffusion_Data/outputs/bev_diffusion_stage1/run_1/checkpoints/best.pt}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/outputs/bev_joint_grpo_open_reward_v2/stage1_run_1}"
DEVELOPMENT_REPORT="${DEVELOPMENT_REPORT:-${ARTIFACT_ROOT}/calibration_v3/A-development.json}"
CALIBRATION_REPORT="${CALIBRATION_REPORT:-${ARTIFACT_ROOT}/calibration_v3/A-holdout.json}"
DEFAULT_OUTPUT_ROOT="${ARTIFACT_ROOT}/training_v2"
DEFAULT_LOG_ROOT="${ARTIFACT_ROOT}/logs_v2"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
LOG_ROOT="${LOG_ROOT:-$DEFAULT_LOG_ROOT}"
DEVICE="${DEVICE:-cuda}"
STATES_PER_EPISODE="${STATES_PER_EPISODE:-3}"
MAX_OPTIMIZER_STEPS="${MAX_OPTIMIZER_STEPS:-10000}"

if [[ "$VARIANT" != "A" ]]; then
  echo "at-risk GRPO-Open is authorized only for Variant A" >&2
  exit 2
fi
if [[ "$RUN_MODE" != "smoke" ]]; then
  echo "at-risk GRPO-Open requires RUN_MODE=smoke" >&2
  exit 2
fi
if [[ "$ALLOW_FAILED_CALIBRATION_DIAGNOSTIC" != "0" ]]; then
  echo "Stage2 joint reward V2 forbids failed-calibration bypass" >&2
  exit 2
fi
if [[ ! "$MAX_OPTIMIZER_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_OPTIMIZER_STEPS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$STATES_PER_EPISODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "STATES_PER_EPISODE must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  echo "Stage1 source checkpoint does not exist: $SOURCE_CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "GRPO config does not exist: $CONFIG" >&2
  exit 2
fi

echo "[GRPO] pipeline_stage=$PIPELINE_STAGE run_mode=$RUN_MODE variant=$VARIANT"
echo "[GRPO] calibration_bypass=$ALLOW_FAILED_CALIBRATION_DIAGNOSTIC max_optimizer_steps=$MAX_OPTIMIZER_STEPS"
echo "[GRPO] calibration_report=$CALIBRATION_REPORT"
echo "[GRPO] output_root=$OUTPUT_ROOT"

mkdir -p "$(dirname "$DEVELOPMENT_REPORT")" "$(dirname "$CALIBRATION_REPORT")" "$OUTPUT_ROOT" "$LOG_ROOT"

run_logged() {
  local name="$1"
  shift
  "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"
}

run_calibration() {
  run_logged development-calibration \
    "$PYTHON_BIN" "${PROJECT_ROOT}/scripts/calibrate_bev_joint_reward.py" \
      --variant A \
      --stage1-checkpoint "$SOURCE_CHECKPOINT" \
      --output "$DEVELOPMENT_REPORT" \
      --device "$DEVICE" \
      --states-per-episode "$STATES_PER_EPISODE" \
      --phase development \
      --quiet

  run_logged holdout-calibration \
    "$PYTHON_BIN" "${PROJECT_ROOT}/scripts/calibrate_bev_joint_reward.py" \
      --variant A \
      --stage1-checkpoint "$SOURCE_CHECKPOINT" \
      --output "$CALIBRATION_REPORT" \
      --device "$DEVICE" \
      --states-per-episode "$STATES_PER_EPISODE" \
      --phase holdout \
      --tracking-envelope-report "$DEVELOPMENT_REPORT" \
      --quiet
}

ARGS=(
  -m train.train_bev_joint_grpo_online
  --config "$CONFIG"
  --variant "$VARIANT"
  --run-mode "$RUN_MODE"
  --source-checkpoint "$SOURCE_CHECKPOINT"
  --calibration-report "$CALIBRATION_REPORT"
  --output-root "$OUTPUT_ROOT"
  --max-optimizer-steps "$MAX_OPTIMIZER_STEPS"
)
run_training() {
  if [[ ! -f "$CALIBRATION_REPORT" ]]; then
    echo "holdout calibration report does not exist: $CALIBRATION_REPORT" >&2
    exit 2
  fi
  run_logged grpo-open-training "$PYTHON_BIN" "${ARGS[@]}"
}

case "$PIPELINE_STAGE" in
  calibrate)
    run_calibration
    ;;
  train)
    run_training
    ;;
  all)
    run_calibration
    run_training
    ;;
  *)
    echo "PIPELINE_STAGE must be calibrate, train, or all" >&2
    exit 2
    ;;
esac
