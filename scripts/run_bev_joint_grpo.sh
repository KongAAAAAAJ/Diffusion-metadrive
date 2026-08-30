#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
VARIANT="${VARIANT:-A}"
RUN_MODE="${RUN_MODE:-smoke}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/train/bev_joint_grpo.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/media/kong/Elements_SE/Diffusion_Data/outputs/bev_diffusion_stage1/run_3/checkpoints/best.pt}"
MAX_ROLLOUT_GROUPS="${MAX_ROLLOUT_GROUPS:-100}"

if [[ -v PIPELINE_STAGE || -v ALLOW_FAILED_CALIBRATION_DIAGNOSTIC || -v DEVELOPMENT_REPORT || -v CALIBRATION_REPORT || -v MAX_OPTIMIZER_STEPS ]]; then
  echo "legacy calibration, pipeline-stage, and optimizer-step environment variables are no longer accepted" >&2
  exit 2
fi
if [[ "$VARIANT" != "A" ]]; then
  echo "at-risk GRPO-Open is authorized only for Variant A" >&2
  exit 2
fi
if [[ "$RUN_MODE" != "smoke" ]]; then
  echo "at-risk GRPO-Open requires RUN_MODE=smoke" >&2
  exit 2
fi
if [[ ! "$MAX_ROLLOUT_GROUPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_ROLLOUT_GROUPS must be a positive integer" >&2
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

SOURCE_CHECKPOINT_DIR="$(cd "$(dirname "$SOURCE_CHECKPOINT")" && pwd)"
SOURCE_RUN_ROOT="$(dirname "$SOURCE_CHECKPOINT_DIR")"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${SOURCE_RUN_ROOT}/grpo_open}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ARTIFACT_ROOT}"
LOG_ROOT="${LOG_ROOT:-${ARTIFACT_ROOT}/logs}"

echo "[GRPO] application=stage2_grpo_open_application_v1 reward_domain=tau_d"
echo "[GRPO] run_mode=$RUN_MODE variant=$VARIANT max_rollout_groups=$MAX_ROLLOUT_GROUPS"
echo "[GRPO] output_root=$OUTPUT_ROOT"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
"$PYTHON_BIN" -m train.train_bev_joint_grpo_online \
  --config "$CONFIG" \
  --variant "$VARIANT" \
  --run-mode "$RUN_MODE" \
  --source-checkpoint "$SOURCE_CHECKPOINT" \
  --output-root "$OUTPUT_ROOT" \
  --max-rollout-groups "$MAX_ROLLOUT_GROUPS" \
  2>&1 | tee "${LOG_ROOT}/grpo-open-training.log"
