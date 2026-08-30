#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/train/bev_joint_grpo.yaml}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/outputs/bev_diffusion_stage1/run_3/grpo_open}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ARTIFACT_ROOT}"
LOG_ROOT="${LOG_ROOT:-${ARTIFACT_ROOT}/logs}"

if [[ -v VARIANT || -v RUN_MODE || -v SOURCE_CHECKPOINT || -v MAX_ROLLOUT_GROUPS ]]; then
  echo "VARIANT, RUN_MODE, SOURCE_CHECKPOINT, and MAX_ROLLOUT_GROUPS are no longer accepted; set run.variant, run.run_mode, run.source_checkpoint, and online.total_rollout_groups in the YAML config" >&2
  exit 2
fi
if [[ -v PIPELINE_STAGE || -v ALLOW_FAILED_CALIBRATION_DIAGNOSTIC || -v DEVELOPMENT_REPORT || -v CALIBRATION_REPORT || -v MAX_OPTIMIZER_STEPS ]]; then
  echo "legacy calibration, pipeline-stage, and optimizer-step environment variables are no longer accepted" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "GRPO config does not exist: $CONFIG" >&2
  exit 2
fi

echo "[GRPO] application=stage2_grpo_open_application_v1 reward_domain=tau_d"
echo "[GRPO] config=$CONFIG"
echo "[GRPO] output_root=$OUTPUT_ROOT"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
"$PYTHON_BIN" -m train.train_bev_joint_grpo_online \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  2>&1 | tee "${LOG_ROOT}/grpo-open-training.log"
