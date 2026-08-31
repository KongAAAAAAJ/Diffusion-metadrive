#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/train/bev_joint_grpo.yaml}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/outputs/bev_diffusion_stage1/run_3/grpo_open}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ARTIFACT_ROOT}"

if (($# != 0)); then
  echo "launcher arguments are not accepted; set CONFIG and OUTPUT_ROOT in the environment" >&2
  exit 2
fi
if [[ -v VARIANT || -v RUN_MODE || -v SOURCE_CHECKPOINT || -v MAX_ROLLOUT_GROUPS ]]; then
  echo "VARIANT, RUN_MODE, SOURCE_CHECKPOINT, and MAX_ROLLOUT_GROUPS are no longer accepted; set run.variant, run.run_mode, run.source_checkpoint, and online.total_rollout_groups in the YAML config" >&2
  exit 2
fi
if [[ -v LOG_ROOT ]]; then
  echo "LOG_ROOT is no longer accepted; each run writes OUTPUT_ROOT/run_N/training.log" >&2
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

mkdir -p "$OUTPUT_ROOT"

max_run_index=0
for existing_run in "$OUTPUT_ROOT"/run_*; do
  [[ -d "$existing_run" ]] || continue
  run_name="${existing_run##*/}"
  if [[ "$run_name" =~ ^run_([0-9]+)$ ]]; then
    run_index=$((10#${BASH_REMATCH[1]}))
    if ((run_index > max_run_index)); then
      max_run_index=$run_index
    fi
  fi
done

candidate_index=$((max_run_index + 1))
while true; do
  RUN_DIR="$OUTPUT_ROOT/run_${candidate_index}"
  if mkdir "$RUN_DIR" 2>/dev/null; then
    break
  fi
  if [[ -e "$RUN_DIR" ]]; then
    ((candidate_index += 1))
    continue
  fi
  echo "unable to create GRPO run directory: $RUN_DIR" >&2
  exit 1
done
RUN_LOG="$RUN_DIR/training.log"

set +e
{
  echo "[GRPO] application=stage2_grpo_open_application_v1 reward_domain=tau_d"
  echo "[GRPO] config=$CONFIG"
  echo "[GRPO] output_root=$OUTPUT_ROOT"
  echo "[GRPO] run_dir=$RUN_DIR"
  "$PYTHON_BIN" -m train.train_bev_joint_grpo_online \
    --config "$CONFIG" \
    --run-dir "$RUN_DIR"
} 2>&1 | tee "$RUN_LOG"
pipeline_status=("${PIPESTATUS[@]}")
python_status="${pipeline_status[0]}"
tee_status="${pipeline_status[1]}"

printf '[GRPO] python_exit_status=%s tee_exit_status=%s\n' \
  "$python_status" "$tee_status" | tee -a "$RUN_LOG"
status_pipeline=("${PIPESTATUS[@]}")
status_tee_status="${status_pipeline[1]}"
set -e

if ((python_status != 0)); then
  exit "$python_status"
fi
if ((tee_status != 0)); then
  exit "$tee_status"
fi
exit "$status_tee_status"
