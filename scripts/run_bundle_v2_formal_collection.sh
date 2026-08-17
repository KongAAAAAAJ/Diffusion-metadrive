#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/dataset/data_collect_candidate_v3_formal50k.yaml}"

ARGS=(--config "${CONFIG}")
if [[ -n "${MAX_EPISODES:-}" ]]; then
  ARGS+=(--max-episodes "${MAX_EPISODES}")
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HOME="${HOME_OVERRIDE:-/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
if [[ -z "${LOG_FILE:-}" ]] && [[ "$(basename "${CONFIG}")" == "data_collect_candidate_v3_s5_release20.yaml" ]]; then
  LOG_DIR="/media/kong/Elements_SE/Diffusion_Data/collection_logs/bev_joint_risk_candidate_v3_s5_release20_v1"
  mkdir -p "${LOG_DIR}"
  LOG_FILE="${LOG_DIR}/collection_$(date +%Y%m%d_%H%M%S).log"
fi
if [[ -n "${LOG_FILE:-}" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  /home/kong/anaconda3/envs/meta_drive/bin/python \
    -m expert_dataset.run_joint_bev_collection "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
  exit "${PIPESTATUS[0]}"
fi
exec /home/kong/anaconda3/envs/meta_drive/bin/python \
  -m expert_dataset.run_joint_bev_collection "${ARGS[@]}"
