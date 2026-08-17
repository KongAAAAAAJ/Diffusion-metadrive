#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/dataset/data_collect_candidate_v3_formal50k.yaml}"

ARGS=(--config "${CONFIG}")
if [[ "${STOP_AFTER_INITIAL_GATE:-0}" != "0" ]] && [[ "${STOP_AFTER_INITIAL_GATE:-0}" != "1" ]]; then
  echo "STOP_AFTER_INITIAL_GATE must be 0 or 1" >&2
  exit 2
fi
export STOP_AFTER_INITIAL_GATE="${STOP_AFTER_INITIAL_GATE:-0}"
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
CONFIG_BASENAME="$(basename "${CONFIG}")"
if [[ -z "${LOG_FILE:-}" ]] && [[ "${CONFIG_BASENAME}" =~ ^data_collect_candidate_v[34]_s5_release20\.yaml$ ]]; then
  BUNDLE_VERSION="${CONFIG_BASENAME#data_collect_}"
  BUNDLE_VERSION="${BUNDLE_VERSION%.yaml}"
  LOG_DIR="/media/kong/Elements_SE/Diffusion_Data/collection_logs/bev_joint_risk_${BUNDLE_VERSION}_v1"
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
