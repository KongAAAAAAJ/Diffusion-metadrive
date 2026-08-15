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
exec /home/kong/anaconda3/envs/meta_drive/bin/python \
  -m expert_dataset.run_joint_bev_collection "${ARGS[@]}"
