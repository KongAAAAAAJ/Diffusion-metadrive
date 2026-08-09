#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/dataset/data_collect_bundle_v2_formal_path_pilot.yaml}"

ARGS=(--config "${CONFIG}")
if [[ -n "${MAX_NEW_EPISODES:-}" ]]; then
  ARGS+=(--max-new-episodes "${MAX_NEW_EPISODES}")
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HOME="${HOME_OVERRIDE:-/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"
exec /home/kong/anaconda3/envs/meta_drive/bin/python \
  -m expert_dataset.joint_risk_bundle_v2_formal "${ARGS[@]}"
