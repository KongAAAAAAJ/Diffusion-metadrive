#!/usr/bin/env bash
# Joint-first simulator-GT BEV expert collection.
# Optional overrides:
#   DATASET_ROOT=/tmp/bundle/platoon_joint_bev
#   SIDECAR_ROOT=/tmp/bundle/riskentry_actor_sidecar TARGET_JOINT_STEPS=10 MAX_EPISODES=2
#   MAX_EPISODE_STEPS=20 RESUME=0 bash scripts/run_multiData_collection.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
DATASET_CONFIG_PATH="${DATASET_CONFIG_PATH:-${REPO_ROOT}/configs/dataset/data_collect.yaml}"

ARGS=(
    -m expert_dataset.run_joint_bev_collection
    --config "${DATASET_CONFIG_PATH}"
)

if [[ -n "${DATASET_ROOT:-}" ]]; then
    ARGS+=(--dataset-root "${DATASET_ROOT}")
fi
if [[ -n "${SIDECAR_ROOT:-}" ]]; then
    ARGS+=(--sidecar-root "${SIDECAR_ROOT}")
fi
if [[ -n "${TARGET_JOINT_STEPS:-}" ]]; then
    ARGS+=(--target-joint-steps "${TARGET_JOINT_STEPS}")
fi
if [[ -n "${MAX_EPISODES:-}" ]]; then
    ARGS+=(--max-episodes "${MAX_EPISODES}")
fi
if [[ -n "${MAX_EPISODE_STEPS:-}" ]]; then
    ARGS+=(--max-episode-steps "${MAX_EPISODE_STEPS}")
fi
if [[ -n "${RESUME:-}" ]]; then
    ARGS+=(--resume "${RESUME}")
fi

exec "${PYTHON_BIN}" "${ARGS[@]}"
