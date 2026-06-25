#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/train/refine_grpo.yaml}"

cd "${REPO_ROOT}"

cmd=("${PYTHON_BIN}" -m train.train_refine_grpo --config "${CONFIG}")

exec "${cmd[@]}"
