#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/riskentry_bundle_v2_real_smoke_v1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
exec /home/kong/anaconda3/envs/meta_drive/bin/python \
  -m expert_dataset.joint_risk_bundle_v2_real \
  --output-root "${OUTPUT_ROOT}" \
  --max-attempts "${MAX_ATTEMPTS}"
