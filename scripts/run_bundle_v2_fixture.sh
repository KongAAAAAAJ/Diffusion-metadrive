#!/usr/bin/env bash
# Generate a diagnostic bundle-v2 fixture. Existing output is never overwritten.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/metadrive-joint-risk-bundle-v2-fixture}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" -m expert_dataset.joint_risk_bundle_v2_fixture \
  --output-root "${OUTPUT_ROOT}"
