#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/dataset/data_collect_candidate_v4_s5_release30.yaml"
BUNDLE_ROOT="/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/bev_joint_risk_candidate_v4_s5_release30_v1"
LOG_DIR="/media/kong/Elements_SE/Diffusion_Data/collection_logs/bev_joint_risk_candidate_v4_s5_release30_v1"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/collection_$(date +%Y%m%d_%H%M%S).log}"

if [[ -e "${BUNDLE_ROOT}" ]] && [[ ! -f "${BUNDLE_ROOT}/dataset_bundle_manifest.json" ]]; then
  echo "Refusing an unknown pre-existing release30 root: ${BUNDLE_ROOT}" >&2
  exit 2
fi

echo "S5 release30 config: ${CONFIG}"
echo "S5 release30 bundle: ${BUNDLE_ROOT}"
echo "S5 release30 log: ${LOG_FILE}"

CONFIG="${CONFIG}" \
LOG_FILE="${LOG_FILE}" \
STOP_AFTER_INITIAL_GATE=0 \
ALLOW_FAILED_INITIAL_GATE_CONTINUE=0 \
bash "${REPO_ROOT}/scripts/run_bundle_v2_formal_collection.sh"
