#!/usr/bin/env bash
# Usage:
#   ./scripts/preview_scenario.sh S3_straight_following [NUM_EPISODES] [--heading-up]
#
# Runs the expert policy for one scenario at a time and saves top-down videos.
# Output goes to /tmp/scenario_preview/<SCENARIO_ID>/
# After collection, prints the video directory path.
#
# Options:
#   --heading-up    Enable camera rotation with ego vehicle heading (default: false)

set -euo pipefail

SCENARIO_ID="${1:-S9_narrow_channel_negotiation}"
NUM_EPISODES="${2:-3}"
HEADING_UP="false"

# Parse optional arguments
shift 2 || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --heading-up)
            HEADING_UP="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

PYTHON_BIN="/home/kong/anaconda3/envs/meta_drive/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="/media/kong/Elements_SE/Diffusion_Data/outputs/scenario/scenario_preview"
DATASET_NAME="${SCENARIO_ID}"
DATASET_ROOT="${OUTPUT_ROOT}/${DATASET_NAME}"
VIDEO_DIR="${DATASET_ROOT}/reports/videos/${SCENARIO_ID}"
MANIFEST_PATH="${DATASET_ROOT}/reports/manifest.json"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

mkdir -p "${OUTPUT_ROOT}"

count_videos() {
    if [[ -d "${VIDEO_DIR}" ]]; then
        find "${VIDEO_DIR}" -maxdepth 1 -type f -name '*.mp4' | wc -l
    else
        echo "0"
    fi
}

read_total_samples() {
    if [[ ! -f "${MANIFEST_PATH}" ]]; then
        echo "0"
        return
    fi
    "${PYTHON_BIN}" - <<'PY' "${MANIFEST_PATH}"
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
data = json.loads(manifest_path.read_text(encoding="utf-8"))
print(int(data.get("samples", 0)))
PY
}

echo "=== Preview: ${SCENARIO_ID} (${NUM_EPISODES} episodes) ==="

existing_videos="$(count_videos)"
target_videos="$(( existing_videos + NUM_EPISODES ))"

while true; do
    current_videos="$(count_videos)"
    if [[ "${current_videos}" -ge "${target_videos}" ]]; then
        break
    fi

    current_samples="$(read_total_samples)"
    next_target_samples="$(( current_samples + 1 ))"

    echo "--- collecting episode $(( current_videos - existing_videos + 1 ))/${NUM_EPISODES} (target_samples=${next_target_samples}) ---"
    "${PYTHON_BIN}" -m metadrive.exp_dataset.collect_expert \
        --target-samples "${next_target_samples}" \
        --output-root "${OUTPUT_ROOT}" \
        --dataset-name "${DATASET_NAME}" \
        --save-videos true \
        --expert-type idm \
        --start-seed 28 \
        --trajectory-correction-enabled true \
        --save-raw-trajectory false \
        --mode-classifier-version v1 \
        --trajectory-visualization-enabled false \
        --traffic-density-min 0.08 \
        --traffic-density-max 0.12 \
        --scenario-weights "{\"${SCENARIO_ID}\": 1.0}" \
        --use-hybrid-map true \
        --map-block-num 5 \
        --num-scenarios 1 \
        --topdown-heading-up "${HEADING_UP}"
done

echo ""
echo "=== Videos saved to: ${VIDEO_DIR} ==="
ls -lh "${VIDEO_DIR}" 2>/dev/null || echo "(no videos found — check logs above)"
