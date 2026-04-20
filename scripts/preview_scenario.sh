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

SCENARIO_ID="${1:-${SCENARIO_ID:-S4_curve_following}}"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
    shift
fi
NUM_EPISODES="${1:-${NUM_EPISODES:-3}}"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
    shift
fi
HEADING_UP="false"
MODE_GENERATE="${MODE_GENERATE:-0}"
MODE_FRAME_LIMIT="${MODE_FRAME_LIMIT:--1}"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --heading-up)
            HEADING_UP="true"
            shift
            ;;
        --mode-generate)
            MODE_GENERATE="1"
            shift
            ;;
        --mode-frame-limit)
            MODE_FRAME_LIMIT="${2:?missing value for --mode-frame-limit}"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/outputs/scenario/scenario_preview}"
DATASET_NAME="${SCENARIO_ID}"
DATASET_ROOT="${OUTPUT_ROOT}/${DATASET_NAME}"
VIDEO_DIR="${DATASET_ROOT}/reports/videos/${SCENARIO_ID}"
MODE_DIR="${DATASET_ROOT}/reports/mode_generate/${SCENARIO_ID}"
MANIFEST_PATH="${DATASET_ROOT}/reports/manifest.json"
DRY_RUN="false"
if [[ "$(basename "${PYTHON_BIN}")" == "echo" ]]; then
    DRY_RUN="true"
fi

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
        --start-seed 59 \
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
        --topdown-heading-up "${HEADING_UP}" \
        --mode-generate-enabled 1 \
        --mode-generate-frame-limit "${MODE_FRAME_LIMIT}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        break
    fi
done

echo ""
echo "=== Videos saved to: ${VIDEO_DIR} ==="
ls -lh "${VIDEO_DIR}" 2>/dev/null || echo "(no videos found — check logs above)"
if [[ "${MODE_GENERATE}" == "1" ]]; then
    echo ""
    echo "=== Mode overlays saved to: ${MODE_DIR} ==="
    ls -lh "${MODE_DIR}" 2>/dev/null || echo "(no mode overlays found — check logs above)"
fi
