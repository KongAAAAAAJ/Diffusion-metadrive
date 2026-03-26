#!/usr/bin/env bash
set -euo pipefail

# Optional offline feature caching only. Training does not require this step by default.

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_ROOT="${INPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/1_metaData_idm_single}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/1_metaData_idm_single_preprocessed}"
MODEL_SIZE="${MODEL_SIZE:-small}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-dir}"

"${PYTHON_BIN}" -m metadrive.policy.diffusion_policy.preprocess_transfuser_dataset \
    --input-root "${INPUT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --output-format "${OUTPUT_FORMAT}" \
    --model-size "${MODEL_SIZE}"
