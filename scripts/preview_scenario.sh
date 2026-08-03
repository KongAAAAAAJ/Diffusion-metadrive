#!/usr/bin/env bash
# Usage:
#   ./scripts/preview_scenario.sh --scenario-id S6_background_merge_in --num-episodes 3 --num-agents 4
#   ./scripts/preview_scenario.sh --all-scenarios --num-agents 3 --planning-policy lattice --control-policy pid
#
# Thin wrapper around:
#   python -m evaluation.preview_and_evaluation
#
# This script only prepares the repo environment:
#   - resolves REPO_ROOT
#   - sets PYTHONPATH
#   - selects PYTHON_BIN
# Then it forwards all CLI arguments unchanged to the Python entrypoint.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

echo "=== Preview Scenario Wrapper ==="
echo "--- repo: ${REPO_ROOT} ---"
echo "--- python: ${PYTHON_BIN} ---"

"${PYTHON_BIN}" -m evaluation.preview_and_evaluation \
    --scenario-id S5_hard_brake_lead \
    --num-agents 3 \
    --decision-policy rule_maker \
    --planning-policy lattice \
    --control-policy adaptive \
    --num-episodes 3 \
    --evaluate \
    "$@"

