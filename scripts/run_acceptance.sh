#!/usr/bin/env bash
# Acceptance test runner — runs all phase acceptance tests in dependency order.
# Usage:
#   bash scripts/run_acceptance.sh              # run all phases
#   bash scripts/run_acceptance.sh 1            # run Phase 1 only
#   bash scripts/run_acceptance.sh 1 2 3        # run Phases 1, 2, 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-python}"
PYTEST="${PYTHON} -m pytest"

PHASES=("${@:-0 1 2 3 4 5 6 7}")

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TOTAL_PASS=0
TOTAL_FAIL=0

for phase in ${PHASES[@]}; do
    pattern="${REPO_ROOT}/tests/acceptance/test_phase${phase}*.py"
    files=$(ls $pattern 2>/dev/null || true)
    if [ -z "$files" ]; then
        echo -e "${YELLOW}[SKIP] Phase ${phase}: no acceptance tests found${NC}"
        continue
    fi

    echo ""
    echo "================================================================"
    echo -e "${YELLOW}  Phase ${phase} Acceptance Tests${NC}"
    echo "================================================================"

    for test_file in $files; do
        test_name=$(basename "$test_file" .py)
        echo -e "\n--- ${test_name} ---"
        if $PYTEST "$test_file" -v --tb=short 2>&1; then
            echo -e "${GREEN}[PASS] ${test_name}${NC}"
            TOTAL_PASS=$((TOTAL_PASS + 1))
        else
            echo -e "${RED}[FAIL] ${test_name}${NC}"
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
        fi
    done
done

echo ""
echo "================================================================"
echo "  Acceptance Summary"
echo "================================================================"
echo -e "  ${GREEN}PASSED: ${TOTAL_PASS}${NC}"
echo -e "  ${RED}FAILED: ${TOTAL_FAIL}${NC}"
echo "================================================================"

if [ $TOTAL_FAIL -gt 0 ]; then
    exit 1
fi
