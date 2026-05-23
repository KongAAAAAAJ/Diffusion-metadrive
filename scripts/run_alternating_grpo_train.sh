#!/usr/bin/env bash
# Alternating GRPO training: cls_head ↔ trajectory_head cycles.
# Usage: bash scripts/run_alternating_grpo_train.sh
# Override any field via env-vars, e.g.:
#   N_CYCLES=1 CLS_STEPS=200 REFINE_STEPS=200 bash scripts/run_alternating_grpo_train.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/train/alternating_grpo.yaml}"

resolve_best_or_final_ckpt() {
    local run_dir="$1"
    local ckpt_name="$2"
    local best_path=""
    local best_score=""
    local ckpt_dir score

    shopt -s nullglob
    for ckpt_dir in "${run_dir%/}"/checkpoints/step_*_score_*; do
        [[ -d "${ckpt_dir}" && -f "${ckpt_dir}/${ckpt_name}" ]] || continue
        score="${ckpt_dir##*_score_}"
        if [[ -z "${best_score}" ]] || awk "BEGIN { exit !(${score} > ${best_score}) }"; then
            best_score="${score}"
            best_path="${ckpt_dir}/${ckpt_name}"
        fi
    done
    shopt -u nullglob

    if [[ -n "${best_path}" ]]; then
        echo "${best_path}"
        return 0
    fi

    local final_path="${run_dir%/}/checkpoints/final/${ckpt_name}"
    if [[ -f "${final_path}" ]]; then
        echo "${final_path}"
        return 0
    fi

    echo "[ERROR] no checkpoint '${ckpt_name}' found under ${run_dir}/checkpoints/{step_*_score_*,final}" >&2
    return 1
}

if [[ "${1:-}" == "--resolve-ckpt" ]]; then
    if [[ $# -ne 3 ]]; then
        echo "Usage: $0 --resolve-ckpt <run_dir> <ckpt_name>" >&2
        exit 2
    fi
    resolve_best_or_final_ckpt "$2" "$3"
    exit $?
fi

# ── read values from yaml (env-vars override) ────────────────────────────────
_yaml() {
    "${PYTHON_BIN}" -c "
import yaml, sys
d = yaml.safe_load(open('${CONFIG}')) or {}
print(d.get('$1', ''))
"
}

N_CYCLES="${N_CYCLES:-$(_yaml n_cycles)}"
CLS_STEPS="${CLS_STEPS:-$(_yaml cls_steps_per_cycle)}"
REFINE_STEPS="${REFINE_STEPS:-$(_yaml refine_steps_per_cycle)}"
CLS_CONFIG="${CLS_CONFIG:-$(_yaml cls_config)}"
REFINE_CONFIG="${REFINE_CONFIG:-$(_yaml refine_config)}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-$(_yaml pretrained_ckpt)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(_yaml output_root)}"

echo "[alternating] config        : ${CONFIG}"
echo "[alternating] n_cycles      : ${N_CYCLES}"
echo "[alternating] cls_steps     : ${CLS_STEPS}"
echo "[alternating] refine_steps  : ${REFINE_STEPS}"
echo "[alternating] output_root   : ${OUTPUT_ROOT}"
echo "[alternating] pretrained    : ${PRETRAINED_CKPT}"

cd "${REPO_ROOT}"

current_ckpt=""   # empty → use PRETRAINED_CKPT on first cycle
SUMMARY_JSON="${OUTPUT_ROOT}/alternating_grpo_summary.json"
# initialise summary JSON
mkdir -p "${OUTPUT_ROOT}"
"${PYTHON_BIN}" -c "
import json, pathlib, time
d = {
    'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'config': '${CONFIG}',
    'n_cycles': ${N_CYCLES},
    'cls_steps_per_cycle': ${CLS_STEPS},
    'refine_steps_per_cycle': ${REFINE_STEPS},
    'pretrained_ckpt': '${PRETRAINED_CKPT}',
    'cycles': [],
}
pathlib.Path('${SUMMARY_JSON}').write_text(json.dumps(d, indent=2))
"

for cycle in $(seq 1 "${N_CYCLES}"); do
    echo ""
    echo "=========================================="
    echo "  Cycle ${cycle} / ${N_CYCLES}"
    echo "=========================================="

    # ── Stage 1: classification head fine-tuning ─────────────────────────────
    CLS_OUT="${OUTPUT_ROOT}/cycle_${cycle}/cls_grpo"
    echo "[alternating] Stage 1 — cls_grpo  → ${CLS_OUT}"

    if [[ -z "${current_ckpt}" ]]; then
        CLS_CKPT_ARG="--pretrained-ckpt ${PRETRAINED_CKPT}"
    else
        CLS_CKPT_ARG="--full-platoon-ckpt ${current_ckpt}"
    fi

    "${PYTHON_BIN}" -m train.train_plan_cls_grpo \
        --config "${CLS_CONFIG}" \
        ${CLS_CKPT_ARG} \
        --output-root "${CLS_OUT}" \
        --total-env-steps "${CLS_STEPS}" \
        ${SCENARIO_IDS:+--scenario-ids "${SCENARIO_IDS}"}

    # locate the final ckpt produced by this run
    cls_run_dir=$(ls -td "${CLS_OUT}"/run_*/ 2>/dev/null | head -1)
    if [[ -z "${cls_run_dir}" ]]; then
        echo "[ERROR] no run directory found under ${CLS_OUT}" >&2
        exit 1
    fi
    current_ckpt="$(resolve_best_or_final_ckpt "${cls_run_dir}" "full_platoon_grpo.ckpt")"
    cls_ckpt="${current_ckpt}"
    echo "[alternating] cls  ckpt : ${current_ckpt}"

    # ── Stage 2: trajectory head fine-tuning ─────────────────────────────────
    REFINE_OUT="${OUTPUT_ROOT}/cycle_${cycle}/refine_grpo"
    echo "[alternating] Stage 2 — refine_grpo → ${REFINE_OUT}"

    "${PYTHON_BIN}" -m train.train_selected_refine_grpo \
        --config "${REFINE_CONFIG}" \
        --cls-grpo-full-ckpt "${current_ckpt}" \
        --output-root "${REFINE_OUT}" \
        --total-env-steps "${REFINE_STEPS}" \
        ${SCENARIO_IDS:+--scenario-ids "${SCENARIO_IDS}"}

    # locate the final ckpt — becomes input for next cycle's cls stage
    refine_run_dir=$(ls -td "${REFINE_OUT}"/run_*/ 2>/dev/null | head -1)
    if [[ -z "${refine_run_dir}" ]]; then
        echo "[ERROR] no run directory found under ${REFINE_OUT}" >&2
        exit 1
    fi
    current_ckpt="$(resolve_best_or_final_ckpt "${refine_run_dir}" "full_platoon_refine_grpo.ckpt")"
    echo "[alternating] refine ckpt : ${current_ckpt}"

    # ── append cycle record to summary JSON ──────────────────────────────────
    cls_run_name=$(basename "${cls_run_dir%/}")
    refine_run_name=$(basename "${refine_run_dir%/}")
    "${PYTHON_BIN}" -c "
import json, pathlib
p = pathlib.Path('${SUMMARY_JSON}')
d = json.loads(p.read_text())
d['cycles'].append({
    'cycle': ${cycle},
    'cls_grpo':    {'run': '${cls_run_name}',    'ckpt': '${cls_ckpt}'},
    'refine_grpo': {'run': '${refine_run_name}', 'ckpt': '${current_ckpt}'},
})
p.write_text(json.dumps(d, indent=2))
"
done

# ── finalise summary JSON ────────────────────────────────────────────────────
"${PYTHON_BIN}" -c "
import json, pathlib, time
p = pathlib.Path('${SUMMARY_JSON}')
d = json.loads(p.read_text())
d['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
d['final_ckpt']  = '${current_ckpt}'
p.write_text(json.dumps(d, indent=2))
"

echo ""
echo "=========================================="
echo "  Alternating GRPO training complete"
echo "  Final ckpt : ${current_ckpt}"
echo "  Summary    : ${SUMMARY_JSON}"
echo "=========================================="
