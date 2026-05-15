#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/train/selected_refine_grpo.yaml}"

# cls-GRPO ckpt: pass via env-var or derived from PLAN_CLS_GRPO_RUN_DIR
PLAN_CLS_GRPO_RUN_DIR="${PLAN_CLS_GRPO_RUN_DIR:-}"
CLS_GRPO_CKPT_DIR="${CLS_GRPO_CKPT_DIR:-}"
CLS_GRPO_FULL_CKPT="${CLS_GRPO_FULL_CKPT:-/media/kong/Elements_SE/Diffusion_Data/outputs/plan_cls_grpo/run_14/checkpoints/step_0025000_score_8.5445/full_platoon_grpo.ckpt}"

if [[ -z "${CLS_GRPO_CKPT_DIR}" && -z "${CLS_GRPO_FULL_CKPT}" && -n "${PLAN_CLS_GRPO_RUN_DIR}" ]]; then
    CLS_GRPO_CKPT_DIR="${PLAN_CLS_GRPO_RUN_DIR%/}/checkpoints/final"
fi

if [[ -n "${CLS_GRPO_CKPT_DIR}" ]]; then
    echo "[refine-grpo] cls GRPO ckpt dir  : ${CLS_GRPO_CKPT_DIR}"
elif [[ -n "${CLS_GRPO_FULL_CKPT}" ]]; then
    echo "[refine-grpo] cls GRPO full ckpt : ${CLS_GRPO_FULL_CKPT}"
else
    echo "[refine-grpo] WARNING: no cls GRPO weights provided; using classifier from PRETRAINED_CKPT in yaml."
fi

cd "${REPO_ROOT}"

cmd=("${PYTHON_BIN}" -m train.train_selected_refine_grpo --config "${CONFIG}")

[[ -n "${CLS_GRPO_CKPT_DIR}" ]]  && cmd+=(--cls-grpo-ckpt-dir  "${CLS_GRPO_CKPT_DIR}")
[[ -n "${CLS_GRPO_FULL_CKPT}" ]] && cmd+=(--cls-grpo-full-ckpt "${CLS_GRPO_FULL_CKPT}")
[[ -n "${SCENARIO_IDS:-}" ]]     && cmd+=(--scenario-ids       "${SCENARIO_IDS}")

exec "${cmd[@]}"

