#!/usr/bin/env bash
set -euo pipefail  # 安全设置：-e 失败时退出，-u 未定义变量时报错，-o pipefail 管道中任何命令失败都会导致整个管道失败

PYTHON_BIN="${PYTHON_BIN:-python}"

cmd=(
    "${PYTHON_BIN}" -m train.train_selector
    --config configs/train/selector.yaml
    --total-env-steps 20000
    --pretrained-ckpt /media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_2/checkpoints/diffusion-epoch=78.ckpt
)

"${cmd[@]}"
