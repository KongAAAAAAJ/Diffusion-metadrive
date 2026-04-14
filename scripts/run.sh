#!/bin/bash

# 依次运行数据处理和训练脚本
set -e

PYTHON_BIN="${PYTHON_BIN:-/home/kong/anaconda3/envs/meta_drive/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/run_data_pipeline.sh"
"$SCRIPT_DIR/run_train.sh"
