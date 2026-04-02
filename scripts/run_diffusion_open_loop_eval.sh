#!/usr/bin/env bash
set -euo pipefail

/home/kong/anaconda3/envs/meta_drive/bin/python -m metadrive.policy.diffusion_policy.eval_transfuser_open_loop \
    --checkpoint /media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_4/checkpoints/diffusion-epoch=50.ckpt \
    --dataset-root /media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metaData_strong_idm_single_preprocessed \
    --dataset-format auto \
    --split val \
    --num-samples 500 \
    --batch-size 8 \
    --num-workers 0 \
    --device auto \
    --plan-anchor-path metadrive/exp_dataset/anchors.npy \
    --output-dir /media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/eval/open \
    --save-images 1 \
    --save-json 1 \
    --save-trajectory-plots 1 \
    --save-csv 1 \
    --overlay-all-anchors 1 \
    --mode-focus 7
