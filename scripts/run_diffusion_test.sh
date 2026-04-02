#!/usr/bin/env bash
set -euo pipefail

/home/kong/anaconda3/envs/meta_drive/bin/python -m metadrive.policy.diffusion_policy.test_transfuser_policy \
  --checkpoint /media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_4/checkpoints/diffusion-epoch=50.ckpt \
  --episodes 5 \
  --render 0 \
  --controller-type stabilized \
  --print-trajectory-debug 1 \
  --image-on-cuda 0 \
  --save-camera-interval 0 \
  --camera-output-dir /media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/eval/cameras \
  --save-3d-video 0 \
  --save-2d-video 1 \
  --save-trajectory-plot 1 \
  --save-step-images 1 \
  --step-image-interval 1 \
  --output-dir /media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/eval/closed \
  --video-fps 10 \
  --topdown-camera-height 80.0
