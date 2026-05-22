#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OMNI_KIT_ACCEPT_EULA=YES /home/horizon/isaacsim_env/bin/python \
  isaacsim_newton_scene.py \
  --headless \
  --record \
  --steps 10 \
  --width 640 \
  --height 480 \
  --physics-backend physx \
  --cloth-mode physical \
  --random-arm-actions \
  --include-gripper-actions \
  --arm-random-scale 0.12 \
  --gripper-random-scale 0.5 \
  --random-seed 7 \
  --physics-steps-per-action 12 \
  --video-path recordings/isaacsim_newton_scene/isaacsim_physx_cloth_random_arm_rgbd.mp4
