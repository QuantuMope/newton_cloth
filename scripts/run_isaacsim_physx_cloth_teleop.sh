#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

. "${HOME}/.vnc/display2-glx-env"

OMNI_KIT_ACCEPT_EULA=YES /home/horizon/isaacsim_env/bin/python \
  isaacsim_newton_scene.py \
  --teleop \
  --width 1280 \
  --height 900 \
  --batch-size 1 \
  --physics-backend physx \
  --cloth-mode physical \
  --include-gripper-actions \
  --arm-random-scale 0.12 \
  --gripper-random-scale 0.5 \
  --physics-steps-per-action 4 \
  --fps 30
