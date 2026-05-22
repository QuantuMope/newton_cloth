#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

profile_dir="recordings/isaacsim_batch4_reset_rgbd"
mkdir -p "$profile_dir"

OMNI_KIT_ACCEPT_EULA=YES /home/horizon/isaacsim_env/bin/python \
  isaacsim_newton_scene.py \
  --headless \
  --record \
  --steps 10 \
  --fps 5 \
  --width 640 \
  --height 480 \
  --batch-size 4 \
  --env-spacing 2.0 \
  --camera-render-mode tiled \
  --physics-backend physx \
  --cloth-mode physical \
  --random-arm-actions \
  --include-gripper-actions \
  --arm-random-scale 0.12 \
  --gripper-random-scale 0.5 \
  --random-seed 7 \
  --reset-every-steps 2 \
  --record-reset-frames \
  --reset-arm-zero \
  --physics-steps-per-action 12 \
  --profile-repeats 1 \
  --compose-video \
  --no-log-random-actions \
  --output-root "$profile_dir" \
  --video-path "$profile_dir/isaacsim_batch4_reset_rgbd.mp4" \
  --profile-json "$profile_dir/profile.json"
