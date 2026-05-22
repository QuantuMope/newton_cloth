#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

profile_dir="recordings/profiles/isaacsim_physx_cloth_rgbd_batch10_tiled_reuse_scene"
mkdir -p "$profile_dir"

OMNI_KIT_ACCEPT_EULA=YES /home/horizon/isaacsim_env/bin/python \
  isaacsim_newton_scene.py \
  --headless \
  --record \
  --steps 10 \
  --width 640 \
  --height 480 \
  --batch-size 10 \
  --env-spacing 2.0 \
  --camera-render-mode tiled \
  --physics-backend physx \
  --cloth-mode physical \
  --random-arm-actions \
  --action-sampling newton \
  --newton-random-joint-scale 0.18 \
  --arm-random-scale 0.12 \
  --gripper-random-scale 0.5 \
  --random-seed 7 \
  --physics-steps-per-action 12 \
  --profile-repeats 10 \
  --no-write-frames \
  --no-compose-video \
  --no-log-random-actions \
  --video-path recordings/isaacsim_newton_scene/isaacsim_physx_cloth_batch10_tiled_rgbd.mp4 \
  --profile-json "$profile_dir/reuse_profile.json"
