#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

profile_dir="recordings/profiles/isaacsim_physx_cloth_rgbd"
mkdir -p "$profile_dir"

for run_index in $(seq 0 9); do
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
    --video-path recordings/isaacsim_newton_scene/isaacsim_physx_cloth_random_arm_rgbd.mp4 \
    --profile-json "$profile_dir/run_${run_index}.json"
done
