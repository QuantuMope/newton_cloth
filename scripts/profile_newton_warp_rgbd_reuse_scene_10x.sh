#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

profile_dir="recordings/profiles/newton_warp_rgbd_reuse_scene"
mkdir -p "$profile_dir"

/home/horizon/newton/.venv/bin/python \
  record_cloth_cameras.py \
  --steps 10 \
  --fps 5 \
  --width 640 \
  --height 480 \
  --seed 7 \
  --initial-pose isaac \
  --action-dt 0.1 \
  --sim-substeps 12 \
  --use-joint-position-targets \
  --render-order tiled \
  --profile-repeats 10 \
  --no-compose-video \
  --profile-json "$profile_dir/reuse_profile.json"
