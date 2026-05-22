#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

output_dir="recordings/newton_batch4_reset_rgbd"
mkdir -p "$output_dir"

/home/horizon/newton/.venv/bin/python \
  record_cloth_cameras.py \
  --steps 10 \
  --fps 5 \
  --width 640 \
  --height 480 \
  --batch-size 4 \
  --env-spacing 2.0 \
  --seed 7 \
  --reset-every-steps 2 \
  --record-reset-frames \
  --reset-arm-zero \
  --profile-repeats 1 \
  --compose-video \
  --output-dir "$output_dir" \
  --profile-json "$output_dir/profile.json"
