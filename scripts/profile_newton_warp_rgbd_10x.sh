#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

profile_dir="recordings/profiles/newton_warp_rgbd"
mkdir -p "$profile_dir"

for run_index in $(seq 0 9); do
  /home/horizon/newton/.venv/bin/python \
    record_cloth_cameras.py \
    --steps 10 \
    --fps 5 \
    --width 256 \
    --height 256 \
    --seed 7 \
    --profile-json "$profile_dir/run_${run_index}.json"
done
