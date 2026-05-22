#!/usr/bin/env bash
set -euo pipefail

cd /home/horizon/newton_cloth
OMNI_KIT_ACCEPT_EULA=YES /home/horizon/isaacsim_env/bin/python \
  /home/horizon/newton_cloth/toy_isaacsim_cloth_grasp.py \
  --headless \
  --record \
  --steps 360 \
  --lift-steps 220 \
  --width 640 \
  --height 480 \
  "$@"
