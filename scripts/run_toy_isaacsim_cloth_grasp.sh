#!/usr/bin/env bash
set -euo pipefail

cd /home/horizon/newton_cloth
OMNI_KIT_ACCEPT_EULA=YES /home/horizon/isaacsim_env/bin/python \
  /home/horizon/newton_cloth/toy_isaacsim_cloth_grasp.py \
  --no-headless \
  --viewport-camera \
  --record \
  --width 640 \
  --height 480 \
  "$@"
