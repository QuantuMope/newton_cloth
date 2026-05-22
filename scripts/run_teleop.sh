#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

/home/horizon/newton/.venv/bin/python cloth_teleop.py \
  --grasp-mode constraint \
  --cloth-asset grid
