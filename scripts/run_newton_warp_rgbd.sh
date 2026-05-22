#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

/home/horizon/newton/.venv/bin/python record_cloth_cameras.py
