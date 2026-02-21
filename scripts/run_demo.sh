#!/usr/bin/env bash
set -euo pipefail

RUN_ID="$(date +%Y-%m-%d_%H%M)"
python3 -m tennisiq.pipeline.run_all \
  --video data/raw/input.mp4 \
  --court-model checkpoints/court/best.pt \
  --ball-model checkpoints/ball/best.pt \
  --output "outputs/runs/${RUN_ID}" \
  "$@"
