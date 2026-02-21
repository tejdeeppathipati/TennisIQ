#!/usr/bin/env bash
set -euo pipefail

python3 -m tennisiq.cv.ball.train \
  --dataset-root data/datasets/balltracking \
  --raw-data-dir data/datasets/balltracking/images \
  --prepare-if-missing \
  "$@"
