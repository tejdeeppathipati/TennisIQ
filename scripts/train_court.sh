#!/usr/bin/env bash
set -euo pipefail

python3 -m tennisiq.cv.court.train \
  --data-root data/datasets/court_identification \
  "$@"
