#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <run_dir> <annotations_jsonl> [baseline_json]"
  exit 1
fi

RUN_DIR="$1"
ANN="$2"
BASELINE="${3:-}"
RUN_ID="$(basename "$RUN_DIR")"
OUT_DIR="eval/predictions/${RUN_ID}"
mkdir -p "$OUT_DIR"

CMD=(python3 eval/run_eval.py --run-dir "$RUN_DIR" --annotations "$ANN" --output "$OUT_DIR/metrics.json")
if [ -n "$BASELINE" ]; then
  CMD+=(--baseline "$BASELINE")
fi

"${CMD[@]}"

echo "Wrote metrics to $OUT_DIR/metrics.json"
