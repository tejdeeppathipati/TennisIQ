# PROJECT.md

## Objective
Build a reliable, modular tennis analysis stack that can detect court structure, track the ball frame-by-frame, derive gameplay events, and export analytics-ready artifacts.

## Architecture Summary

- `tennisiq/cv/ball`: TrackNet-style ball model training + inference.
- `tennisiq/cv/court`: Court keypoint heatmap model training + inference.
- `tennisiq/geometry`: Court reference model + homography utilities.
- `tennisiq/tracking`: Gap filling and smoothing helpers.
- `tennisiq/events`: Bounce/hit/in-out/segmentation logic.
- `tennisiq/analytics`: Stats, heatmaps, insight summarization.
- `tennisiq/io`: Video read/write, JSON/JSONL export, schemas.
- `tennisiq/pipeline`: Ordered step runner from raw video to outputs.

## Pipeline Contract

Input:
- Video: `data/raw/input.mp4`
- Model weights:
  - `checkpoints/court/best.pt`
  - `checkpoints/ball/best.pt`

Output (under `outputs/runs/<run_id>/`):
- `frames.jsonl`
- `tracks.json`
- `points.json`
- `overlay.mp4`
- `insights.json`
- `clips/`

## Engineering Decisions

- Single canonical implementation: code only under `tennisiq/`.
- CPU/CUDA auto-device fallback in runtime paths.
- Model/data artifact directories are present but git-ignored.
- Scripts provide stable CLI entrypoints for setup, training, and demo.

## Current Status

- Refactor complete and legacy duplicate model trees removed.
- Checkpoint files validated against current model definitions.
- Repo layout is aligned for clean GitHub publishing.

## Next Improvements

- Add tests for dataset loaders and pipeline step contracts.
- Add CI for lint + import/compile checks.
- Add player detection integration (currently placeholder).
