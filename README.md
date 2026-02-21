# TennisIQ

TennisIQ is a modular tennis video intelligence project for:
- court keypoint detection,
- ball tracking,
- event extraction (bounce/hit/in-out),
- match analytics export.

## What This Repo Contains

- A refactored Python package in `tennisiq/` (single canonical codebase).
- Training and inference entry points for court and ball models.
- A step-based end-to-end pipeline.
- Scripts for setup/train/demo run.
- Lightweight UI viewer via Streamlit.

## Project Structure

```text
TennisIQ/
  README.md
  PROJECT.md
  requirements.txt
  .env.example
  LICENSE

  data/
    raw/
      input.mp4
    datasets/
      balltracking/
        images/
        splits/
          train.txt
          val.txt
          test.txt
        labels_train.csv
        labels_val.csv
      court_identification/
        images/
        keypoints/
        splits/
    annotations/

  checkpoints/
    ball/
      best.pt
      last.pt
    court/
      best.pt
      last.pt

  outputs/
    runs/

  tennisiq/
    config/
    cv/
    geometry/
    tracking/
    events/
    analytics/
    io/
    pipeline/

  scripts/
    setup_env.sh
    train_ball.sh
    train_court.sh
    run_demo.sh

  ui/
    streamlit_app.py

  docs/
    workflow.md
    data_contract.md
    refactor.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Checkpoints

Expected checkpoint paths:
- `checkpoints/ball/best.pt`
- `checkpoints/court/best.pt`

Current training scripts also write `last.pt` under the same folders.

## Run End-to-End Pipeline

```bash
python3 -m tennisiq.pipeline.run_all \
  --video data/raw/input.mp4 \
  --court-model checkpoints/court/best.pt \
  --ball-model checkpoints/ball/best.pt \
  --player-model yolov8n.pt \
  --event-model-path /path/to/event_model.cbm \
  --event-threshold 0.5 \
  --line-margin-px 12 \
  --serve-speed-thresh 600 \
  --inactivity-frames 24 \
  --ball-lost-frames 12 \
  --output outputs/runs/$(date +%Y-%m-%d_%H%M)
```

Player model notes:
- If `yolov8n.pt` exists locally, it will be used.
- To allow auto-download via Ultralytics, add `--allow-player-model-download`.

Outputs include:
- `frames.jsonl`, `tracks.json`, `points.json`, `insights.json`
- `overlay.mp4`
- `clips/point_XXXX.mp4`

## Train Models

Ball:
```bash
python3 -m tennisiq.cv.ball.train \
  --dataset-root data/datasets/balltracking
```

Court:
```bash
python3 -m tennisiq.cv.court.train \
  --data-root data/datasets/court_identification
```

## Dataset Splits

For showcase/testing, clip-level split files are included at:
- `data/datasets/balltracking/splits/train.txt`
- `data/datasets/balltracking/splits/val.txt`
- `data/datasets/balltracking/splits/test.txt`

## UI

```bash
streamlit run ui/streamlit_app.py
```

## Evaluation Harness

Create annotation template from a run:
```bash
python3 scripts/make_annotation_template.py \
  --run-dir outputs/runs/<run_id> \
  --output eval/annotations/match_01.jsonl
```

Validate annotations:
```bash
python3 scripts/review_annotations.py \
  --annotations eval/annotations/match_01.jsonl
```

Evaluate and apply regression gates:
```bash
bash scripts/eval_run.sh \
  outputs/runs/<run_id> \
  eval/annotations/match_01.jsonl \
  eval/baseline_metrics.json
```

## Notes

- The repository is organized for GitHub upload with one primary readme (`README.md`) and one technical project companion (`PROJECT.md`).
- Large artifacts (datasets, checkpoints, outputs) are ignored by `.gitignore` by default.
