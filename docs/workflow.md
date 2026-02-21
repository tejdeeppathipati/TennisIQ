# Workflow

## 1) Prepare inputs
- Put match video in `data/raw/`
- Ensure checkpoints:
  - `checkpoints/court/best.pt`
  - `checkpoints/ball/best.pt`
- Optional: provide CatBoost event model (`.cbm`) path

## 2) Run pipeline
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

Notes:
- Player model auto-download is OFF by default.
- To allow Ultralytics download if model is missing, add:
  - `--allow-player-model-download`

## 3) Inspect outputs
Open `outputs/runs/<run_id>/`:
- `frames.jsonl`
- `tracks.json`
- `points.json`
- `insights.json`
- `overlay.mp4`
- `clips/`

## 4) Dashboard
```bash
streamlit run ui/streamlit_app.py
```
Use the run directory selector to load timeline, clips, charts, and point cards.

## 5) Manual benchmark + regression
Create template:
```bash
python3 scripts/make_annotation_template.py \
  --run-dir outputs/runs/<run_id> \
  --output eval/annotations/match_01.jsonl
```

Review annotations:
```bash
python3 scripts/review_annotations.py \
  --annotations eval/annotations/match_01.jsonl
```

Evaluate:
```bash
bash scripts/eval_run.sh \
  outputs/runs/<run_id> \
  eval/annotations/match_01.jsonl \
  eval/baseline_metrics.json
```
