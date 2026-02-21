# Workflow

1. Put video in `data/raw/`.
2. Ensure checkpoints exist:
   - `checkpoints/court/best.pt`
   - `checkpoints/ball/best.pt`
3. Run pipeline:
```bash
python3 -m tennisiq.pipeline.run_all \
  --video data/raw/input.mp4 \
  --court-model checkpoints/court/best.pt \
  --ball-model checkpoints/ball/best.pt \
  --player-model yolov8n.pt \
  --output outputs/runs/$(date +%Y-%m-%d_%H%M)
```
4. Open outputs in `outputs/runs/<run_id>/`:
   - `overlay.mp4`
   - `frames.jsonl`
   - `tracks.json`
   - `points.json`
   - `insights.json`
   - `clips/`

Notes:
- By default, player model auto-download is OFF.
- To allow Ultralytics model download (if not present locally), add:
  - `--allow-player-model-download`
