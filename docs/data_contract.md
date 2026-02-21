# Data Contract

## `frames.jsonl`
One JSON row per frame with at least:
- `frame_idx`, `ts_sec`
- `court_keypoints` (14 points, image space)
- `ball_xy` (image space)
- `playerA_bbox`, `playerB_bbox` (image space)
- `ball_court_xy`, `playerA_court_xy`, `playerB_court_xy` (projected court space)
- `homography_ok`
- `ball_inout`

## `tracks.json`
Projected court-space tracks:
- `ball_projected`
- `playerA_projected`
- `playerB_projected`

## `points.json`
Point timeline and replay mapping:
- `points[]`: `point_id`, `start_frame`, `end_frame`, `start_sec`, `end_sec`, `end_reason`, `bounces`
- `clips[]`: clip path metadata per point

## `insights.json`
- `stats`: summary metrics (frames, points, bounces, serve in %)
- `insights`: 2 short text insights
- `visuals`: serve placement points + error heatmap points

## `overlay.mp4`
Debug overlay with:
- court keypoints
- tracked players
- tracked ball colored by in/out status

## `clips/`
Per-point replay clips (`point_XXXX.mp4`).
