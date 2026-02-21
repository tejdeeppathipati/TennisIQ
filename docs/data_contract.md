# Data Contract

## `frames.jsonl`
One JSON object per frame.

Core fields:
- `frame_idx`, `ts_sec`
- `court_keypoints`
- `ball_xy`
- `playerA_bbox`, `playerB_bbox`
- `ball_court_xy`, `playerA_court_xy`, `playerB_court_xy`
- `homography_ok`, `homography_confidence`
- `ball_visible`, `ball_speed`, `ball_accel`
- `event_candidates` (per frame)
- `event_scores` (per frame)
- `ball_inout` in `{in, out, line, unknown}`

## `tracks.json`
Projected and stabilized tracks:
- `ball_projected`
- `playerA_projected`
- `playerB_projected`
- `ball_smoothed`
- `homography_series`
  - `series[]`: `frame_idx`, `confidence`, `carried`, `ok`
  - `mean_confidence`, `valid_ratio`

## `points.json`
Point-level output:
- `taxonomy_version`
- `points[]`
  - `point_id`, `start_frame`, `end_frame`, `start_sec`, `end_sec`, `end_reason`
  - `serve_frame`
  - `first_bounce_frame`, `first_bounce_court_xy`
  - `serve_zone`
  - `rally_hit_count`
  - `confidence`
  - `bounces[]`
- `clips[]`
  - `point_id`, `clip`, `start_frame`, `end_frame`

End reason taxonomy for this phase:
- `OUT`
- `BALL_LOST`
- `NET`
- `DOUBLE_BOUNCE`

## `insights.json`
- `stats`
  - includes serve in %, point count, bounce/hit counts, end-reason distribution
- `visuals`
  - `serve_placement_points`
  - `error_heatmap_points`
- `insights`
  - two short insight strings
- `point_cards`
  - per-point `why` and `try_instead`

## Media outputs
- `overlay.mp4`: frame overlay with court/players/ball and in/out coloring
- `clips/point_XXXX.mp4`: per-point replay clips
