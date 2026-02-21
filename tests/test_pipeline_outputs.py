from __future__ import annotations

import numpy as np

from tennisiq.io.schemas import FrameRecord
from tennisiq.pipeline.step_05_map_and_points import run_step_05_map_and_points
from tennisiq.pipeline.step_06_export_clips import run_step_06_export_clips


def test_step05_schema_and_homography_fallback(tmp_path):
    records = []
    for i in range(12):
        rec = FrameRecord(
            frame_idx=i,
            ts_sec=i / 30.0,
            court_keypoints=[(None, None)] * 14,
            ball_xy=(100.0 + i, 200.0 + i),
            playerA_bbox=None,
            playerB_bbox=None,
        )
        records.append(rec)

    out = run_step_05_map_and_points(records, fps=30)
    assert "frames" in out and "points" in out and "tracks" in out
    assert len(out["frames"]) == len(records)
    assert "homography_series" in out["tracks"]
    assert all("event_scores" in f for f in out["frames"])


def test_step06_clip_export(tmp_path):
    frames = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(20)]
    points = [{"point_id": 1, "start_frame": 5, "end_frame": 10}]

    clips_dir, meta = run_step_06_export_clips(str(tmp_path), frames, points, fps=30)
    assert len(meta) == 1
    assert meta[0]["point_id"] == 1
    assert (tmp_path / "clips" / "point_0001.mp4").exists()
    assert clips_dir.endswith("clips")
