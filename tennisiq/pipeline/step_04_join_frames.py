from __future__ import annotations

from typing import List

from tennisiq.io.schemas import FrameRecord


def run_step_04_join_frames(court_points: List[list], ball_track: List[tuple]):
    joined = []
    n = min(len(court_points), len(ball_track))
    for idx in range(n):
        rec = FrameRecord(frame_idx=idx, court_points=court_points[idx], ball_point=ball_track[idx])
        joined.append(rec)
    return joined
