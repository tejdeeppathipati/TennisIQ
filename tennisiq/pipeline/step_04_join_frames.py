from __future__ import annotations

from typing import List

from tennisiq.io.schemas import FrameRecord


def run_step_04_join_frames(court_points: List[list], ball_track: List[tuple], players: List[dict], fps: int) -> List[FrameRecord]:
    joined = []
    n = min(len(court_points), len(ball_track), len(players))
    for idx in range(n):
        p = players[idx] if idx < len(players) else {}
        rec = FrameRecord(
            frame_idx=idx,
            ts_sec=float(idx / max(fps, 1)),
            court_keypoints=court_points[idx],
            ball_xy=ball_track[idx],
            playerA_bbox=p.get("playerA_bbox"),
            playerB_bbox=p.get("playerB_bbox"),
            playerA_id=p.get("playerA_id"),
            playerB_id=p.get("playerB_id"),
        )
        joined.append(rec)
    return joined
