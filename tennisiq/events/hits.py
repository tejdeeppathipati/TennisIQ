from __future__ import annotations

from typing import List, Optional, Tuple


Point = Tuple[Optional[float], Optional[float]]


def detect_hits(ball_track: List[Point], min_speed_change: float = 10.0) -> List[int]:
    hits = []
    for i in range(2, len(ball_track)):
        x0, y0 = ball_track[i - 2]
        x1, y1 = ball_track[i - 1]
        x2, y2 = ball_track[i]
        if None in (x0, y0, x1, y1, x2, y2):
            continue
        v1 = (x1 - x0, y1 - y0)
        v2 = (x2 - x1, y2 - y1)
        if (v1[0] * v2[0] + v1[1] * v2[1]) < -min_speed_change:
            hits.append(i - 1)
    return hits
