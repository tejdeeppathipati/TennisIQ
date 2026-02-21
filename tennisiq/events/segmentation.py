from __future__ import annotations

from typing import List, Optional, Tuple


Point = Tuple[Optional[float], Optional[float]]


def segment_rallies(ball_track: List[Point], max_missing: int = 20) -> List[Tuple[int, int]]:
    segments = []
    start = None
    missing = 0
    for i, p in enumerate(ball_track):
        visible = p[0] is not None and p[1] is not None
        if visible and start is None:
            start = i
            missing = 0
        elif start is not None:
            if visible:
                missing = 0
            else:
                missing += 1
            if missing > max_missing:
                segments.append((start, i - missing))
                start = None
    if start is not None:
        segments.append((start, len(ball_track) - 1))
    return segments
