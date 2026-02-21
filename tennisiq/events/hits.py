from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


Point = Tuple[Optional[float], Optional[float]]


def _vec(a: Point, b: Point):
    if a[0] is None or a[1] is None or b[0] is None or b[1] is None:
        return None
    return np.array([float(b[0]) - float(a[0]), float(b[1]) - float(a[1])], dtype=np.float32)


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 <= 1e-8 or n2 <= 1e-8:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def detect_hit_events(
    ball_track: Sequence[Point],
    min_turn_angle_deg: float = 50.0,
    min_speed: float = 6.0,
    nms_window: int = 3,
) -> List[Dict]:
    candidates: List[Dict] = []

    for i in range(2, len(ball_track) - 2):
        v_prev = _vec(ball_track[i - 1], ball_track[i])
        v_next = _vec(ball_track[i], ball_track[i + 1])
        if v_prev is None or v_next is None:
            continue

        s_prev = float(np.linalg.norm(v_prev))
        s_next = float(np.linalg.norm(v_next))
        if s_prev < min_speed and s_next < min_speed:
            continue

        turn = _angle_deg(v_prev, v_next)
        if turn < min_turn_angle_deg:
            continue

        score = min(turn / 180.0, 1.0) * 0.7 + min(max(s_prev, s_next) / 25.0, 1.0) * 0.3
        candidates.append({"frame_idx": i, "score": float(score), "turn_angle": float(turn), "speed_prev": s_prev, "speed_next": s_next})

    # NMS by score.
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    picked: List[Dict] = []
    blocked: set[int] = set()
    for c in candidates:
        i = c["frame_idx"]
        if i in blocked:
            continue
        picked.append(c)
        for k in range(i - nms_window, i + nms_window + 1):
            blocked.add(k)

    picked.sort(key=lambda x: x["frame_idx"])
    return picked


def detect_hits(
    ball_track: Sequence[Point],
    min_speed_change: float = 10.0,
) -> List[int]:
    # Backward-compatible wrapper.
    events = detect_hit_events(ball_track, min_turn_angle_deg=max(40.0, min_speed_change * 4.0), min_speed=6.0, nms_window=2)
    return [int(e["frame_idx"]) for e in events]
