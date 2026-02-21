from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Point = Tuple[float | None, float | None]


FEATURE_NAMES = [
    "rule_score",
    "reversal",
    "speed_drop",
    "ball_visible",
    "speed",
    "accel",
    "vy",
    "vx",
    "speed_prev",
    "speed_next",
    "accel_prev",
    "accel_next",
    "missing_prev_5",
    "missing_next_5",
    "dist_to_net",
    "dist_to_playerA",
    "dist_to_playerB",
    "homography_confidence",
]


def _dist(a: Point, b: Point) -> float:
    if a[0] is None or a[1] is None or b[0] is None or b[1] is None:
        return float("nan")
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def compute_track_kinematics(ball_track: Sequence[Point], fps: int) -> List[Dict[str, float | bool]]:
    n = len(ball_track)
    dt = 1.0 / max(fps, 1)
    out: List[Dict[str, float | bool]] = []

    for i in range(n):
        x, y = ball_track[i]
        visible = x is not None and y is not None
        row: Dict[str, float | bool] = {
            "visible": bool(visible),
            "x": float(x) if visible else float("nan"),
            "y": float(y) if visible else float("nan"),
            "vx": float("nan"),
            "vy": float("nan"),
            "speed": float("nan"),
            "accel": float("nan"),
        }

        if i > 0:
            x0, y0 = ball_track[i - 1]
            if visible and x0 is not None and y0 is not None:
                vx = (float(x) - float(x0)) / dt
                vy = (float(y) - float(y0)) / dt
                row["vx"] = vx
                row["vy"] = vy
                row["speed"] = float(np.hypot(vx, vy))

        out.append(row)

    for i in range(1, n):
        s0 = out[i - 1]["speed"]
        s1 = out[i]["speed"]
        if np.isfinite(s0) and np.isfinite(s1):
            out[i]["accel"] = float((float(s1) - float(s0)) / dt)

    return out


def missing_ratio(ball_track: Sequence[Point], start: int, end: int) -> float:
    start = max(0, start)
    end = min(len(ball_track), end)
    if end <= start:
        return 1.0
    total = end - start
    miss = 0
    for i in range(start, end):
        x, y = ball_track[i]
        if x is None or y is None:
            miss += 1
    return miss / total


def _safe_float(v, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (float, int)) and np.isfinite(float(v)):
        return float(v)
    return default


def build_candidate_feature_row(
    frame_idx: int,
    candidate: Dict,
    kinematics: Sequence[Dict[str, float | bool]],
    ball_track: Sequence[Point],
    mapped_frames: Sequence[Dict],
    net_y: float,
) -> Dict[str, float]:
    i = frame_idx
    curr = kinematics[i] if 0 <= i < len(kinematics) else {}
    prev = kinematics[i - 1] if i - 1 >= 0 else {}
    nxt = kinematics[i + 1] if i + 1 < len(kinematics) else {}

    row = {
        "rule_score": _safe_float(candidate.get("rule_score"), 0.0),
        "reversal": float(candidate.get("reversal", False)),
        "speed_drop": float(candidate.get("speed_drop", False)),
        "ball_visible": float(curr.get("visible", False)),
        "speed": _safe_float(curr.get("speed"), 0.0),
        "accel": _safe_float(curr.get("accel"), 0.0),
        "vy": _safe_float(curr.get("vy"), 0.0),
        "vx": _safe_float(curr.get("vx"), 0.0),
        "speed_prev": _safe_float(prev.get("speed"), 0.0),
        "speed_next": _safe_float(nxt.get("speed"), 0.0),
        "accel_prev": _safe_float(prev.get("accel"), 0.0),
        "accel_next": _safe_float(nxt.get("accel"), 0.0),
        "missing_prev_5": missing_ratio(ball_track, i - 5, i),
        "missing_next_5": missing_ratio(ball_track, i + 1, i + 6),
        "dist_to_net": 9999.0,
        "dist_to_playerA": 9999.0,
        "dist_to_playerB": 9999.0,
        "homography_confidence": 0.0,
    }

    if 0 <= i < len(mapped_frames):
        fr = mapped_frames[i]
        ball_court = tuple(fr.get("ball_court_xy", (None, None)))
        if ball_court[1] is not None:
            row["dist_to_net"] = abs(float(ball_court[1]) - float(net_y))

        p_a = tuple(fr.get("playerA_court_xy", (None, None)))
        p_b = tuple(fr.get("playerB_court_xy", (None, None)))
        row["dist_to_playerA"] = _safe_float(_dist(ball_court, p_a), 9999.0)
        row["dist_to_playerB"] = _safe_float(_dist(ball_court, p_b), 9999.0)
        row["homography_confidence"] = _safe_float(fr.get("homography_confidence", 0.0), 0.0)

    return row


def feature_matrix(feature_rows: Iterable[Dict[str, float]]) -> np.ndarray:
    rows = []
    for row in feature_rows:
        rows.append([float(row.get(name, 0.0)) for name in FEATURE_NAMES])
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)
