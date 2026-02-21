from __future__ import annotations

from typing import Optional, Tuple

from tennisiq.geometry.polygons import is_inside_polygon


Point = Tuple[Optional[float], Optional[float]]


def classify_in_out(ball_point: Point, court_polygon) -> str:
    x, y = ball_point
    if x is None or y is None:
        return "unknown"
    return "in" if is_inside_polygon((x, y), court_polygon) else "out"
