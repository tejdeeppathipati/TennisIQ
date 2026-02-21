from __future__ import annotations

from typing import Optional, Tuple

from tennisiq.geometry.polygons import CourtGeometry, build_court_geometry, classify_point_in_out_line, is_inside_polygon


Point = Tuple[Optional[float], Optional[float]]


def classify_in_out(
    ball_point: Point,
    court_polygon=None,
    geometry: CourtGeometry | None = None,
    line_margin: float = 12.0,
) -> str:
    x, y = ball_point
    if x is None or y is None:
        return "unknown"

    # New calibrated path.
    if geometry is not None:
        label, _ = classify_point_in_out_line((float(x), float(y)), geometry, line_margin=line_margin)
        return label

    # Backward-compatible polygon path.
    if court_polygon is None:
        court_polygon = build_court_geometry().singles_polygon
    return "in" if is_inside_polygon((float(x), float(y)), court_polygon) else "out"
