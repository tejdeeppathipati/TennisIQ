from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from tennisiq.geometry.court_reference import CourtReference


Point = Tuple[float, float]
Line = Tuple[Point, Point]


@dataclass
class CourtGeometry:
    singles_polygon: List[Point]
    doubles_polygon: List[Point]
    service_boxes: Dict[str, List[Point]]
    line_segments: Dict[str, Line]
    net_line: Line
    center_x: float



def _poly(*pts: Point) -> List[Point]:
    return [tuple(map(float, p)) for p in pts]


def build_court_geometry(ref: CourtReference | None = None) -> CourtGeometry:
    ref = ref or CourtReference()

    singles_polygon = _poly(
        ref.left_inner_line[0],
        ref.right_inner_line[0],
        ref.right_inner_line[1],
        ref.left_inner_line[1],
    )
    doubles_polygon = _poly(
        ref.left_court_line[0],
        ref.right_court_line[0],
        ref.right_court_line[1],
        ref.left_court_line[1],
    )

    service_boxes = {
        "top_left": _poly(ref.left_inner_line[0], ref.middle_line[0], (ref.middle_line[1][0], ref.net[0][1]), (ref.left_inner_line[1][0], ref.net[0][1])),
        "top_right": _poly(ref.middle_line[0], ref.right_inner_line[0], (ref.right_inner_line[1][0], ref.net[0][1]), (ref.middle_line[1][0], ref.net[0][1])),
        "bottom_left": _poly((ref.left_inner_line[0][0], ref.net[0][1]), (ref.middle_line[0][0], ref.net[0][1]), ref.middle_line[1], ref.left_inner_line[1]),
        "bottom_right": _poly((ref.middle_line[0][0], ref.net[0][1]), (ref.right_inner_line[0][0], ref.net[0][1]), ref.right_inner_line[1], ref.middle_line[1]),
    }

    line_segments = {
        "singles_left": (tuple(map(float, ref.left_inner_line[0])), tuple(map(float, ref.left_inner_line[1]))),
        "singles_right": (tuple(map(float, ref.right_inner_line[0])), tuple(map(float, ref.right_inner_line[1]))),
        "baseline_top": (tuple(map(float, ref.baseline_top[0])), tuple(map(float, ref.baseline_top[1]))),
        "baseline_bottom": (tuple(map(float, ref.baseline_bottom[0])), tuple(map(float, ref.baseline_bottom[1]))),
        "service_top": (tuple(map(float, ref.top_inner_line[0])), tuple(map(float, ref.top_inner_line[1]))),
        "service_bottom": (tuple(map(float, ref.bottom_inner_line[0])), tuple(map(float, ref.bottom_inner_line[1]))),
        "service_center": (tuple(map(float, ref.middle_line[0])), tuple(map(float, ref.middle_line[1]))),
    }

    net_line = (tuple(map(float, ref.net[0])), tuple(map(float, ref.net[1])))
    center_x = float(ref.middle_line[0][0])

    return CourtGeometry(
        singles_polygon=singles_polygon,
        doubles_polygon=doubles_polygon,
        service_boxes=service_boxes,
        line_segments=line_segments,
        net_line=net_line,
        center_x=center_x,
    )


def default_court_polygon() -> List[Point]:
    return build_court_geometry().singles_polygon


def is_inside_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    contour = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, point, False) >= 0


def mask_from_polygon(shape: Tuple[int, int], polygon: Iterable[Point]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    contour = np.array(list(polygon), dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [contour], 1)
    return mask


def point_to_segment_distance(point: Point, segment: Line) -> float:
    px, py = point
    (x1, y1), (x2, y2) = segment
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1

    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return float(np.hypot(px - x1, py - y1))

    c2 = vx * vx + vy * vy
    if c2 <= 1e-12:
        return float(np.hypot(px - x1, py - y1))

    b = c1 / c2
    bx, by = x1 + b * vx, y1 + b * vy
    return float(np.hypot(px - bx, py - by))


def closest_line(point: Point, line_segments: Dict[str, Line]) -> Tuple[str, float]:
    best_name = ""
    best_dist = float("inf")
    for name, seg in line_segments.items():
        d = point_to_segment_distance(point, seg)
        if d < best_dist:
            best_name = name
            best_dist = d
    return best_name, best_dist


def service_box_label(point: Point, geometry: CourtGeometry) -> str | None:
    for name, poly in geometry.service_boxes.items():
        if is_inside_polygon(point, poly):
            return name
    return None


def classify_point_in_out_line(
    point: Point | None,
    geometry: CourtGeometry,
    line_margin: float = 12.0,
) -> Tuple[str, Dict[str, float | str]]:
    if point is None:
        return "unknown", {"closest_line": "", "line_distance": float("inf")}

    x, y = point
    if x is None or y is None:
        return "unknown", {"closest_line": "", "line_distance": float("inf")}

    p = (float(x), float(y))
    in_singles = is_inside_polygon(p, geometry.singles_polygon)
    line_name, line_dist = closest_line(p, geometry.line_segments)

    if line_dist <= line_margin:
        return "line", {"closest_line": line_name, "line_distance": line_dist}
    if in_singles:
        return "in", {"closest_line": line_name, "line_distance": line_dist}
    return "out", {"closest_line": line_name, "line_distance": line_dist}


def side_label(point: Point | None, center_x: float) -> str:
    if point is None or point[0] is None:
        return "unknown"
    return "deuce" if point[0] >= center_x else "ad"
