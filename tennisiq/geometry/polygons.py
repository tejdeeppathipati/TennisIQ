from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from tennisiq.geometry.court_reference import CourtReference


Point = Tuple[float, float]


def default_court_polygon() -> List[Point]:
    ref = CourtReference()
    return [
        ref.baseline_top[0],
        ref.baseline_top[1],
        ref.baseline_bottom[1],
        ref.baseline_bottom[0],
    ]


def is_inside_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    contour = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, point, False) >= 0


def mask_from_polygon(shape: Tuple[int, int], polygon: Iterable[Point]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    contour = np.array(list(polygon), dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [contour], 1)
    return mask
