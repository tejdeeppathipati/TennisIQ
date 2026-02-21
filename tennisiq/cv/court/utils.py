from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import sympy
from sympy import Line


Point = Tuple[Optional[float], Optional[float]]


def gaussian2d(shape, sigma=1.0):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap: np.ndarray, center, radius: int, k: float = 1.0) -> np.ndarray:
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def line_intersection(line1, line2):
    l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
    l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))
    intersection = l1.intersection(l2)
    if intersection and isinstance(intersection[0], sympy.geometry.point.Point2D):
        return intersection[0].coordinates
    return None


def is_point_in_image(x: Optional[float], y: Optional[float], input_width: int = 1280, input_height: int = 720) -> bool:
    if x is None or y is None:
        return False
    return 0 <= x <= input_width and 0 <= y <= input_height
