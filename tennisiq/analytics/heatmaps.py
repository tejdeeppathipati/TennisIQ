from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


Point = Tuple[Optional[float], Optional[float]]


def ball_heatmap(points: List[Point], height: int, width: int) -> np.ndarray:
    hm = np.zeros((height, width), dtype=np.float32)
    for x, y in points:
        if x is None or y is None:
            continue
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            hm[yi, xi] += 1.0
    return hm
