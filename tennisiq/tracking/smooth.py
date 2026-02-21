from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


Point = Tuple[Optional[float], Optional[float]]


def moving_average(points: List[Point], window: int = 3) -> List[Point]:
    if window <= 1:
        return points
    half = window // 2
    out: List[Point] = []
    for i in range(len(points)):
        xs, ys = [], []
        for j in range(max(0, i - half), min(len(points), i + half + 1)):
            x, y = points[j]
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        if not xs:
            out.append((None, None))
        else:
            out.append((float(np.mean(xs)), float(np.mean(ys))))
    return out
