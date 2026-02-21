from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


Point = Tuple[Optional[float], Optional[float]]


def interpolate_track(points: List[Point]) -> List[Point]:
    x = np.array([p[0] if p[0] is not None else np.nan for p in points], dtype=float)
    y = np.array([p[1] if p[1] is not None else np.nan for p in points], dtype=float)

    def _interp(arr: np.ndarray) -> np.ndarray:
        idx = np.arange(arr.size)
        nans = np.isnan(arr)
        if nans.all() or (~nans).sum() < 2:
            return arr
        arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
        return arr

    x = _interp(x)
    y = _interp(y)
    return [(float(px), float(py)) if not np.isnan(px) and not np.isnan(py) else (None, None) for px, py in zip(x, y)]
