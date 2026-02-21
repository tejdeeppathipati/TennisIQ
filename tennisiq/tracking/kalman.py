from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[Optional[float], Optional[float]]


def kalman_smooth(points: List[Point]) -> List[Point]:
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

    out: List[Point] = []
    for x, y in points:
        pred = kf.predict()
        if x is not None and y is not None:
            meas = np.array([[np.float32(x)], [np.float32(y)]])
            est = kf.correct(meas)
            out.append((float(est[0]), float(est[1])))
        else:
            out.append((float(pred[0]), float(pred[1])))
    return out
