from __future__ import annotations

import cv2
import numpy as np


def postprocess(feature_map, min_peak: int = 140):
    feature_map = (feature_map * 255).reshape((360, 640)).astype("uint8")
    peak_y, peak_x = np.unravel_index(int(np.argmax(feature_map)), feature_map.shape)
    peak_value = int(feature_map[peak_y, peak_x])
    if peak_value < min_peak:
        return None, None

    _, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=50,
        param2=2,
        minRadius=2,
        maxRadius=7,
    )

    if circles is None:
        return float(peak_x), float(peak_y)

    # HoughCircles returns shape (1, N, 3). Pick the circle nearest to the heatmap peak.
    circles = circles[0]
    if len(circles) == 0:
        return float(peak_x), float(peak_y)

    best = min(circles, key=lambda c: (c[0] - peak_x) ** 2 + (c[1] - peak_y) ** 2)
    return float(best[0]), float(best[1])
