from __future__ import annotations

import cv2


def postprocess(feature_map, scale: int = 2):
    feature_map = (feature_map * 255).reshape((360, 640)).astype("uint8")
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
    x, y = None, None
    if circles is not None and len(circles) == 1:
        x = float(circles[0][0][0] * scale)
        y = float(circles[0][0][1] * scale)
    return x, y
