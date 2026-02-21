from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def read_video(path_video: str) -> Tuple[List[np.ndarray], int]:
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def write_video(frames: List[np.ndarray], fps: int, path_output: str, codec: str = "DIVX") -> None:
    if not frames:
        return
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(path_output, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
