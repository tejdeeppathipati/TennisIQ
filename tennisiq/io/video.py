from __future__ import annotations

from pathlib import Path
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


def normalize_video_fps_moviepy(path_input: str, path_output: str, target_fps: int = 30) -> str:
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")

    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
    except Exception:
        from moviepy.editor import VideoFileClip

    path_out = Path(path_output)
    path_out.parent.mkdir(parents=True, exist_ok=True)

    clip = VideoFileClip(path_input)
    try:
        if hasattr(clip, "with_fps"):
            clip_out = clip.with_fps(target_fps)
        else:
            clip_out = clip.set_fps(target_fps)

        try:
            clip_out.write_videofile(str(path_out), fps=target_fps, audio=False, logger=None)
        finally:
            clip_out.close()
    finally:
        clip.close()

    return str(path_out)
