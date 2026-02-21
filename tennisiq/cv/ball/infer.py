from __future__ import annotations

import argparse
from itertools import groupby
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm

from tennisiq.common import get_device
from tennisiq.cv.ball.model import BallTrackerNet
from tennisiq.cv.ball.postprocess import postprocess
from tennisiq.io.video import normalize_video_fps_moviepy, read_video
from tennisiq.tracking.gap_fill import interpolate_track


Point = Tuple[Optional[float], Optional[float]]


def _letterbox(frame: np.ndarray, dst_w: int, dst_h: int):
    src_h, src_w = frame.shape[:2]
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((dst_h, dst_w, 3), dtype=frame.dtype)
    pad_x = (dst_w - new_w) // 2
    pad_y = (dst_h - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def infer_model(frames: List[np.ndarray], model, device: str):
    height, width = 360, 640
    if not frames:
        return [], []

    src_h, src_w = frames[0].shape[:2]
    _, scale, pad_x, pad_y = _letterbox(frames[0], width, height)

    def preprocess(frame: np.ndarray) -> np.ndarray:
        # Most videos keep constant size. If not, recompute letterbox for the frame.
        if frame.shape[0] == src_h and frame.shape[1] == src_w:
            framed = cv2.resize(frame, (int(round(src_w * scale)), int(round(src_h * scale))))
            canvas = np.zeros((height, width, 3), dtype=frame.dtype)
            canvas[pad_y : pad_y + framed.shape[0], pad_x : pad_x + framed.shape[1]] = framed
            return canvas
        return _letterbox(frame, width, height)[0]

    dists = [-1.0, -1.0]
    ball_track: List[Point] = [(None, None), (None, None)]

    for num in tqdm(range(2, len(frames))):
        img = preprocess(frames[num])
        img_prev = preprocess(frames[num - 1])
        img_preprev = preprocess(frames[num - 2])

        imgs = np.concatenate((img, img_prev, img_preprev), axis=2).astype(np.float32) / 255.0
        inp = np.expand_dims(np.rollaxis(imgs, 2, 0), axis=0)

        out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess(output)

        if x_pred is not None and y_pred is not None:
            x_src = (x_pred - pad_x) / scale
            y_src = (y_pred - pad_y) / scale
            if 0.0 <= x_src < src_w and 0.0 <= y_src < src_h:
                ball_track.append((float(x_src), float(y_src)))
            else:
                ball_track.append((None, None))
        else:
            ball_track.append((None, None))

        p1, p0 = ball_track[-1], ball_track[-2]
        if p1[0] is not None and p1[1] is not None and p0[0] is not None and p0[1] is not None:
            dists.append(float(distance.euclidean(p1, p0)))
        else:
            dists.append(-1.0)
    return ball_track, dists


def remove_outliers(ball_track: List[Point], dists: List[float], max_dist: float = 100.0) -> List[Point]:
    track = list(ball_track)
    for i in range(1, len(track) - 1):
        if dists[i] <= max_dist:
            continue
        next_dist = dists[i + 1] if i + 1 < len(dists) else -1
        prev_dist = dists[i - 1] if i - 1 >= 0 else -1
        if next_dist > max_dist or next_dist == -1:
            track[i] = (None, None)
        elif prev_dist == -1:
            track[i - 1] = (None, None)
    return track


def split_track(ball_track: List[Point], max_gap: int = 4, max_dist_gap: float = 80.0, min_track: int = 5):
    list_det = [0 if (p[0] is not None and p[1] is not None) else 1 for p in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor, min_value = 0, 0
    result = []
    for i, (k, length) in enumerate(groups):
        if k == 1 and 0 < i < len(groups) - 1:
            if ball_track[cursor - 1][0] is not None and ball_track[cursor + length][0] is not None:
                dist = distance.euclidean(ball_track[cursor - 1], ball_track[cursor + length])
            else:
                dist = np.inf
            if length >= max_gap or dist / max(length, 1) > max_dist_gap:
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + length - 1
        cursor += length
    if len(list_det) - min_value > min_track:
        result.append([min_value, len(list_det)])
    return result


def write_track(frames: List[np.ndarray], ball_track: List[Point], path_output_video: str, fps: int, trace: int = 7):
    if not frames:
        return
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height))

    for num in range(len(frames)):
        frame = frames[num].copy()
        for i in range(trace):
            idx = num - i
            if idx <= 0:
                continue
            x, y = ball_track[idx]
            if x is None or y is None:
                break
            frame = cv2.circle(frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=max(1, 10 - i))
        out.write(frame)
    out.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--video-out-path", type=str, required=True)
    parser.add_argument("--normalize-fps", type=int, default=0, help="If > 0, preprocess input video with MoviePy to this FPS.")
    parser.add_argument("--extrapolation", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    video_path = args.video_path

    if args.normalize_fps > 0:
        out_path = Path(args.video_out_path)
        normalized_input = out_path.parent / f"{out_path.stem}_input_{args.normalize_fps}fps.mp4"
        video_path = normalize_video_fps_moviepy(args.video_path, str(normalized_input), target_fps=args.normalize_fps)

    model = BallTrackerNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    frames, fps = read_video(video_path)
    ball_track, dists = infer_model(frames, model, device)
    ball_track = remove_outliers(ball_track, dists)

    if args.extrapolation:
        subtracks = split_track(ball_track)
        for start, end in subtracks:
            ball_track[start:end] = interpolate_track(ball_track[start:end])

    write_track(frames, ball_track, args.video_out_path, fps)


if __name__ == "__main__":
    main()
