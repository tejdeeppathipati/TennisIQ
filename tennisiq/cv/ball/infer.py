from __future__ import annotations

import argparse
from itertools import groupby
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm

from tennisiq.common import get_device
from tennisiq.cv.ball.model import BallTrackerNet
from tennisiq.cv.ball.postprocess import postprocess
from tennisiq.io.video import read_video
from tennisiq.tracking.gap_fill import interpolate_track


Point = Tuple[Optional[float], Optional[float]]


def infer_model(frames: List[np.ndarray], model, device: str):
    height, width = 360, 640
    dists = [-1.0, -1.0]
    ball_track: List[Point] = [(None, None), (None, None)]

    for num in tqdm(range(2, len(frames))):
        img = cv2.resize(frames[num], (width, height))
        img_prev = cv2.resize(frames[num - 1], (width, height))
        img_preprev = cv2.resize(frames[num - 2], (width, height))

        imgs = np.concatenate((img, img_prev, img_preprev), axis=2).astype(np.float32) / 255.0
        inp = np.expand_dims(np.rollaxis(imgs, 2, 0), axis=0)

        out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess(output)
        ball_track.append((x_pred, y_pred))

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
    parser.add_argument("--extrapolation", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    model = BallTrackerNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    frames, fps = read_video(args.video_path)
    ball_track, dists = infer_model(frames, model, device)
    ball_track = remove_outliers(ball_track, dists)

    if args.extrapolation:
        subtracks = split_track(ball_track)
        for start, end in subtracks:
            ball_track[start:end] = interpolate_track(ball_track[start:end])

    write_track(frames, ball_track, args.video_out_path, fps)


if __name__ == "__main__":
    main()
