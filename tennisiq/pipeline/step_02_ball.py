from __future__ import annotations

import torch

from tennisiq.common import get_device
from tennisiq.cv.ball.infer import infer_model, remove_outliers, split_track
from tennisiq.cv.ball.model import BallTrackerNet
from tennisiq.tracking.gap_fill import interpolate_track


def run_step_02_ball(frames, model_path: str, extrapolation: bool = True, device: str = "auto"):
    runtime_device = get_device(device)
    model = BallTrackerNet().to(runtime_device)
    model.load_state_dict(torch.load(model_path, map_location=runtime_device))
    model.eval()

    ball_track, dists = infer_model(frames, model, runtime_device)
    ball_track = remove_outliers(ball_track, dists)

    if extrapolation:
        for start, end in split_track(ball_track):
            ball_track[start:end] = interpolate_track(ball_track[start:end])
    return ball_track
