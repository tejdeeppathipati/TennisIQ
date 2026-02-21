from __future__ import annotations

from typing import List

import torch

from tennisiq.common import get_device
from tennisiq.cv.court.infer import infer_points
from tennisiq.cv.court.model import CourtKeypointNet


def run_step_01_court(frames, model_path: str, use_refine_kps: bool = False, use_homography: bool = False, device: str = "auto") -> List[list]:
    runtime_device = get_device(device)
    model = CourtKeypointNet(out_channels=15).to(runtime_device)
    model.load_state_dict(torch.load(model_path, map_location=runtime_device))
    model.eval()

    points = []
    for frame in frames:
        points.append(infer_points(model, frame, runtime_device, use_refine_kps, use_homography))
    return points
