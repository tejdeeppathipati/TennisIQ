from __future__ import annotations

from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from tennisiq.common import get_device
from tennisiq.cv.court.infer import infer_points
from tennisiq.cv.court.model import CourtKeypointNet
from tennisiq.geometry.homography import get_trans_matrix, refer_kps


Point = tuple[float | None, float | None]


def _project_reference_points(h_ref_to_img: np.ndarray | None) -> List[Point]:
    if h_ref_to_img is None:
        return [(None, None)] * int(refer_kps.shape[0])
    projected = cv2.perspectiveTransform(refer_kps, h_ref_to_img)
    return [tuple(map(float, p[0])) for p in projected]


def _homography_confidence(
    pred_points: List[Point],
    h_ref_to_img: np.ndarray | None,
    prev_h_ref_to_img: np.ndarray | None,
) -> float:
    if h_ref_to_img is None:
        return 0.0

    projected = cv2.perspectiveTransform(refer_kps, h_ref_to_img)
    errors = []
    for i in range(min(len(pred_points), projected.shape[0])):
        px, py = pred_points[i]
        if px is None or py is None:
            continue
        rx, ry = projected[i][0]
        errors.append(float(np.hypot(float(px) - rx, float(py) - ry)))

    if errors:
        err_mean = float(np.mean(errors))
        conf_err = float(np.exp(-err_mean / 45.0))
    else:
        conf_err = 0.12

    if prev_h_ref_to_img is not None:
        denom = max(1e-6, float(np.linalg.norm(prev_h_ref_to_img)))
        diff = float(np.linalg.norm(h_ref_to_img - prev_h_ref_to_img) / denom)
        conf_temp = float(np.exp(-2.0 * diff))
    else:
        conf_temp = 1.0

    return float(max(0.0, min(1.0, 0.75 * conf_err + 0.25 * conf_temp)))


def run_step_01_court(
    frames,
    model_path: str,
    use_refine_kps: bool = False,
    use_homography: bool = False,
    stabilize_homography: bool = True,
    homography_min_confidence: float = 0.18,
    homography_carry_frames: int = 5,
    device: str = "auto",
) -> List[list]:
    runtime_device = get_device(device)
    model = CourtKeypointNet(out_channels=15).to(runtime_device)
    model.load_state_dict(torch.load(model_path, map_location=runtime_device))
    model.eval()

    points = []
    progress = tqdm(frames, desc="Step 1/6 Court keypoints", unit="frame")
    last_reliable_h_ref_to_img: np.ndarray | None = None
    last_reliable_conf = 0.0
    last_reliable_idx = -999999

    for frame_idx, frame in enumerate(progress):
        raw_points = infer_points(model, frame, runtime_device, use_refine_kps, use_homography=False)
        frame_points = raw_points

        if stabilize_homography:
            h_ref_to_img = get_trans_matrix(raw_points)
            conf = _homography_confidence(raw_points, h_ref_to_img, last_reliable_h_ref_to_img)
            carried = False

            if h_ref_to_img is not None and conf >= homography_min_confidence:
                frame_points = _project_reference_points(h_ref_to_img)
                last_reliable_h_ref_to_img = h_ref_to_img
                last_reliable_conf = conf
                last_reliable_idx = frame_idx
            elif last_reliable_h_ref_to_img is not None and (frame_idx - last_reliable_idx) <= homography_carry_frames:
                frame_points = _project_reference_points(last_reliable_h_ref_to_img)
                conf = max(0.0, last_reliable_conf * (0.85 ** (frame_idx - last_reliable_idx)))
                carried = True
            else:
                frame_points = [(None, None)] * len(raw_points)

            if frame_idx % 20 == 0:
                progress.set_postfix({"hom_conf": f"{conf:.2f}", "carried": int(carried)})
        elif use_homography:
            h_ref_to_img = get_trans_matrix(raw_points)
            frame_points = _project_reference_points(h_ref_to_img)

        points.append(frame_points)
    return points
