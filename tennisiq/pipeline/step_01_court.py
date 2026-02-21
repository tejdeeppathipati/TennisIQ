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


def _is_valid_point(point: Point) -> bool:
    return point[0] is not None and point[1] is not None


def _project_reference_points(h_ref_to_img: np.ndarray | None) -> List[Point]:
    if h_ref_to_img is None:
        return [(None, None)] * int(refer_kps.shape[0])
    projected = cv2.perspectiveTransform(refer_kps, h_ref_to_img)
    return [tuple(map(float, p[0])) for p in projected]


def _homography_confidence(
    pred_points: List[Point],
    h_ref_to_img: np.ndarray | None,
    prev_h_ref_to_img: np.ndarray | None,
    inlier_ratio: float = 0.0,
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

    inlier_term = float(max(0.0, min(1.0, inlier_ratio)))
    return float(max(0.0, min(1.0, 0.68 * conf_err + 0.22 * conf_temp + 0.10 * inlier_term)))


def _estimate_homography(raw_points: List[Point]) -> tuple[np.ndarray | None, float]:
    src = []
    dst = []
    for i, point in enumerate(raw_points):
        if not _is_valid_point(point):
            continue
        src.append(refer_kps[i][0].tolist())
        dst.append([float(point[0]), float(point[1])])

    if len(src) >= 6:
        matrix, inliers = cv2.findHomography(
            np.float32(src),
            np.float32(dst),
            method=cv2.RANSAC,
            ransacReprojThreshold=10.0,
            maxIters=2000,
            confidence=0.995,
        )
        if matrix is not None:
            if inliers is None:
                return matrix, 0.0
            inlier_ratio = float(np.mean(inliers.astype(np.float32)))
            if inlier_ratio >= 0.40:
                return matrix, inlier_ratio

    # Fallback to legacy estimator if RANSAC cannot find a stable mapping.
    return get_trans_matrix(raw_points), 0.0


def _is_plausible_projection(points: List[Point], frame_shape: tuple[int, ...]) -> bool:
    if len(points) < 14:
        return False

    valid_indices = [i for i, p in enumerate(points) if _is_valid_point(p)]
    if len(valid_indices) < 12:
        return False

    h, w = frame_shape[:2]
    margin = float(max(80.0, 0.06 * min(w, h)))
    in_bounds = 0
    for i in valid_indices:
        x, y = points[i]
        if (-margin <= float(x) <= (w + margin)) and (-margin <= float(y) <= (h + margin)):
            in_bounds += 1
    if in_bounds < 12:
        return False

    # Outer-court geometry checks.
    for i in (0, 1, 2, 3):
        if not _is_valid_point(points[i]):
            return False
    p0, p1, p2, p3 = points[0], points[1], points[2], points[3]

    if not (float(p0[0]) < float(p1[0]) and float(p2[0]) < float(p3[0])):
        return False
    if not (float(p0[1]) < float(p2[1]) and float(p1[1]) < float(p3[1])):
        return False

    quad = np.array([[p0[0], p0[1]], [p1[0], p1[1]], [p3[0], p3[1]], [p2[0], p2[1]]], dtype=np.float32)
    area = float(cv2.contourArea(quad))
    if area < 0.03 * float(w * h) or area > 0.95 * float(w * h):
        return False

    top_w = float(np.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])))
    bot_w = float(np.hypot(float(p3[0]) - float(p2[0]), float(p3[1]) - float(p2[1])))
    if min(top_w, bot_w) < 0.10 * float(w):
        return False
    width_ratio = max(top_w, bot_w) / max(1.0, min(top_w, bot_w))
    if width_ratio > 4.5:
        return False

    if all(_is_valid_point(points[i]) for i in (8, 9, 10, 11)):
        y_top = 0.5 * (float(p0[1]) + float(p1[1]))
        y_bottom = 0.5 * (float(p2[1]) + float(p3[1]))
        y_top_service = 0.5 * (float(points[8][1]) + float(points[9][1]))
        y_bottom_service = 0.5 * (float(points[10][1]) + float(points[11][1]))
        if not (y_top < y_top_service < y_bottom_service < y_bottom):
            return False

    return True


def run_step_01_court(
    frames,
    model_path: str,
    use_refine_kps: bool = False,
    use_homography: bool = False,
    stabilize_homography: bool = True,
    homography_min_confidence: float = 0.24,
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
            h_ref_to_img, inlier_ratio = _estimate_homography(raw_points)
            if h_ref_to_img is not None:
                projected = _project_reference_points(h_ref_to_img)
                if not _is_plausible_projection(projected, frame.shape):
                    h_ref_to_img = None
                    inlier_ratio = 0.0
            conf = _homography_confidence(raw_points, h_ref_to_img, last_reliable_h_ref_to_img, inlier_ratio=inlier_ratio)
            carried = False

            if h_ref_to_img is not None and conf >= homography_min_confidence:
                frame_points = projected
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
