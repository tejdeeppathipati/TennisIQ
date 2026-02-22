"""
Homography computation, temporal stabilization, and confidence scoring.

Implements FR-07, FR-08, and FR-09:
  FR-07: Compute homography from detected keypoints using court reference.
  FR-08: Carry last reliable homography forward up to 5 frames during occlusion.
  FR-09: Confidence score per frame via reprojection error; flag unreliable frames.
"""
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import distance

from .court_reference import CourtReference

logger = logging.getLogger(__name__)

MIN_KEYPOINTS_FOR_HOMOGRAPHY = 4
MAX_CARRY_FORWARD_FRAMES = 5
CONFIDENCE_THRESHOLD = 0.7


@dataclass
class FrameHomography:
    """Per-frame homography result."""
    frame_idx: int
    matrix: np.ndarray | None          # 3x3 homography or None
    inverse_matrix: np.ndarray | None  # 3x3 inverse (court→pixel) or None
    reprojection_error: float          # mean pixel error across keypoints
    confidence: float                  # 0.0–1.0 score
    reliable: bool                     # whether this frame is usable for in/out
    detected_count: int                # how many keypoints were detected
    carried_forward: bool              # True if matrix was inherited from prior frame


def _build_homography(
    detected_kps: list[tuple[float | None, float | None]],
    court_ref: CourtReference,
    conf_indices: dict[int, list[int]],
    refer_kps: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """
    Find the best homography by trying all 12 four-point court configurations
    and selecting the one with lowest median reprojection error.

    Returns (matrix, reprojection_error).  matrix is None if fewer than 4
    keypoints are detected.
    """
    points = [(x, y) if x is not None else (None, None) for x, y in detected_kps]

    valid_count = sum(1 for x, y in points if x is not None)
    if valid_count < MIN_KEYPOINTS_FOR_HOMOGRAPHY:
        return None, float("inf")

    best_matrix = None
    best_error = float("inf")

    for conf_id in range(1, 13):
        inds = conf_indices[conf_id]
        conf_pts = court_ref.court_conf[conf_id]

        src_pts = [conf_pts[j] for j in range(4)]
        dst_pts = [points[inds[j]] for j in range(4)]

        if any(p[0] is None for p in dst_pts):
            continue

        matrix, _ = cv2.findHomography(
            np.float32(src_pts), np.float32(dst_pts), method=0,
        )
        if matrix is None:
            continue

        refer_shaped = refer_kps.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(refer_shaped, matrix)

        dists = []
        for i in range(len(detected_kps)):
            if i not in inds and points[i][0] is not None:
                d = distance.euclidean(points[i], transformed[i].flatten())
                dists.append(d)

        if not dists:
            continue

        median_err = float(np.median(dists))
        if median_err < best_error:
            best_error = median_err
            best_matrix = matrix

    return best_matrix, best_error


def _error_to_confidence(reprojection_error: float) -> float:
    """
    Map reprojection error (pixels) to a 0–1 confidence score.
    < 5 px  → 1.0  (excellent)
    5–20 px → linear decay 1.0 → 0.5
    20–50   → linear decay 0.5 → 0.0
    > 50    → 0.0
    """
    if reprojection_error <= 5.0:
        return 1.0
    if reprojection_error <= 20.0:
        return 1.0 - 0.5 * (reprojection_error - 5.0) / 15.0
    if reprojection_error <= 50.0:
        return 0.5 - 0.5 * (reprojection_error - 20.0) / 30.0
    return 0.0


def compute_homographies(
    all_keypoints: list[list[tuple[float | None, float | None]]],
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    max_carry_forward: int = MAX_CARRY_FORWARD_FRAMES,
) -> list[FrameHomography]:
    """
    Compute per-frame homographies with temporal stabilization and confidence scoring.

    Args:
        all_keypoints: per-frame keypoint lists from CourtDetector.predict_video()
        confidence_threshold: frames below this are flagged unreliable
        max_carry_forward: max frames to carry a reliable homography forward

    Returns:
        List of FrameHomography, one per input frame.
    """
    court_ref = CourtReference()
    conf_indices = court_ref.get_conf_indices()
    refer_kps = court_ref.get_keypoints_array()

    results: list[FrameHomography] = []
    last_reliable_matrix: np.ndarray | None = None
    last_reliable_inverse: np.ndarray | None = None
    frames_since_reliable = 0

    for idx, kps in enumerate(all_keypoints):
        detected_count = sum(1 for x, y in kps if x is not None)
        matrix, reproj_err = _build_homography(kps, court_ref, conf_indices, refer_kps)

        if matrix is not None:
            confidence = _error_to_confidence(reproj_err)
        else:
            confidence = 0.0
            reproj_err = float("inf")

        carried = False
        inv_matrix = None

        if matrix is not None and confidence >= confidence_threshold:
            try:
                inv_matrix = np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                inv_matrix = None
            last_reliable_matrix = matrix
            last_reliable_inverse = inv_matrix
            frames_since_reliable = 0
        elif last_reliable_matrix is not None and frames_since_reliable < max_carry_forward:
            matrix = last_reliable_matrix
            inv_matrix = last_reliable_inverse
            carried = True
            frames_since_reliable += 1
        else:
            matrix = None
            inv_matrix = None
            frames_since_reliable += 1

        reliable = matrix is not None and (confidence >= confidence_threshold or carried)

        results.append(FrameHomography(
            frame_idx=idx,
            matrix=matrix,
            inverse_matrix=inv_matrix,
            reprojection_error=reproj_err,
            confidence=confidence,
            reliable=reliable,
            detected_count=detected_count,
            carried_forward=carried,
        ))

    total = len(results)
    reliable_count = sum(1 for r in results if r.reliable)
    carried_count = sum(1 for r in results if r.carried_forward)
    unreliable_count = total - reliable_count

    logger.info(
        f"Homography summary: {total} frames — "
        f"{reliable_count} reliable ({reliable_count/total*100:.0f}%), "
        f"{carried_count} carried forward, "
        f"{unreliable_count} unreliable"
    )

    return results


def pixel_to_court(
    x_pixel: float,
    y_pixel: float,
    homography: FrameHomography,
) -> tuple[float | None, float | None]:
    """
    Transform a pixel coordinate to court-space using the frame's inverse homography.

    Returns (court_x, court_y) or (None, None) if the frame has no valid homography.
    """
    if homography.inverse_matrix is None:
        return None, None

    pt = np.array([[[x_pixel, y_pixel]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, homography.inverse_matrix)
    cx, cy = transformed[0][0]
    return float(cx), float(cy)


def court_to_pixel(
    court_x: float,
    court_y: float,
    homography: FrameHomography,
) -> tuple[float | None, float | None]:
    """
    Transform a court-space coordinate to pixel using the frame's homography.

    Returns (pixel_x, pixel_y) or (None, None) if no valid homography.
    """
    if homography.matrix is None:
        return None, None

    pt = np.array([[[court_x, court_y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, homography.matrix)
    px, py = transformed[0][0]
    return float(px), float(py)
