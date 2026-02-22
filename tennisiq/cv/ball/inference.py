"""
Ball tracking inference — FR-10 through FR-15.

FR-10: 3-frame temporal stacking + BallTrackerNet inference
FR-11: Heatmap postprocessing via peak detection + Hough circles
FR-12: Outlier removal (inter-frame distance threshold)
FR-13: Gap interpolation within sub-tracks
FR-14: Court-space projection via homography
FR-15: Ball speed + acceleration via finite differences
"""
import logging
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import distance

from .model import BallTrackerNet

logger = logging.getLogger(__name__)

MODEL_INPUT_W = 640
MODEL_INPUT_H = 360
OUTLIER_MAX_DIST = 100
MIN_SUBTRACK_LENGTH = 5
MAX_INTERP_GAP = 4
MAX_DIST_GAP = 80
# Physics cap: fastest recorded tennis serve is ~73 m/s (263 km/h).
# Anything above 80 m/s is a detection artifact (scene cut, interpolation error).
MAX_PLAUSIBLE_SPEED_M_S = 80.0


# ─── FR-11: Postprocessing ───────────────────────────────────────────────────

def postprocess_ball_heatmap(
    feature_map: np.ndarray,
    scale: int = 2,
) -> tuple[float | None, float | None]:
    """Extract ball (x, y) from the argmax feature map via thresholding + Hough circles."""
    # argmax output is already 0-255 (class index = heatmap intensity)
    fm = feature_map.astype(np.uint8).reshape((MODEL_INPUT_H, MODEL_INPUT_W))
    _, binary = cv2.threshold(fm, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        binary, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
        param1=50, param2=2, minRadius=2, maxRadius=7,
    )
    if circles is not None and len(circles[0]) > 0:
        best = min(circles[0], key=lambda c: c[2])
        x = float(best[0]) * scale
        y = float(best[1]) * scale
        return x, y
    return None, None


# ─── FR-12: Outlier removal ──────────────────────────────────────────────────

def remove_outliers(
    ball_track: list[tuple[float | None, float | None]],
    dists: list[float],
    max_dist: float = OUTLIER_MAX_DIST,
) -> list[tuple[float | None, float | None]]:
    """Remove detections where inter-frame distance exceeds threshold."""
    track = list(ball_track)
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if i + 1 < len(dists):
            if dists[i + 1] > max_dist or dists[i + 1] == -1:
                track[i] = (None, None)
            elif i > 0 and dists[i - 1] == -1:
                track[i - 1] = (None, None)
    return track


# ─── FR-13: Sub-track splitting + interpolation ──────────────────────────────

def split_track(
    ball_track: list[tuple[float | None, float | None]],
    max_gap: int = MAX_INTERP_GAP,
    max_dist_gap: float = MAX_DIST_GAP,
    min_track: int = MIN_SUBTRACK_LENGTH,
) -> list[list[int]]:
    """Split ball track into sub-tracks suitable for interpolation."""
    list_det = [0 if x[0] is not None else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []

    for i, (k, length) in enumerate(groups):
        if k == 1 and i > 0 and i < len(groups) - 1:
            pt_before = ball_track[cursor - 1]
            pt_after = ball_track[cursor + length]
            if pt_before[0] is not None and pt_after[0] is not None:
                dist = distance.euclidean(pt_before, pt_after)
            else:
                dist = float("inf")

            if length >= max_gap or (length > 0 and dist / length > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                min_value = cursor + length - 1
        cursor += length

    if len(ball_track) - min_value > min_track:
        result.append([min_value, len(ball_track)])

    return result


def interpolate_subtrack(
    coords: list[tuple[float | None, float | None]],
) -> list[tuple[float, float]]:
    """Fill gaps in a sub-track via linear interpolation."""
    x = np.array([c[0] if c[0] is not None else np.nan for c in coords])
    y = np.array([c[1] if c[1] is not None else np.nan for c in coords])

    def _interp(arr):
        nans = np.isnan(arr)
        if nans.all() or (~nans).sum() < 2:
            return arr
        indices = np.arange(len(arr))
        arr[nans] = np.interp(indices[nans], indices[~nans], arr[~nans])
        return arr

    x = _interp(x)
    y = _interp(y)

    return list(zip(x.tolist(), y.tolist()))


# ─── Track cleaning (outlier removal + interpolation) ────────────────────────

def clean_ball_track(
    ball_track: list[tuple[float | None, float | None]],
    max_outlier_dist: float = OUTLIER_MAX_DIST,
) -> list[tuple[float | None, float | None]]:
    """
    Apply outlier removal and gap interpolation to any ball track.

    Works with tracks from BallTrackerNet or YOLO — just needs a list of
    (x, y) or (None, None) per frame in pixel coordinates.
    """
    n = len(ball_track)
    if n < 3:
        return ball_track

    dists = [-1.0]
    for i in range(1, n):
        if ball_track[i][0] is not None and ball_track[i - 1][0] is not None:
            d = distance.euclidean(ball_track[i], ball_track[i - 1])
        else:
            d = -1.0
        dists.append(d)

    track = remove_outliers(ball_track, dists, max_dist=max_outlier_dist)

    subtracks = split_track(track)
    for r in subtracks:
        sub = track[r[0]:r[1]]
        interpolated = interpolate_subtrack(sub)
        track[r[0]:r[1]] = interpolated

    before = sum(1 for x, y in ball_track if x is not None)
    after = sum(1 for x, y in track if x is not None)
    logger.info(
        f"Ball track cleaned: {before} raw detections → {after} after "
        f"outlier removal + interpolation ({n} frames)"
    )

    return track


# ─── FR-15: Speed + acceleration ─────────────────────────────────────────────

@dataclass
class BallPhysics:
    """Per-frame ball physics in court-space."""
    frame_idx: int
    pixel_xy: tuple[float | None, float | None]
    court_xy: tuple[float | None, float | None]
    speed_m_per_s: float | None
    accel_m_per_s2: float | None


def compute_ball_physics(
    ball_track: list[tuple[float | None, float | None]],
    homographies: list | None = None,
    fps: float = 30.0,
    court_scale: float | None = None,
) -> list[BallPhysics]:
    """
    FR-14 + FR-15: Project ball positions to court-space and compute speed/acceleration.

    Args:
        ball_track: per-frame (x_pixel, y_pixel) or (None, None)
        homographies: per-frame FrameHomography objects (from geometry module)
        fps: video frame rate
        court_scale: meters per court-reference unit. If None, uses the ITF standard
                     derived from CourtReference (10.97m / 1117 units ≈ 0.00982).

    Returns:
        List of BallPhysics, one per frame.
    """
    if court_scale is None:
        from tennisiq.geometry.court_reference import CourtReference
        court_scale = CourtReference().meters_per_unit

    n = len(ball_track)
    court_positions: list[tuple[float | None, float | None]] = []

    if homographies is not None:
        from tennisiq.geometry.homography import pixel_to_court
        for i, (px, py) in enumerate(ball_track):
            if px is not None and i < len(homographies) and homographies[i].reliable:
                cx, cy = pixel_to_court(px, py, homographies[i])
                court_positions.append((cx, cy))
            else:
                court_positions.append((None, None))
    else:
        court_positions = [(None, None)] * n

    dt = 1.0 / fps
    speeds: list[float | None] = [None] * n
    accels: list[float | None] = [None] * n

    for i in range(1, n):
        c0 = court_positions[i - 1]
        c1 = court_positions[i]
        if c0[0] is not None and c1[0] is not None:
            dx = (c1[0] - c0[0]) * court_scale
            dy = (c1[1] - c0[1]) * court_scale
            dist = np.sqrt(dx ** 2 + dy ** 2)
            raw_speed = dist / dt
            if raw_speed <= MAX_PLAUSIBLE_SPEED_M_S:
                speeds[i] = raw_speed
            # else: leave as None — physically impossible, likely scene cut artifact

    for i in range(1, n):
        if speeds[i] is not None and speeds[i - 1] is not None:
            accels[i] = (speeds[i] - speeds[i - 1]) / dt

    results = []
    for i in range(n):
        results.append(BallPhysics(
            frame_idx=i,
            pixel_xy=ball_track[i],
            court_xy=court_positions[i],
            speed_m_per_s=round(speeds[i], 2) if speeds[i] is not None else None,
            accel_m_per_s2=round(accels[i], 2) if accels[i] is not None else None,
        ))

    return results


# ─── FR-10: Main detector class ──────────────────────────────────────────────

class BallDetector:
    """Loads BallTrackerNet and runs ball tracking on video frames."""

    def __init__(self, checkpoint_path: str, device: str | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model = BallTrackerNet(out_channels=256)

        cp = Path(checkpoint_path)
        if not cp.exists():
            raise FileNotFoundError(f"Ball checkpoint not found: {cp}")

        state = torch.load(str(cp), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"BallTrackerNet loaded from {cp} on {self.device}")

    def predict_video(
        self,
        video_path: str,
        start_sec: float | None = None,
        end_sec: float | None = None,
        interpolate: bool = True,
        progress_callback=None,
    ) -> tuple[list[tuple[float | None, float | None]], list[float]]:
        """
        Run ball tracking on a video segment.

        Uses 3-frame temporal stacking: for each frame N, the model sees
        frames [N, N-1, N-2] concatenated along the channel axis.

        Returns:
            (ball_track, inter_frame_dists)
            ball_track: list of (x, y) in original pixel coords or (None, None)
            inter_frame_dists: list of distances between consecutive detections (-1 if unavailable)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_sec * fps) if start_sec is not None else 0
        end_frame = int(end_sec * fps) if end_sec is not None else total_video_frames
        start_frame = max(0, min(start_frame, total_video_frames))
        end_frame = max(start_frame, min(end_frame, total_video_frames))
        total_to_process = end_frame - start_frame

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale_x = orig_w / MODEL_INPUT_W
        scale_y = orig_h / MODEL_INPUT_H

        logger.info(f"Ball tracking: frames {start_frame}-{end_frame} ({total_to_process} frames)")

        frames_buffer: list[np.ndarray] = []
        ball_track: list[tuple[float | None, float | None]] = []
        dists: list[float] = []
        frames_read = 0

        while frames_read < total_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, (MODEL_INPUT_W, MODEL_INPUT_H))
            frames_buffer.append(resized)
            if len(frames_buffer) > 3:
                frames_buffer.pop(0)
            frames_read += 1

            if len(frames_buffer) < 3:
                ball_track.append((None, None))
                dists.append(-1)
                continue

            img = frames_buffer[-1]
            img_prev = frames_buffer[-2]
            img_preprev = frames_buffer[-3]

            stacked = np.concatenate((img, img_prev, img_preprev), axis=2)
            stacked = stacked.astype(np.float32) / 255.0
            stacked = np.rollaxis(stacked, 2, 0)
            inp = torch.from_numpy(stacked).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(inp, testing=True)

            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = postprocess_ball_heatmap(output[0], scale=1)

            if x_pred is not None:
                x_pred *= scale_x
                y_pred *= scale_y

            ball_track.append((x_pred, y_pred))

            if ball_track[-1][0] is not None and ball_track[-2][0] is not None:
                d = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                d = -1
            dists.append(d)

            if progress_callback and frames_read % 50 == 0:
                progress_callback(frames_read, total_to_process)

        cap.release()

        ball_track = remove_outliers(ball_track, dists)

        if interpolate:
            subtracks = split_track(ball_track)
            for r in subtracks:
                sub = ball_track[r[0]:r[1]]
                interpolated = interpolate_subtrack(sub)
                ball_track[r[0]:r[1]] = interpolated

        detected = sum(1 for x, y in ball_track if x is not None)
        logger.info(
            f"Ball tracking complete: {frames_read} frames, "
            f"{detected} with ball detected ({detected/frames_read*100:.0f}%)"
        )

        return ball_track, dists
