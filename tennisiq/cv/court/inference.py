"""
Court keypoint inference — loads CourtKeypointNet and detects 14 keypoints per frame.

Input:  BGR frame (any resolution)
Output: list of 14 (x, y) keypoint coordinates in original pixel space, or (None, None) if not detected.
"""
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from .model import CourtKeypointNet

logger = logging.getLogger(__name__)

MODEL_INPUT_W = 640
MODEL_INPUT_H = 360
NUM_KEYPOINTS = 14
DEFAULT_BATCH_SIZE = 16


def postprocess_heatmap(
    heatmap: np.ndarray,
    low_thresh: int = 155,
    min_radius: int = 10,
    max_radius: int = 30,
) -> tuple[float | None, float | None]:
    """Extract keypoint (x, y) from a single heatmap channel via thresholding + Hough circles."""
    _, binary = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        binary, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=2, minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is not None:
        return float(circles[0][0][0]), float(circles[0][0][1])
    return None, None


def _decode_heatmaps(
    heatmaps: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> list[tuple[float | None, float | None]]:
    """Convert raw model heatmaps (C, H, W) into 14 keypoint (x, y) tuples."""
    keypoints: list[tuple[float | None, float | None]] = []
    for i in range(NUM_KEYPOINTS):
        hm = (heatmaps[i] * 255).clip(0, 255).astype(np.uint8)
        x, y = postprocess_heatmap(hm)
        if x is not None and y is not None:
            keypoints.append((x * scale_x, y * scale_y))
        else:
            keypoints.append((None, None))
    return keypoints


class CourtDetector:
    """Loads CourtKeypointNet and runs keypoint inference on video frames."""

    def __init__(self, checkpoint_path: str, device: str | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model = CourtKeypointNet(out_channels=15)

        cp = Path(checkpoint_path)
        if not cp.exists():
            raise FileNotFoundError(f"Court checkpoint not found: {cp}")

        state = torch.load(str(cp), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"CourtKeypointNet loaded from {cp} on {self.device}")

    def predict_frame(self, frame_bgr: np.ndarray) -> list[tuple[float | None, float | None]]:
        """
        Run court keypoint detection on a single BGR frame.

        Returns:
            List of 14 (x, y) tuples in original frame pixel coordinates.
            Each entry is (None, None) if that keypoint was not detected.
        """
        orig_h, orig_w = frame_bgr.shape[:2]
        scale_x = orig_w / MODEL_INPUT_W
        scale_y = orig_h / MODEL_INPUT_H

        resized = cv2.resize(frame_bgr, (MODEL_INPUT_W, MODEL_INPUT_H))
        inp = resized.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))  # HWC → CHW
        inp = torch.from_numpy(inp).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(inp)

        heatmaps = out.squeeze(0).cpu().numpy()
        return _decode_heatmaps(heatmaps, scale_x, scale_y)

    def predict_batch(
        self, frames_bgr: list[np.ndarray],
    ) -> list[list[tuple[float | None, float | None]]]:
        """
        Run court keypoint detection on a batch of BGR frames in a single forward pass.
        All frames must have the same resolution.
        """
        if not frames_bgr:
            return []

        orig_h, orig_w = frames_bgr[0].shape[:2]
        scale_x = orig_w / MODEL_INPUT_W
        scale_y = orig_h / MODEL_INPUT_H

        batch_np = np.stack([
            np.transpose(
                cv2.resize(f, (MODEL_INPUT_W, MODEL_INPUT_H)).astype(np.float32) / 255.0,
                (2, 0, 1),
            )
            for f in frames_bgr
        ])
        batch_tensor = torch.from_numpy(batch_np).to(self.device)

        with torch.no_grad():
            out = self.model(batch_tensor)

        all_heatmaps = out.cpu().numpy()  # (B, 15, H, W)

        return [
            _decode_heatmaps(all_heatmaps[i], scale_x, scale_y)
            for i in range(len(frames_bgr))
        ]

    def predict_video(
        self,
        video_path: str,
        start_sec: float | None = None,
        end_sec: float | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress_callback=None,
    ) -> list[list[tuple[float | None, float | None]]]:
        """
        Run court keypoint detection on frames of a video.

        Args:
            video_path: path to the video file
            start_sec: if set, seek to this timestamp before processing
            end_sec: if set, stop processing after this timestamp
            batch_size: number of frames per forward pass (higher = faster but more RAM)
            progress_callback: optional callable(frames_done, total_frames)

        Returns:
            List of per-frame keypoint lists (one per frame in video order).
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

        logger.info(
            f"Processing frames {start_frame}–{end_frame} "
            f"({total_to_process} frames, batch_size={batch_size})"
        )

        all_keypoints: list[list[tuple[float | None, float | None]]] = []
        frames_done = 0

        while frames_done < total_to_process:
            batch_frames = []
            for _ in range(min(batch_size, total_to_process - frames_done)):
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)

            if not batch_frames:
                break

            batch_kps = self.predict_batch(batch_frames)
            all_keypoints.extend(batch_kps)
            frames_done += len(batch_frames)

            if progress_callback:
                progress_callback(frames_done, total_to_process)

        cap.release()
        logger.info(f"Court detection complete: {frames_done} frames processed")
        return all_keypoints
