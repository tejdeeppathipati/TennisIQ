"""
Tennis ball detection using YOLOv5 fine-tuned model.

Replaces BallTrackerNet with a YOLOv5L6u model that was fine-tuned
specifically for tennis ball detection (single class: "tennis ball").

The model outputs bounding boxes, so ball position is the box center.
"""
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BallDetectionYOLO:
    """Single ball detection from YOLO."""
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    center_xy: tuple[float, float]
    confidence: float


class BallDetectorYOLO:
    """Tennis ball detector using YOLOv5 fine-tuned weights."""

    def __init__(self, checkpoint_path: str, device: str | None = None):
        from ultralytics import YOLO

        cp = Path(checkpoint_path)
        if not cp.exists():
            raise FileNotFoundError(f"YOLOv5 ball checkpoint not found: {cp}")

        self.model = YOLO(str(cp))
        if device:
            self.model.to(device)
        self.device = device

        logger.info(f"BallDetectorYOLO loaded from {cp} (device={device})")
        logger.info(f"  Classes: {self.model.names}")

    def detect_frame(
        self, frame_bgr: np.ndarray, conf_threshold: float = 0.15,
    ) -> BallDetectionYOLO | None:
        """Detect the tennis ball in a single frame. Returns best detection or None."""
        results = self.model(frame_bgr, conf=conf_threshold, verbose=False)

        best = None
        best_conf = 0.0
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0].cpu())
                if conf > best_conf:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    best = BallDetectionYOLO(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        center_xy=(float(cx), float(cy)),
                        confidence=conf,
                    )
                    best_conf = conf
        return best

    def detect_frame_all(
        self, frame_bgr: np.ndarray, conf_threshold: float = 0.15,
    ) -> list[BallDetectionYOLO]:
        """Return all ball detections in a frame (for visualization)."""
        results = self.model(frame_bgr, conf=conf_threshold, verbose=False)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu())
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                dets.append(BallDetectionYOLO(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    center_xy=(float(cx), float(cy)),
                    confidence=conf,
                ))
        return dets

    def detect_video(
        self,
        video_path: str,
        start_sec: float | None = None,
        end_sec: float | None = None,
        conf_threshold: float = 0.15,
        progress_callback=None,
    ) -> tuple[list[tuple[float | None, float | None]], list[BallDetectionYOLO | None]]:
        """
        Detect tennis ball across video frames.

        Returns:
            (ball_track, raw_detections)
            ball_track: list of (x, y) center coords or (None, None) per frame
            raw_detections: list of BallDetectionYOLO or None per frame
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

        logger.info(f"[YOLO5] Ball detection: frames {start_frame}-{end_frame} ({total_to_process} frames)")

        ball_track = []
        raw_detections = []
        frames_read = 0

        while frames_read < total_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            det = self.detect_frame(frame, conf_threshold=conf_threshold)
            if det is not None:
                ball_track.append(det.center_xy)
            else:
                ball_track.append((None, None))
            raw_detections.append(det)

            frames_read += 1
            if progress_callback and frames_read % 50 == 0:
                progress_callback(frames_read, total_to_process)

        cap.release()

        detected = sum(1 for x, y in ball_track if x is not None)
        logger.info(
            f"[YOLO5] Ball detection complete: {frames_read} frames, "
            f"{detected} with ball ({detected/max(frames_read,1)*100:.0f}%)"
        )

        return ball_track, raw_detections
