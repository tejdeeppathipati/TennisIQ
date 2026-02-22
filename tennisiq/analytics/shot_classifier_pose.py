"""
YOLOv8 pose-based shot type classifier.

Uses per-shot player crop and keypoints to classify:
  - serve
  - forehand
  - backhand
Fallback remains "neutral" when keypoint confidence is low.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SHOT_CLASSES = ["backhand", "forehand", "neutral", "serve"]


@dataclass
class PoseShotClassification:
    shot_type: str
    confidence: float
    probabilities: dict[str, float]
    method: str = "pose"


_POSE_MODEL = None


def _get_pose_model():
    global _POSE_MODEL
    if _POSE_MODEL is None:
        from ultralytics import YOLO

        _POSE_MODEL = YOLO("yolov8n-pose.pt")
    return _POSE_MODEL


def _probs(main: str, conf: float) -> dict[str, float]:
    remainder = (1.0 - conf) / max(len(SHOT_CLASSES) - 1, 1)
    return {c: (conf if c == main else remainder) for c in SHOT_CLASSES}


def classify_shot_from_pose(
    frame_bgr: np.ndarray,
    player_bbox: tuple[float, float, float, float] | None,
) -> PoseShotClassification:
    """
    Classify shot type from YOLOv8 pose keypoints on a player crop.

    Heuristic:
      - Serve: wrist clearly above shoulder
      - Forehand/Backhand: wrist horizontal side vs shoulder
    """
    if player_bbox is None or frame_bgr is None or frame_bgr.size == 0:
        return PoseShotClassification("neutral", 0.0, _probs("neutral", 0.0), method="pose_missing_bbox")

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = player_bbox
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return PoseShotClassification("neutral", 0.0, _probs("neutral", 0.0), method="pose_bad_crop")

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return PoseShotClassification("neutral", 0.0, _probs("neutral", 0.0), method="pose_empty_crop")

    try:
        model = _get_pose_model()
        results = model(crop, verbose=False)
        if results is None or len(results) == 0:
            return PoseShotClassification("neutral", 0.0, _probs("neutral", 0.0), method="pose_no_results")

        kp = results[0].keypoints
        if kp is None or kp.xy is None or len(kp.xy) == 0:
            return PoseShotClassification("neutral", 0.0, _probs("neutral", 0.0), method="pose_no_keypoints")

        kpts = kp.xy[0].cpu().numpy()
        conf = kp.conf[0].cpu().numpy() if kp.conf is not None else np.ones(len(kpts), dtype=np.float32) * 0.3

        # Prefer the better-visible side.
        # COCO keypoints:
        # left_shoulder=5, right_shoulder=6, left_elbow=7, right_elbow=8, left_wrist=9, right_wrist=10
        right_vis = float(min(conf[6], conf[8], conf[10]))
        left_vis = float(min(conf[5], conf[7], conf[9]))

        if max(right_vis, left_vis) < 0.5:
            return PoseShotClassification("neutral", 0.2, _probs("neutral", 0.2), method="pose_low_conf")

        if right_vis >= left_vis:
            wrist = kpts[10]
            shoulder = kpts[6]
            side_vis = right_vis
        else:
            wrist = kpts[9]
            shoulder = kpts[5]
            side_vis = left_vis

        # Serve: wrist significantly above shoulder.
        if wrist[1] < shoulder[1] * 0.9:
            conf_val = 0.85 if side_vis >= 0.7 else 0.7
            return PoseShotClassification("serve", conf_val, _probs("serve", conf_val))

        # Forehand/backhand by horizontal wrist side relative to shoulder.
        if wrist[0] > shoulder[0]:
            conf_val = 0.78 if side_vis >= 0.7 else 0.65
            return PoseShotClassification("forehand", conf_val, _probs("forehand", conf_val))
        conf_val = 0.78 if side_vis >= 0.7 else 0.65
        return PoseShotClassification("backhand", conf_val, _probs("backhand", conf_val))
    except Exception:
        return PoseShotClassification("neutral", 0.1, _probs("neutral", 0.1), method="pose_error")
