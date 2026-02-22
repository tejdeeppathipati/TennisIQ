"""
MoveNet Lightning pose estimation for player bounding boxes.

Extracts 17 keypoints per player per frame using TensorFlow Lite.
Falls back to a stub that returns None if tflite-runtime is unavailable,
allowing the rest of the pipeline to run without pose data.

Keypoints (COCO ordering):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""
import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MOVENET_INPUT_SIZE = 192

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Subset used for shot classification: upper body + hips (13 keypoints → 26 features)
# Matches the trained RNN model's expected input dimension.
SHOT_KEYPOINT_INDICES = list(range(13))  # nose(0)..right_hip(12)


@dataclass
class PlayerPose:
    """Pose estimation result for a single player in a single frame."""
    frame_idx: int
    player: str                                  # "player_a" or "player_b"
    keypoints: np.ndarray                        # shape (17, 3) — y, x, confidence per keypoint
    bbox: tuple[float, float, float, float]      # original bounding box


class MoveNetDetector:
    """MoveNet Lightning pose estimator using TFLite."""

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.interpreter = None
        self._available = False

        if model_path and os.path.isfile(model_path):
            try:
                interp = None
                for loader in [
                    lambda: __import__("ai_edge_litert.interpreter", fromlist=["Interpreter"]).Interpreter(model_path=model_path),
                    lambda: __import__("tflite_runtime.interpreter", fromlist=["Interpreter"]).Interpreter(model_path=model_path),
                    lambda: __import__("tensorflow", fromlist=["lite"]).lite.Interpreter(model_path=model_path),
                ]:
                    try:
                        interp = loader()
                        break
                    except (ImportError, Exception) as e:
                        logger.debug(f"TFLite loader failed: {e}")
                        continue
                if interp is None:
                    raise ImportError("No TFLite runtime available (tried ai-edge-litert, tflite-runtime, tensorflow)")
                interp.allocate_tensors()
                self.interpreter = interp
                self._available = True
                logger.info(f"MoveNet loaded from {model_path}")
            except Exception as e:
                logger.warning(f"MoveNet failed to load: {e}; pose estimation disabled")
        else:
            logger.warning(f"MoveNet model not found at {model_path}; pose estimation disabled")

    @property
    def available(self) -> bool:
        return self._available

    def estimate_pose(self, frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
        """
        Estimate pose for a single player crop.

        Args:
            frame_bgr: full frame in BGR
            bbox: (x1, y1, x2, y2) bounding box

        Returns:
            (17, 3) array of (y, x, confidence) in normalized [0,1] coordinates
            relative to the bounding box, or None if unavailable.
        """
        if not self._available:
            return None

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(crop_rgb, (MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE))

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        expected_dtype = input_details[0]["dtype"]
        input_data = np.expand_dims(resized.astype(expected_dtype), axis=0)

        self.interpreter.set_tensor(input_details[0]["index"], input_data)
        self.interpreter.invoke()

        keypoints = self.interpreter.get_tensor(output_details[0]["index"])
        return keypoints[0, 0]  # shape (17, 3)

    def estimate_video(
        self,
        video_path: str,
        player_results: list,
        start_sec: float = 0.0,
        end_sec: float | None = None,
        progress_callback=None,
    ) -> dict[int, dict[str, PlayerPose]]:
        """
        Run pose estimation on all detected players across the video.

        Returns:
            dict mapping frame_idx -> {"player_a": PlayerPose, "player_b": PlayerPose}
        """
        if not self._available:
            logger.warning("MoveNet not available; returning empty poses")
            return {}

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_sec * fps) if start_sec > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        total_frames = len(player_results)
        poses: dict[int, dict[str, PlayerPose]] = {}

        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            pr = player_results[idx]
            frame_poses = {}

            for label, player in [("player_a", pr.player_a), ("player_b", pr.player_b)]:
                if player is None:
                    continue

                kps = self.estimate_pose(frame, player.bbox)
                if kps is not None:
                    frame_poses[label] = PlayerPose(
                        frame_idx=idx,
                        player=label,
                        keypoints=kps,
                        bbox=player.bbox,
                    )

            if frame_poses:
                poses[idx] = frame_poses

            if progress_callback and (idx + 1) % 50 == 0:
                progress_callback(idx + 1, total_frames)

        cap.release()

        if progress_callback:
            progress_callback(total_frames, total_frames)

        logger.info(f"Pose estimation: {len(poses)} frames with poses out of {total_frames}")
        return poses


NUM_POSE_FEATURES = len(SHOT_KEYPOINT_INDICES) * 2   # 13 × 2 = 26


def extract_pose_features(pose: PlayerPose) -> np.ndarray:
    """
    Extract the feature vector used by the shot classifier.

    Returns a flat array of (y, x) for the 13 upper-body keypoints = 26 values.
    Low-confidence keypoints are zeroed out.
    """
    features = []
    for idx in SHOT_KEYPOINT_INDICES:
        y, x, conf = pose.keypoints[idx]
        if conf > 0.1:
            features.extend([y, x])
        else:
            features.extend([0.0, 0.0])
    return np.array(features, dtype=np.float32)


def extract_pose_sequence(
    poses: dict[int, dict[str, PlayerPose]],
    player: str,
    center_frame: int,
    window: int = 15,
) -> np.ndarray | None:
    """
    Extract a temporal sequence of pose features for RNN classification.

    Returns shape (30, 26) — 30 frames × 26 features.
    The window covers center_frame-window to center_frame+window-1.
    """
    seq = []
    for fi in range(center_frame - window, center_frame + window):
        if fi in poses and player in poses[fi]:
            seq.append(extract_pose_features(poses[fi][player]))
        else:
            seq.append(np.zeros(NUM_POSE_FEATURES, dtype=np.float32))

    return np.stack(seq)
