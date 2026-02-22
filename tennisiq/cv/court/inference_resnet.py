"""
Court keypoint inference using ResNet50 regression model.

Input:  BGR frame (any resolution)
Output: list of 14 (x, y) keypoint coordinates in original pixel space.

Unlike the heatmap-based CourtKeypointNet, this model directly regresses
14 (x, y) pairs (28 values) from a 224x224 input image using a ResNet50 backbone.
"""
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from .model_resnet import build_court_resnet

logger = logging.getLogger(__name__)

RESNET_INPUT_SIZE = 224
NUM_KEYPOINTS = 14
DEFAULT_BATCH_SIZE = 32

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class CourtDetectorResNet:
    """Loads ResNet50 keypoint model and runs regression inference on video frames."""

    def __init__(self, checkpoint_path: str, device: str | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model = build_court_resnet(num_keypoints=NUM_KEYPOINTS)

        cp = Path(checkpoint_path)
        if not cp.exists():
            raise FileNotFoundError(f"Court ResNet checkpoint not found: {cp}")

        state = torch.load(str(cp), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        logger.info(f"CourtDetectorResNet loaded from {cp} on {self.device}")

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """BGR frame → normalized tensor ready for ResNet."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        return self.transform(tensor)

    def predict_frame(self, frame_bgr: np.ndarray) -> list[tuple[float | None, float | None]]:
        orig_h, orig_w = frame_bgr.shape[:2]
        scale_x = orig_w / RESNET_INPUT_SIZE
        scale_y = orig_h / RESNET_INPUT_SIZE

        inp = self._preprocess(frame_bgr).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(inp)

        coords = out.squeeze(0).cpu().numpy()
        keypoints = []
        for i in range(NUM_KEYPOINTS):
            x = float(coords[i * 2]) * scale_x
            y = float(coords[i * 2 + 1]) * scale_y
            keypoints.append((x, y))
        return keypoints

    def predict_batch(
        self, frames_bgr: list[np.ndarray],
    ) -> list[list[tuple[float | None, float | None]]]:
        if not frames_bgr:
            return []

        orig_h, orig_w = frames_bgr[0].shape[:2]
        scale_x = orig_w / RESNET_INPUT_SIZE
        scale_y = orig_h / RESNET_INPUT_SIZE

        batch = torch.stack([self._preprocess(f) for f in frames_bgr]).to(self.device)

        with torch.no_grad():
            out = self.model(batch)

        all_coords = out.cpu().numpy()
        results = []
        for b in range(len(frames_bgr)):
            keypoints = []
            for i in range(NUM_KEYPOINTS):
                x = float(all_coords[b, i * 2]) * scale_x
                y = float(all_coords[b, i * 2 + 1]) * scale_y
                keypoints.append((x, y))
            results.append(keypoints)
        return results

    def predict_video(
        self,
        video_path: str,
        start_sec: float | None = None,
        end_sec: float | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress_callback=None,
    ) -> list[list[tuple[float | None, float | None]]]:
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
            f"[ResNet] Processing frames {start_frame}-{end_frame} "
            f"({total_to_process} frames, batch_size={batch_size})"
        )

        all_keypoints = []
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
        logger.info(f"[ResNet] Court detection complete: {frames_done} frames processed")
        return all_keypoints
