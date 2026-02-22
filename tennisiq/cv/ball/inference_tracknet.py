"""
Tennis ball detection using TrackNetV3 — a UNet-style encoder-decoder with
motion prompts and skip connections.

Takes 3 consecutive RGB frames (9-channel input) at 640×360 and outputs a
3-channel heatmap (one per input frame). Ball position is the peak of the
middle frame's channel.

Architecture matches the model_best.pt checkpoint:
  enc1..enc4  : encoder (9→64→128→256→512)
  dec1..dec3  : decoder with skip connections (skip-concat then conv-down)
  output      : 3×1×1 conv → 3 heatmap channels
  motion_prompt.a/b : learnable motion-attention scalars
"""
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MODEL_INPUT_W = 640
MODEL_INPUT_H = 360


# ── Model architecture ────────────────────────────────────────────────────────
# The checkpoint saves enc1/enc2/... as nn.Sequential directly, so the keys
# look like "enc1.0.weight", "enc1.1.weight", etc. (no ".block." wrapper).

def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two Conv-BN-ReLU blocks — matches checkpoint key pattern enc*.0..enc*.4"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),   # .0
        nn.BatchNorm2d(out_ch),                                # .1
        nn.ReLU(inplace=True),                                 # .2  (no params)
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),   # .3
        nn.BatchNorm2d(out_ch),                                # .4
        nn.ReLU(inplace=True),                                 # .5  (no params)
    )


def _triple_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """Three Conv-BN-ReLU blocks — matches checkpoint key pattern enc*.0..enc*.7"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),   # .0
        nn.BatchNorm2d(out_ch),                                # .1
        nn.ReLU(inplace=True),                                 # .2
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),   # .3
        nn.BatchNorm2d(out_ch),                                # .4
        nn.ReLU(inplace=True),                                 # .5
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),   # .6
        nn.BatchNorm2d(out_ch),                                # .7
        nn.ReLU(inplace=True),                                 # .8
    )


class TrackNetV3(nn.Module):
    """
    UNet-style TrackNet with motion prompts and skip connections.

    Input:  (B, 9, H, W)  — 3 stacked RGB frames
    Output: (B, 3, H, W)  — heatmap per frame (sigmoid-activated)

    State-dict key pattern mirrors the original training code where
    enc1/enc2/dec1/dec2/dec3 are stored as nn.Sequential directly, so
    parameter keys are e.g. "enc1.0.weight", "enc1.4.running_mean", etc.
    """

    def __init__(self):
        super().__init__()
        # learnable motion-attention scalars
        self.motion_prompt = nn.ParameterDict({
            "a": nn.Parameter(torch.ones(1)),
            "b": nn.Parameter(torch.zeros(1)),
        })

        # Encoder — stored as plain nn.Sequential in checkpoint
        self.enc1 = _double_conv(9, 64)            # (B,9,H,W)   → (B,64,H,W)
        self.enc2 = _double_conv(64, 128)           # after pool1 → (B,128,H/2,W/2)
        self.enc3 = _triple_conv(128, 256)          # after pool2 → (B,256,H/4,W/4)
        self.enc4 = _triple_conv(256, 512)          # after pool3 → (B,512,H/8,W/8)

        self.pool = nn.MaxPool2d(2, 2)

        # Decoder — each takes upsampled + skip concatenated
        self.dec1 = _triple_conv(512 + 256, 256)   # 768  → 256
        self.dec2 = _double_conv(256 + 128, 128)   # 384  → 128
        self.dec3 = _double_conv(128 + 64, 64)     # 192  → 64

        self.output = nn.Conv2d(64, 3, 1)           # 1×1 conv → 3 channels

    def forward(self, x):
        # Motion prompt: scalar affine on input
        x = self.motion_prompt["a"] * x + self.motion_prompt["b"]

        # Encoder
        e1 = self.enc1(x)                    # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))        # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))        # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))        # (B, 512, H/8, W/8)

        # Decoder with skip connections
        d1 = F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e3], dim=1))    # (B, 256, H/4, W/4)

        d2 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))    # (B, 128, H/2, W/2)

        d3 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e1], dim=1))    # (B, 64, H, W)

        return torch.sigmoid(self.output(d3))          # (B, 3, H, W)


# ── Inference ─────────────────────────────────────────────────────────────────

def _preprocess_frames(
    frames: list[np.ndarray],
) -> torch.Tensor:
    """Stack 3 BGR frames into a (1, 9, H, W) float32 tensor at 640×360."""
    processed = []
    for f in frames:
        f_resized = cv2.resize(f, (MODEL_INPUT_W, MODEL_INPUT_H))
        f_rgb = cv2.cvtColor(f_resized, cv2.COLOR_BGR2RGB)
        processed.append(f_rgb.astype(np.float32) / 255.0)

    # Stack: (3, H, W, 3) → (9, H, W) via channel-first reshape
    stacked = np.concatenate([p.transpose(2, 0, 1) for p in processed], axis=0)  # (9, H, W)
    return torch.from_numpy(stacked).unsqueeze(0)   # (1, 9, H, W)


def _extract_peak(
    heatmap: np.ndarray,
    orig_w: int,
    orig_h: int,
    threshold: float = 0.5,
) -> tuple[float, float] | None:
    """
    Find the ball center from a single-channel heatmap (H×W, values 0-1).
    Returns (x, y) in original frame coordinates, or None if no detection.
    """
    hmap_h, hmap_w = heatmap.shape
    max_val = float(heatmap.max())
    if max_val < threshold:
        return None

    # Primary: argmax → unravel to (row, col) then convert to (x, y)
    iy, ix = np.unravel_index(heatmap.argmax(), heatmap.shape)
    x = float(ix) * orig_w / hmap_w
    y = float(iy) * orig_h / hmap_h

    # Refine with weighted centroid around the peak for sub-pixel accuracy
    radius = 5
    y0 = max(0, int(iy) - radius)
    y1 = min(hmap_h, int(iy) + radius + 1)
    x0 = max(0, int(ix) - radius)
    x1 = min(hmap_w, int(ix) + radius + 1)
    patch = heatmap[y0:y1, x0:x1]
    patch_sum = patch.sum()
    if patch_sum > 0:
        rows, cols = np.mgrid[y0:y1, x0:x1]
        cx = float((cols * patch).sum() / patch_sum)
        cy = float((rows * patch).sum() / patch_sum)
        x = cx * orig_w / hmap_w
        y = cy * orig_h / hmap_h

    return x, y


class BallDetectorTrackNet:
    """
    Tennis ball detector using TrackNetV3 (model_best.pt).

    Processes 3-frame windows and outputs per-frame ball positions.
    Drop-in companion to BallDetectorYOLO — returns the same
    (ball_track, raw_detections) tuple format.
    """

    def __init__(self, checkpoint_path: str, device: str | None = None):
        cp = Path(checkpoint_path)
        if not cp.exists():
            raise FileNotFoundError(f"TrackNetV3 checkpoint not found: {cp}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrackNetV3()

        ckpt = torch.load(str(cp), map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            epoch = ckpt.get("epoch", "?")
            val_loss = ckpt.get("val_loss", "?")
            logger.info(f"TrackNetV3 checkpoint: epoch={epoch}, val_loss={val_loss:.6f}" if isinstance(val_loss, float) else f"epoch={epoch}")
        else:
            state_dict = ckpt

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"BallDetectorTrackNet loaded from {cp} (device={self.device})")

    def detect_video(
        self,
        video_path: str,
        start_sec: float | None = None,
        end_sec: float | None = None,
        conf_threshold: float = 0.5,
        progress_callback=None,
    ) -> tuple[list[tuple[float | None, float | None]], list]:
        """
        Detect tennis ball across video frames using 3-frame windows.

        Returns:
            (ball_track, raw_detections)
            ball_track: list of (x, y) center coords or (None, None) per frame
            raw_detections: list of (x, y) or None per frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(start_sec * fps) if start_sec is not None else 0
        end_frame = int(end_sec * fps) if end_sec is not None else total_frames
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(start_frame, min(end_frame, total_frames))
        n_frames = end_frame - start_frame

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        logger.info(f"[TrackNetV3] Ball detection: frames {start_frame}-{end_frame} ({n_frames} frames)")

        # Read all frames into a buffer first (needed for 3-frame windows)
        frame_buffer: list[np.ndarray] = []
        count = 0
        while count < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_buffer.append(frame)
            count += 1
        cap.release()

        n = len(frame_buffer)
        ball_track: list[tuple[float | None, float | None]] = [(None, None)] * n
        raw_detections: list = [None] * n

        if n < 3:
            logger.warning(f"[TrackNetV3] Too few frames ({n}) for 3-frame inference")
            return ball_track, raw_detections

        with torch.no_grad():
            for i in range(n):
                # Use clamped window: [i-1, i, i+1] clamped to [0, n-1]
                i0 = max(0, i - 1)
                i1 = i
                i2 = min(n - 1, i + 1)

                inp = _preprocess_frames([frame_buffer[i0], frame_buffer[i1], frame_buffer[i2]])
                inp = inp.to(self.device)

                out = self.model(inp)           # (1, 3, H, W)
                # Channel 1 = middle frame heatmap (0-indexed)
                heatmap = out[0, 1].cpu().numpy()  # (H, W)

                xy = _extract_peak(heatmap, orig_w, orig_h, threshold=conf_threshold)
                ball_track[i] = xy if xy is not None else (None, None)
                raw_detections[i] = xy

                if progress_callback and (i + 1) % 50 == 0:
                    progress_callback(i + 1, n)

        detected = sum(1 for x, y in ball_track if x is not None)
        logger.info(
            f"[TrackNetV3] Ball detection complete: {n} frames, "
            f"{detected} with ball ({detected / max(n, 1) * 100:.0f}%)"
        )
        return ball_track, raw_detections
