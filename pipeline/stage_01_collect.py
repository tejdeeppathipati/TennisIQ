"""
Stage 01: Frame extraction from YouTube URL or MP4 upload.

- Attempts yt-dlp download ONCE. On any failure, raises DownloadError immediately (FR-03).
- Extracts frames at configurable fps (default: 30).
- Detects and skips between-point dead frames using motion delta thresholding (FR-06).
- Auto-splits frames: first 75% = training domain, last 25% = held-out generalization test.
- No silent retries on YouTube failure.
"""
import subprocess
import sys
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FRAMES_DIR = Path("/data/frames")
VIDEO_DIR = Path("/data/video")


class DownloadError(Exception):
    pass


def download_youtube(url: str, output_path: Path) -> Path:
    """Download YouTube video via yt-dlp. Raises DownloadError on any failure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise DownloadError(
            f"YouTube download failed. Please upload an MP4 file instead.\n"
            f"Details: {result.stderr[:500]}"
        )
    if not output_path.exists():
        raise DownloadError("Download completed but output file not found. Please upload an MP4 file instead.")
    return output_path


def compute_motion_delta(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Compute normalized mean absolute difference between consecutive grayscale frames."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff)) / 255.0


def extract_frames(
    video_path: Path,
    fps: float,
    output_dir: Path,
    motion_threshold: float = 0.02,
) -> tuple[list[Path], int]:
    """
    Extract frames from video at given fps, skipping dead frames.

    Between-point dead frames (towel breaks, players standing at baseline,
    ball boys) are detected via motion delta thresholding and skipped to
    maximize information density.

    Returns:
        (kept_frame_paths, total_dead_skipped)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(video_fps / fps)))

    frame_paths = []
    frame_count = 0
    saved_count = 0
    dead_skipped = 0
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.resize(curr_gray, (160, 90))

            if prev_gray is not None:
                delta = compute_motion_delta(prev_gray, curr_gray)
                if delta < motion_threshold:
                    dead_skipped += 1
                    prev_gray = curr_gray
                    frame_count += 1
                    continue

            prev_gray = curr_gray
            frame_path = output_dir / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_paths.append(frame_path)
            saved_count += 1

        frame_count += 1

    cap.release()
    logger.info(
        f"Extracted {saved_count} frames from {video_path} at ~{fps} fps "
        f"({dead_skipped} dead frames skipped)"
    )
    return frame_paths, dead_skipped


def split_frames(frame_paths: list[Path], train_ratio: float = 0.75) -> tuple[list[Path], list[Path]]:
    """Split frames temporally: first N% = train, remaining = held-out test."""
    split_idx = int(len(frame_paths) * train_ratio)
    train = frame_paths[:split_idx]
    test = frame_paths[split_idx:]
    logger.info(f"Frame split: {len(train)} train / {len(test)} generalization test")
    return train, test


def run(job_id: str, footage_url: str, config: dict) -> dict:
    """
    Download footage and extract frames with dead frame skip.

    Returns:
        dict with 'train_frames', 'test_frames', 'video_path', 'total_frames', 'dead_skipped'
    Raises:
        DownloadError: if YouTube download fails (signal to frontend to prompt MP4 upload)
    """
    fps = config.get("fps", 30)
    train_ratio = config.get("train_split", 0.75)
    motion_threshold = config.get("dead_frame_motion_threshold", 0.02)

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    frames_dir = FRAMES_DIR / job_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    if footage_url.startswith("file://"):
        video_path = Path(footage_url.replace("file://", ""))
    else:
        video_path = VIDEO_DIR / f"{job_id}.mp4"
        logger.info(f"Downloading footage from {footage_url}")
        download_youtube(footage_url, video_path)

    logger.info(f"Extracting frames at fps={fps} (motion threshold={motion_threshold})")
    all_frames, dead_skipped = extract_frames(
        video_path, fps=fps, output_dir=frames_dir, motion_threshold=motion_threshold
    )

    if not all_frames:
        raise ValueError("No frames extracted from video. The file may be corrupted or empty.")

    train_frames, test_frames = split_frames(all_frames, train_ratio=train_ratio)

    return {
        "video_path": str(video_path),
        "all_frames": [str(p) for p in all_frames],
        "train_frames": [str(p) for p in train_frames],
        "test_frames": [str(p) for p in test_frames],
        "total_frames": len(all_frames),
        "dead_skipped": dead_skipped,
        "fps_used": fps,
    }
