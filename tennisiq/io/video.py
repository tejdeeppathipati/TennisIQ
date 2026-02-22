"""Video I/O utilities: reading, fps detection, and fps normalization."""
import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

TARGET_FPS = 30
FPS_TOLERANCE = 1.0  # don't normalize if within ±1 fps of target


def get_video_info(video_path: str) -> dict:
    """Read basic metadata from a video file via OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_sec"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def needs_fps_normalization(fps: float) -> bool:
    """Return True if the video fps deviates enough from TARGET_FPS to warrant normalization."""
    return abs(fps - TARGET_FPS) > FPS_TOLERANCE


def normalize_fps(input_path: str, output_dir: str) -> tuple[str, bool]:
    """
    Normalize video to TARGET_FPS if needed using MoviePy.

    Returns:
        (output_path, was_normalized)
        - If normalization was performed, output_path points to the new file.
        - If the video was already at ~30fps, returns the original path unchanged.
    """
    info = get_video_info(input_path)
    original_fps = info["fps"]

    if not needs_fps_normalization(original_fps):
        logger.info(
            f"Video fps={original_fps:.2f} is within tolerance of {TARGET_FPS}fps — skipping normalization"
        )
        return input_path, False

    logger.info(
        f"Normalizing video from {original_fps:.2f}fps to {TARGET_FPS}fps via MoviePy"
    )

    from moviepy import VideoFileClip

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    output_path = str(out_dir / f"{stem}_30fps.mp4")

    clip = VideoFileClip(input_path)
    normalized = clip.with_fps(TARGET_FPS)
    normalized.write_videofile(
        output_path,
        fps=TARGET_FPS,
        codec="libx264",
        audio=False,
        logger=None,
    )
    clip.close()

    new_info = get_video_info(output_path)
    logger.info(
        f"Normalized: {original_fps:.2f}fps → {new_info['fps']:.2f}fps, "
        f"{new_info['frame_count']} frames, {new_info['duration_sec']:.1f}s"
    )

    return output_path, True
