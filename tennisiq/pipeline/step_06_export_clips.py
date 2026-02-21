from __future__ import annotations

from pathlib import Path
from typing import List

from tqdm import tqdm

from tennisiq.io.video import write_video


def run_step_06_export_clips(
    output_dir: str,
    frames: List,
    points: List[dict],
    fps: int,
    pre_sec: float = 1.0,
    post_sec: float = 0.2,
):
    clips_dir = Path(output_dir) / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    pre_frames = int(pre_sec * fps)
    post_frames = int(post_sec * fps)

    clip_meta = []
    for p in tqdm(points, desc="Step 6/6 Export clips", unit="clip"):
        start = max(0, p["start_frame"] - pre_frames)
        end = min(len(frames) - 1, p["end_frame"] + post_frames)
        clip_frames = frames[start : end + 1]
        clip_name = f"point_{p['point_id']:04d}.mp4"
        clip_path = clips_dir / clip_name
        if clip_frames:
            write_video(clip_frames, fps, str(clip_path), codec="mp4v")
        clip_meta.append({"point_id": p["point_id"], "clip": str(clip_path), "start_frame": start, "end_frame": end})

    return str(clips_dir), clip_meta
