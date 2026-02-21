from __future__ import annotations

from pathlib import Path


def run_step_06_export_clips(output_dir: str):
    clips_dir = Path(output_dir) / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    return str(clips_dir)
