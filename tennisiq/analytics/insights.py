from __future__ import annotations

from typing import Dict


def summarize_insights(stats: Dict[str, float]) -> str:
    return (
        f"Frames: {int(stats['frames'])}, "
        f"Bounces: {int(stats['bounces'])}, "
        f"Hits: {int(stats['hits'])}, "
        f"Events/100 frames: {stats['events_per_100_frames']:.2f}"
    )
