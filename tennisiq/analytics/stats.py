from __future__ import annotations

from typing import Dict, List


def basic_match_stats(num_bounces: int, num_hits: int, num_frames: int) -> Dict[str, float]:
    return {
        "frames": float(num_frames),
        "bounces": float(num_bounces),
        "hits": float(num_hits),
        "events_per_100_frames": (num_bounces + num_hits) * 100.0 / max(num_frames, 1),
    }
