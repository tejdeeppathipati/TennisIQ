from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


Point = Tuple[Optional[float], Optional[float]]


@dataclass
class FrameRecord:
    frame_idx: int
    court_points: List[Point]
    ball_point: Point

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
