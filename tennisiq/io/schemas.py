from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


Point = Tuple[Optional[float], Optional[float]]
BBox = Tuple[float, float, float, float]


@dataclass
class FrameRecord:
    frame_idx: int
    ts_sec: float
    court_keypoints: List[Point]
    ball_xy: Point
    playerA_bbox: Optional[BBox]
    playerB_bbox: Optional[BBox]
    playerA_id: Optional[int] = None
    playerB_id: Optional[int] = None
    ball_court_xy: Point = (None, None)
    playerA_court_xy: Point = (None, None)
    playerB_court_xy: Point = (None, None)
    homography_ok: bool = False
    ball_inout: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
