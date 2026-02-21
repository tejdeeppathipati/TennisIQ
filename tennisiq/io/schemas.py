from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    homography_confidence: float = 0.0

    ball_visible: bool = False
    ball_speed: float = 0.0
    ball_accel: float = 0.0

    event_candidates: Dict[str, bool] = field(default_factory=lambda: {"bounce": False, "hit": False})
    event_scores: Dict[str, float] = field(default_factory=lambda: {"bounce": 0.0, "hit": 0.0})
    event_reasons: List[str] = field(default_factory=list)

    ball_inout: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
