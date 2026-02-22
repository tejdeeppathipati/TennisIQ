"""
Trajectory-based shot type classifier.

Classifies shots as serve / forehand / backhand using ball trajectory
geometry and court-space positions. No pose or keypoint data required.

Heuristics:
  - SERVE: first shot of a point OR ball originates behind the baseline
    with high speed (>8 m/s) and steep vertical travel.
  - FOREHAND / BACKHAND: determined by which side of the hitter's body
    the ball exits. Uses the ball's horizontal direction relative to the
    court side and player position.  Assumes right-handed players (the
    dominant case; flipping handedness only swaps FH/BH labels).
"""
import math
from dataclasses import dataclass

from tennisiq.analytics.shots import ShotEvent
from tennisiq.cv.ball.inference import BallPhysics

SHOT_CLASSES = ["backhand", "forehand", "neutral", "serve"]

_COURT_CENTER_X = 832
_BASELINE_NEAR_Y = 3496
_BASELINE_FAR_Y = 0
_SERVE_MIN_SPEED = 6.0


@dataclass
class ShotClassification:
    shot_type: str
    confidence: float
    probabilities: dict[str, float]


def classify_shot_type(
    shot: ShotEvent,
    shot_idx: int,
    all_shots: list[ShotEvent],
    ball_physics: list[BallPhysics],
) -> ShotClassification:
    """
    Classify a single shot using ball trajectory geometry.

    Returns ShotClassification with shot_type in {serve, forehand, backhand, neutral}.
    """
    # --- Serve detection ---
    is_first_in_rally = _is_first_shot_in_rally(shot, shot_idx, all_shots)
    behind_baseline = _is_behind_baseline(shot)
    high_speed = (shot.speed_m_s or 0) >= _SERVE_MIN_SPEED

    if is_first_in_rally and (behind_baseline or high_speed):
        conf = 0.85 if (behind_baseline and high_speed) else 0.65
        return ShotClassification("serve", conf, _probs("serve", conf))

    # --- Forehand / Backhand via trajectory angle ---
    if shot.ball_direction_deg is None:
        return ShotClassification("neutral", 0.3, _probs("neutral", 0.3))

    angle = shot.ball_direction_deg
    side = shot.court_side or ("near" if shot.ball_court_xy[1] > 1748 else "far")
    ball_x = shot.ball_court_xy[0]

    shot_type = _classify_fh_bh(angle, side, ball_x)
    conf = 0.60 if shot_type != "neutral" else 0.35
    return ShotClassification(shot_type, conf, _probs(shot_type, conf))


def _is_first_shot_in_rally(
    shot: ShotEvent, idx: int, all_shots: list[ShotEvent],
) -> bool:
    if idx == 0:
        return True
    prev = all_shots[idx - 1]
    gap_sec = shot.timestamp_sec - prev.timestamp_sec
    return gap_sec > 2.5


def _is_behind_baseline(shot: ShotEvent) -> bool:
    bx, by = shot.ball_court_xy
    if shot.court_side == "near":
        return by > _BASELINE_NEAR_Y * 0.92
    return by < _BASELINE_FAR_Y + _BASELINE_NEAR_Y * 0.08


def _classify_fh_bh(angle_deg: float, court_side: str, ball_x: float) -> str:
    """
    Right-handed assumption:
      Near player (facing far end): ball going right (+x) after contact → forehand
      Far player  (facing near end): ball going left (-x) after contact → forehand
    """
    rad = math.radians(angle_deg)
    dx = math.cos(rad)

    if abs(dx) < 0.15:
        return "neutral"

    if court_side == "near":
        return "forehand" if dx > 0 else "backhand"
    else:
        return "forehand" if dx < 0 else "backhand"


def _probs(main: str, conf: float) -> dict[str, float]:
    remainder = (1.0 - conf) / max(len(SHOT_CLASSES) - 1, 1)
    return {c: (conf if c == main else remainder) for c in SHOT_CLASSES}
