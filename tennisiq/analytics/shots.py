"""
Shot (contact) detection and ball ownership assignment.

A "shot" = a racquet contact event, detected from ball trajectory physics:
  1. Smooth court-space ball positions with exponential moving average (EMA)
  2. Compute velocity vectors from smoothed data
  3. A contact candidate fires when direction changes sharply (angle > θ)
  4. Filter by speed floor/ceiling, consecutive-detection stability, and cooldown
  5. Ownership via player proximity + court-side heuristic

This replaces the old per-frame dot-product approach which over-counted by ~3x.
"""
import logging
import math
from dataclasses import dataclass

from tennisiq.cv.ball.inference import BallPhysics
from tennisiq.cv.players.inference import FramePlayers

logger = logging.getLogger(__name__)

_NET_Y = 1748  # court-space y coordinate of the net

# ─── Tuned parameters (calibrated against broadcast Nadal footage) ───────────
EMA_ALPHA = 0.3           # smoothing factor (lower = smoother)
MIN_ANGLE_DEG = 80        # minimum direction change to count as contact
MIN_SPEED = 5.0           # court-units/frame; below this is noise
MAX_SPEED = 300.0         # above this is a tracking artifact
COOLDOWN_FRAMES = 12      # ~0.4s at 30fps — realistic min gap between hits
MIN_STABLE_FRAMES = 3     # ball must be detected for N consecutive frames


@dataclass
class ShotEvent:
    """A detected contact event with ownership and optional classification."""
    frame_idx: int
    timestamp_sec: float
    owner: str                                    # "player_a" or "player_b"
    ball_court_xy: tuple[float, float]
    shot_type: str | None = None                  # forehand/backhand/serve/neutral
    shot_type_confidence: float = 0.0
    ball_direction_deg: float | None = None       # direction angle after the shot
    speed_m_s: float | None = None
    court_side: str | None = None                 # "near" or "far"
    ownership_method: str = "combined"            # "distance", "court_side", or "combined"


def detect_contacts(
    ball_physics: list[BallPhysics],
    player_results: list[FramePlayers],
    fps: float,
    start_sec: float = 0.0,
) -> list[ShotEvent]:
    """
    Detect racquet-contact events from ball trajectory physics.

    Uses EMA-smoothed coordinates, angle reversal, speed gating,
    stability checks, and cooldown debouncing to produce a
    human-plausible shot count.
    """
    n = len(ball_physics)
    if n < 4:
        return []

    # Raw court-space positions
    raw_x = [bp.court_xy[0] if bp.court_xy[0] is not None else None for bp in ball_physics]
    raw_y = [bp.court_xy[1] if bp.court_xy[0] is not None else None for bp in ball_physics]

    # Step 1: EMA smoothing
    sx, sy = _ema_smooth(raw_x, raw_y, EMA_ALPHA)

    # Step 2: Velocity vectors + stability counter
    vx, vy, stable = _compute_velocities(sx, sy, raw_x)

    # Step 3: Contact detection with gating
    shots: list[ShotEvent] = []
    last_contact = -999

    for i in range(2, n - 1):
        if vx[i] is None or vx[i - 1] is None:
            continue
        if stable[i] < MIN_STABLE_FRAMES:
            continue

        sb = math.hypot(vx[i - 1], vy[i - 1])
        sa = math.hypot(vx[i], vy[i])
        peak = max(sb, sa)

        if peak < MIN_SPEED or peak > MAX_SPEED:
            continue

        # Angle between consecutive velocity vectors
        mag_product = sb * sa
        if mag_product < 0.01:
            continue
        cos_angle = max(-1.0, min(1.0, (vx[i - 1] * vx[i] + vy[i - 1] * vy[i]) / mag_product))
        angle_deg = math.degrees(math.acos(cos_angle))

        if angle_deg < MIN_ANGLE_DEG:
            continue
        if (i - last_contact) < COOLDOWN_FRAMES:
            continue

        # ── Contact confirmed ──
        ball_xy = (sx[i], sy[i]) if sx[i] is not None else (raw_x[i], raw_y[i])
        if ball_xy[0] is None:
            continue

        owner, method = _assign_ownership(ball_xy, player_results, i)
        if owner is None:
            court_side = "near" if ball_xy[1] > _NET_Y else "far"
            owner = "player_a" if court_side == "near" else "player_b"
            method = "court_side"

        dir_after = None
        if vx[i] is not None and vy[i] is not None:
            if abs(vx[i]) > 0.01 or abs(vy[i]) > 0.01:
                dir_after = math.degrees(math.atan2(vy[i], vx[i]))

        court_side = "near" if ball_xy[1] > _NET_Y else "far"
        timestamp = start_sec + i / fps

        speed_m_s = ball_physics[i].speed_m_per_s

        shots.append(ShotEvent(
            frame_idx=i,
            timestamp_sec=round(timestamp, 3),
            owner=owner,
            ball_court_xy=ball_xy,
            ball_direction_deg=round(dir_after, 1) if dir_after is not None else None,
            speed_m_s=round(speed_m_s, 2) if speed_m_s is not None else None,
            court_side=court_side,
            ownership_method=method,
        ))
        last_contact = i

    logger.info(f"Contact detection: {len(shots)} contacts from {n} frames")
    return shots


# Keep the old name as an alias for backwards compatibility
detect_shots = detect_contacts


# ─── Internal helpers ────────────────────────────────────────────────────────

def _ema_smooth(
    raw_x: list[float | None],
    raw_y: list[float | None],
    alpha: float,
) -> tuple[list[float | None], list[float | None]]:
    """Exponential moving average smoothing, skipping missing values."""
    n = len(raw_x)
    sx = [None] * n
    sy = [None] * n
    lx, ly = None, None
    for i in range(n):
        if raw_x[i] is None:
            sx[i], sy[i] = lx, ly
            continue
        if lx is None:
            sx[i], sy[i] = raw_x[i], raw_y[i]
        else:
            sx[i] = alpha * raw_x[i] + (1 - alpha) * lx
            sy[i] = alpha * raw_y[i] + (1 - alpha) * ly
        lx, ly = sx[i], sy[i]
    return sx, sy


def _compute_velocities(
    sx: list[float | None],
    sy: list[float | None],
    raw_x: list[float | None],
) -> tuple[list[float | None], list[float | None], list[int]]:
    """Compute velocity vectors and consecutive-detection stability count."""
    n = len(sx)
    vx: list[float | None] = [None] * n
    vy: list[float | None] = [None] * n
    stable: list[int] = [0] * n
    sc = 0
    for i in range(n):
        if raw_x[i] is None:
            sc = 0
        else:
            sc += 1
        stable[i] = sc

    for i in range(1, n):
        if sx[i] is not None and sx[i - 1] is not None:
            vx[i] = sx[i] - sx[i - 1]
            vy[i] = sy[i] - sy[i - 1]
    return vx, vy, stable


def _player_centroid(player) -> tuple[float, float] | None:
    if player is None:
        return None
    cx, cy = player.foot_court
    if cx is None or cy is None:
        return None
    return (cx, cy)


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _assign_ownership(
    ball_xy: tuple[float, float],
    player_results: list[FramePlayers],
    frame_idx: int,
    search_radius: int = 2,
) -> tuple[str | None, str]:
    """Assign ownership via distance + court-side heuristic."""
    n_results = len(player_results)
    ball_side = "near" if ball_xy[1] > _NET_Y else "far"
    court_side_owner = "player_a" if ball_side == "near" else "player_b"

    best_label = None
    best_dist = float("inf")

    for offset in range(-search_radius, search_radius + 1):
        fi = frame_idx + offset
        if fi < 0 or fi >= n_results:
            continue
        pr = player_results[fi]
        for label, player in [("player_a", pr.player_a), ("player_b", pr.player_b)]:
            centroid = _player_centroid(player)
            if centroid is None:
                continue
            dist = _distance(ball_xy, centroid)
            if dist < best_dist:
                best_dist = dist
                best_label = label

    if best_label is None:
        return None, "none"

    if best_label == court_side_owner:
        return best_label, "combined"
    if best_dist > 500:
        return court_side_owner, "court_side"
    return best_label, "distance"


def classify_shot_direction(
    shot: ShotEvent,
    court_side: str,
) -> str:
    """Classify shot direction as cross-court, down-the-line, or middle."""
    if shot.ball_direction_deg is None:
        return "unknown"

    angle = shot.ball_direction_deg

    if court_side == "near":
        if -135 < angle < -45:
            return "cross_court" if shot.ball_court_xy[0] < 832 else "down_the_line"
        elif -45 <= angle <= 45:
            return "middle"
        else:
            return "down_the_line" if shot.ball_court_xy[0] < 832 else "cross_court"
    else:
        if 45 < angle < 135:
            return "cross_court" if shot.ball_court_xy[0] > 832 else "down_the_line"
        elif -45 <= angle <= 45:
            return "middle"
        else:
            return "down_the_line" if shot.ball_court_xy[0] > 832 else "cross_court"
