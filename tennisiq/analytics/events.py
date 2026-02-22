"""
Event detection — FR-21 through FR-24.

FR-21: Detect bounce candidates via ball velocity reversal + speed drop.
FR-22: Score candidates with kinematic features + temporal NMS.
FR-23: Detect hit events via ball direction change ≥50° with sufficient speed + NMS.
FR-24: Classify bounces as in / out / line using court polygon geometry.

The detector operates on court-space ball trajectory and player positions
produced by the upstream pipeline (BallPhysics, FramePlayers, FrameHomography).
"""
import logging
import math
from dataclasses import dataclass

from tennisiq.cv.ball.inference import BallPhysics
from tennisiq.cv.players.inference import FramePlayers

logger = logging.getLogger(__name__)

# ─── Tunables ────────────────────────────────────────────────────────────────

SMOOTH_WINDOW = 3
MIN_SPEED_FOR_EVENT = 3.0          # m/s — ignore direction changes when ball barely moves
DIRECTION_CHANGE_DEG = 50.0        # FR-23: minimum angle change for a hit candidate
HIT_MIN_SPEED = 5.0                # m/s — minimum speed at the event frame for a hit
NMS_WINDOW_FRAMES = 5              # temporal NMS: suppress duplicates within ±N frames
PLAYER_PROXIMITY_UNITS = 350.0     # court-space distance to assign hit to player
LINE_MARGIN_UNITS = 20.0           # FR-24: pixels within line counted as "line"

# Court boundaries (singles sidelines)
_SINGLES_LEFT_X = 423
_SINGLES_RIGHT_X = 1242
_BASELINE_TOP_Y = 561
_BASELINE_BOTTOM_Y = 2935
_NET_Y = 1748


@dataclass
class TennisEvent:
    """A detected bounce or hit event."""
    event_type: str                          # "bounce" or "hit"
    frame_idx: int
    timestamp_sec: float
    court_xy: tuple[float, float]            # court-space position
    speed_before_m_s: float | None           # speed in the frames leading up to the event
    speed_after_m_s: float | None            # speed in the frames following the event
    direction_change_deg: float | None       # angle change at the event
    score: float                             # confidence / quality score 0-1
    player: str | None = None                # "player_a" or "player_b" for hits
    player_distance: float | None = None     # court-space distance to assigned player
    in_out: str | None = None                # FR-24: "in", "out", or "line" for bounces
    court_side: str | None = None            # "near" or "far" (relative to camera)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _smooth(values: list[float | None], window: int = SMOOTH_WINDOW) -> list[float | None]:
    """Simple moving average, skipping None values."""
    n = len(values)
    out: list[float | None] = [None] * n
    half = window // 2
    for i in range(n):
        vals = []
        for j in range(max(0, i - half), min(n, i + half + 1)):
            if values[j] is not None:
                vals.append(values[j])
        out[i] = sum(vals) / len(vals) if vals else None
    return out


def _direction_deg(dx: float, dy: float) -> float:
    """Angle in degrees from court +x axis, range [-180, 180]."""
    return math.degrees(math.atan2(dy, dx))


def _angle_diff(a1: float, a2: float) -> float:
    """Absolute angular difference in [0, 180]."""
    d = abs(a1 - a2) % 360
    return d if d <= 180 else 360 - d


def _court_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _has_tracking_gap(ball_physics: list[BallPhysics], center: int, radius: int = 1) -> bool:
    """True if there's a tracking gap (None court_xy) within ±radius of center.

    Radius of 1 catches direct adjacency issues (the frame itself or its
    immediate neighbours are missing) without rejecting valid events that
    happen to be near—but not at—a gap boundary.
    """
    n = len(ball_physics)
    for j in range(max(0, center - radius), min(n, center + radius + 1)):
        if ball_physics[j].court_xy[0] is None:
            return True
    return False


# ─── FR-24: In/Out Classification ───────────────────────────────────────────

def classify_in_out(
    court_x: float,
    court_y: float,
    line_margin: float = LINE_MARGIN_UNITS,
) -> str:
    """
    Classify a bounce position as 'in', 'out', or 'line' using singles court polygon.

    A bounce is 'in' if it lands strictly inside the singles court boundaries.
    A bounce is 'line' if it's within `line_margin` units of any boundary.
    Otherwise 'out'.
    """
    inside_x = _SINGLES_LEFT_X <= court_x <= _SINGLES_RIGHT_X
    inside_y = _BASELINE_TOP_Y <= court_y <= _BASELINE_BOTTOM_Y

    if inside_x and inside_y:
        dx_left = court_x - _SINGLES_LEFT_X
        dx_right = _SINGLES_RIGHT_X - court_x
        dy_top = court_y - _BASELINE_TOP_Y
        dy_bottom = _BASELINE_BOTTOM_Y - court_y
        min_dist = min(dx_left, dx_right, dy_top, dy_bottom)
        return "line" if min_dist <= line_margin else "in"

    dx = max(_SINGLES_LEFT_X - court_x, 0, court_x - _SINGLES_RIGHT_X)
    dy = max(_BASELINE_TOP_Y - court_y, 0, court_y - _BASELINE_BOTTOM_Y)
    dist_outside = math.hypot(dx, dy)
    return "line" if dist_outside <= line_margin else "out"


# ─── FR-21/22: Bounce Detection ─────────────────────────────────────────────

def _detect_bounces(
    ball_physics: list[BallPhysics],
    fps: float,
) -> list[TennisEvent]:
    """
    Detect bounce candidates: frames where the ball's y-direction reverses
    AND speed drops (or is low) at the reversal point.
    """
    n = len(ball_physics)
    if n < 5:
        return []

    cy = [bp.court_xy[1] if bp.court_xy[0] is not None else None for bp in ball_physics]
    cx = [bp.court_xy[0] if bp.court_xy[0] is not None else None for bp in ball_physics]
    speeds = [bp.speed_m_per_s for bp in ball_physics]

    cy_smooth = _smooth(cy, SMOOTH_WINDOW)

    dy: list[float | None] = [None] * n
    for i in range(1, n):
        if cy_smooth[i] is not None and cy_smooth[i - 1] is not None:
            dy[i] = cy_smooth[i] - cy_smooth[i - 1]

    candidates: list[TennisEvent] = []

    for i in range(2, n - 2):
        if dy[i] is None or dy[i - 1] is None:
            continue

        sign_change = (dy[i - 1] > 0 and dy[i] < 0) or (dy[i - 1] < 0 and dy[i] > 0)
        if not sign_change:
            continue

        if cx[i] is None or cy[i] is None:
            continue

        if _has_tracking_gap(ball_physics, i, radius=3):
            continue

        speed_at = speeds[i]
        speed_before = _avg_speed(speeds, i, window=3, direction=-1)
        speed_after = _avg_speed(speeds, i, window=3, direction=1)

        if speed_before is not None and speed_before < MIN_SPEED_FOR_EVENT:
            continue

        is_speed_drop = (
            speed_before is not None
            and speed_at is not None
            and speed_at < speed_before * 0.8
        )

        is_deceleration = (
            ball_physics[i].accel_m_per_s2 is not None
            and ball_physics[i].accel_m_per_s2 < -10.0
        )

        if not (is_speed_drop or is_deceleration):
            continue

        score = _bounce_score(speed_before, speed_at, speed_after, ball_physics[i].accel_m_per_s2)

        t = ball_physics[i].frame_idx / fps
        side = "near" if cy[i] > _NET_Y else "far"

        evt = TennisEvent(
            event_type="bounce",
            frame_idx=i,
            timestamp_sec=round(t, 3),
            court_xy=(cx[i], cy[i]),
            speed_before_m_s=round(speed_before, 2) if speed_before is not None else None,
            speed_after_m_s=round(speed_after, 2) if speed_after is not None else None,
            direction_change_deg=None,
            score=round(score, 3),
            court_side=side,
        )
        evt.in_out = classify_in_out(cx[i], cy[i])
        candidates.append(evt)

    return candidates


def _bounce_score(
    speed_before: float | None,
    speed_at: float | None,
    speed_after: float | None,
    accel: float | None,
) -> float:
    """Heuristic quality score for a bounce candidate (0-1)."""
    score = 0.3

    if speed_before is not None and speed_at is not None and speed_before > 0:
        drop_ratio = 1.0 - (speed_at / speed_before)
        score += min(drop_ratio, 0.5) * 0.4

    if accel is not None and accel < -20:
        score += min(abs(accel) / 500.0, 0.3)

    return min(score, 1.0)


# ─── FR-23: Hit Detection ───────────────────────────────────────────────────

def _detect_hits(
    ball_physics: list[BallPhysics],
    fps: float,
) -> list[TennisEvent]:
    """
    Detect hit candidates: frames where ball direction changes by ≥50°
    with sufficient speed.
    """
    n = len(ball_physics)
    if n < 5:
        return []

    cx = [bp.court_xy[0] if bp.court_xy[0] is not None else None for bp in ball_physics]
    cy = [bp.court_xy[1] if bp.court_xy[0] is not None else None for bp in ball_physics]
    speeds = [bp.speed_m_per_s for bp in ball_physics]

    directions: list[float | None] = [None] * n
    for i in range(1, n):
        if cx[i] is not None and cx[i - 1] is not None:
            dx = cx[i] - cx[i - 1]
            dy = cy[i] - cy[i - 1]
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                directions[i] = _direction_deg(dx, dy)

    candidates: list[TennisEvent] = []

    for i in range(2, n - 2):
        if _has_tracking_gap(ball_physics, i, radius=3):
            continue

        dir_before = _avg_direction(directions, i, window=3, direction=-1)
        dir_after = _avg_direction(directions, i, window=3, direction=1)

        if dir_before is None or dir_after is None:
            continue

        angle_change = _angle_diff(dir_before, dir_after)
        if angle_change < DIRECTION_CHANGE_DEG:
            continue

        speed_at = speeds[i]
        speed_before = _avg_speed(speeds, i, window=3, direction=-1)
        speed_after = _avg_speed(speeds, i, window=3, direction=1)

        relevant_speed = max(
            s for s in [speed_at, speed_before, speed_after] if s is not None
        ) if any(s is not None for s in [speed_at, speed_before, speed_after]) else 0

        if relevant_speed < HIT_MIN_SPEED:
            continue

        if cx[i] is None or cy[i] is None:
            continue

        score = _hit_score(angle_change, speed_before, speed_after)
        t = ball_physics[i].frame_idx / fps
        side = "near" if cy[i] > _NET_Y else "far"

        evt = TennisEvent(
            event_type="hit",
            frame_idx=i,
            timestamp_sec=round(t, 3),
            court_xy=(cx[i], cy[i]),
            speed_before_m_s=round(speed_before, 2) if speed_before is not None else None,
            speed_after_m_s=round(speed_after, 2) if speed_after is not None else None,
            direction_change_deg=round(angle_change, 1),
            score=round(score, 3),
            court_side=side,
        )
        candidates.append(evt)

    return candidates


def _hit_score(angle_change: float, speed_before: float | None, speed_after: float | None) -> float:
    """Heuristic quality score for a hit candidate (0-1)."""
    score = 0.0

    angle_contrib = min(angle_change / 180.0, 1.0) * 0.5
    score += angle_contrib

    if speed_after is not None and speed_before is not None and speed_before > 0.01:
        boost = speed_after / speed_before
        score += min(boost / 3.0, 0.3)
    elif speed_after is not None:
        score += min(speed_after / 50.0, 0.2)

    return min(score, 1.0)


# ─── Temporal NMS ────────────────────────────────────────────────────────────

def _temporal_nms(events: list[TennisEvent], window: int = NMS_WINDOW_FRAMES) -> list[TennisEvent]:
    """Suppress duplicate detections: keep highest-scoring event within each window."""
    if not events:
        return []

    events_sorted = sorted(events, key=lambda e: -e.score)
    kept: list[TennisEvent] = []
    suppressed: set[int] = set()

    for evt in events_sorted:
        if evt.frame_idx in suppressed:
            continue
        kept.append(evt)
        for f in range(evt.frame_idx - window, evt.frame_idx + window + 1):
            suppressed.add(f)

    return sorted(kept, key=lambda e: e.frame_idx)


# ─── Player Assignment ───────────────────────────────────────────────────────

CROSS_DEDUP_WINDOW = 1  # ±1 frame — same physical moment only


def _cross_event_dedup(events: list[TennisEvent], window: int = CROSS_DEDUP_WINDOW) -> list[TennisEvent]:
    """When a bounce and hit overlap within ±window frames, keep the higher-scoring one.

    Uses a tight window (default ±1 frame) so that legitimate bounce→hit
    sequences a few frames apart are preserved.
    """
    if not events:
        return []

    kept: list[TennisEvent] = []
    i = 0
    while i < len(events):
        cluster = [events[i]]
        j = i + 1
        while j < len(events) and events[j].frame_idx - cluster[-1].frame_idx <= window:
            cluster.append(events[j])
            j += 1

        types_in_cluster = {e.event_type for e in cluster}
        if len(types_in_cluster) > 1 and len(cluster) > 1:
            best = max(cluster, key=lambda e: e.score)
            kept.append(best)
        else:
            kept.extend(cluster)

        i = j

    return kept


PLAYER_SEARCH_RADIUS = 2  # check ±N neighboring frames for player positions


def _assign_players(
    events: list[TennisEvent],
    player_results: list[FramePlayers],
    proximity_threshold: float = PLAYER_PROXIMITY_UNITS,
    search_radius: int = PLAYER_SEARCH_RADIUS,
) -> None:
    """Assign player_a or player_b to hit events based on court-space proximity.

    Searches the event frame and ±search_radius neighboring frames
    for the nearest player, but only assigns if within proximity_threshold.
    """
    n_results = len(player_results)

    for evt in events:
        if evt.event_type != "hit":
            continue

        best_label = None
        best_dist = float("inf")

        for offset in range(-search_radius, search_radius + 1):
            fi = evt.frame_idx + offset
            if fi < 0 or fi >= n_results:
                continue

            pr = player_results[fi]
            for label, player in [("player_a", pr.player_a), ("player_b", pr.player_b)]:
                if player is None or player.foot_court[0] is None:
                    continue
                dist = _court_distance(evt.court_xy, (player.foot_court[0], player.foot_court[1]))
                if dist < best_dist:
                    best_dist = dist
                    best_label = label

        if best_label is not None and best_dist <= proximity_threshold:
            evt.player = best_label
            evt.player_distance = round(best_dist, 1)


# ─── Speed / Direction Averaging Helpers ─────────────────────────────────────

def _avg_speed(speeds: list[float | None], center: int, window: int = 3, direction: int = -1) -> float | None:
    """Average non-None speeds in a range before (direction=-1) or after (direction=1) center."""
    vals = []
    if direction < 0:
        for j in range(max(0, center - window), center):
            if speeds[j] is not None:
                vals.append(speeds[j])
    else:
        for j in range(center + 1, min(len(speeds), center + window + 1)):
            if speeds[j] is not None:
                vals.append(speeds[j])
    return sum(vals) / len(vals) if vals else None


def _avg_direction(
    directions: list[float | None], center: int, window: int = 3, direction: int = -1,
) -> float | None:
    """Average non-None direction angles near center. Uses circular mean."""
    vals = []
    if direction < 0:
        for j in range(max(0, center - window), center):
            if directions[j] is not None:
                vals.append(directions[j])
    else:
        for j in range(center + 1, min(len(directions), center + window + 1)):
            if directions[j] is not None:
                vals.append(directions[j])

    if not vals:
        return None

    sin_sum = sum(math.sin(math.radians(v)) for v in vals)
    cos_sum = sum(math.cos(math.radians(v)) for v in vals)
    return math.degrees(math.atan2(sin_sum, cos_sum))


# ─── Main Entry Point ───────────────────────────────────────────────────────

def detect_events(
    ball_physics: list[BallPhysics],
    player_results: list[FramePlayers],
    fps: float,
    start_sec: float = 0.0,
    line_margin: float = LINE_MARGIN_UNITS,
    nms_window: int = NMS_WINDOW_FRAMES,
) -> list[TennisEvent]:
    """
    Detect and classify bounce and hit events from pipeline data.

    Args:
        ball_physics: per-frame BallPhysics from compute_ball_physics()
        player_results: per-frame FramePlayers from PlayerDetector
        fps: video frame rate
        start_sec: segment start time (for absolute timestamps)
        line_margin: margin in court units for 'line' calls
        nms_window: temporal NMS window in frames

    Returns:
        List of TennisEvent sorted by frame_idx.
    """
    bounces = _detect_bounces(ball_physics, fps)
    hits = _detect_hits(ball_physics, fps)

    logger.info(f"Raw candidates: {len(bounces)} bounces, {len(hits)} hits")

    bounces = _temporal_nms(bounces, window=nms_window)
    hits = _temporal_nms(hits, window=nms_window)

    logger.info(f"After NMS: {len(bounces)} bounces, {len(hits)} hits")

    all_events = sorted(bounces + hits, key=lambda e: e.frame_idx)
    all_events = _cross_event_dedup(all_events)

    logger.info(f"After cross-event dedup: {len(all_events)} events")

    hits_only = [e for e in all_events if e.event_type == "hit"]
    _assign_players(hits_only, player_results)

    for evt in all_events:
        evt.timestamp_sec = round(start_sec + evt.frame_idx / fps, 3)

    n_bounces = sum(1 for e in all_events if e.event_type == "bounce")
    n_hits = sum(1 for e in all_events if e.event_type == "hit")
    n_in = sum(1 for e in all_events if e.in_out == "in")
    n_out = sum(1 for e in all_events if e.in_out == "out")
    n_line = sum(1 for e in all_events if e.in_out == "line")
    n_assigned = sum(1 for e in all_events if e.player is not None)

    logger.info(
        f"Event detection complete: {len(all_events)} events "
        f"({n_bounces} bounces [{n_in} in, {n_out} out, {n_line} line], "
        f"{n_hits} hits [{n_assigned} player-assigned])"
    )

    return all_events
