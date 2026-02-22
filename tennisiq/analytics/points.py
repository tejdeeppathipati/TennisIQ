"""
Point segmentation — FR-25 through FR-29.

FR-25: State machine segments footage into discrete tennis points.
FR-26: End reason classification (OUT, DOUBLE_BOUNCE, NET, BALL_LOST).
FR-27: Per-point data record (start/end frame, serve, bounces, rally count).
FR-28: Serve zone classification (first bounce → service box).
FR-29: Serve fault type (wide, long, net).

The segmenter consumes the event stream from detect_events() and groups
events into discrete point objects with full metadata.
"""
import logging
from dataclasses import dataclass

from tennisiq.analytics.events import TennisEvent

logger = logging.getLogger(__name__)

# ─── Tunables ────────────────────────────────────────────────────────────────

INACTIVITY_GAP_FRAMES = 50   # ~2s at 25fps — no events for this long ends a point
MIN_EVENTS_FOR_POINT = 2     # a valid point needs at least this many events
HOMOGRAPHY_CONF_THRESHOLD = 0.7
HOMOGRAPHY_DROP_FRAMES = 5   # NFR-R03: sustained drop length to trigger flagging

# ─── Court geometry: service boxes ───────────────────────────────────────────
# (from CourtReference)
_LEFT_INNER_X = 423
_RIGHT_INNER_X = 1242
_CENTER_X = 832
_NET_Y = 1748
_FAR_SERVICE_Y = 1110      # top_inner_line
_NEAR_SERVICE_Y = 2386     # bottom_inner_line
_BASELINE_TOP_Y = 561
_BASELINE_BOTTOM_Y = 2935

# Service boxes: (x_min, y_min, x_max, y_max)
SERVICE_BOXES = {
    "far_left":   (_LEFT_INNER_X, _FAR_SERVICE_Y, _CENTER_X,      _NET_Y),
    "far_right":  (_CENTER_X,     _FAR_SERVICE_Y, _RIGHT_INNER_X,  _NET_Y),
    "near_left":  (_LEFT_INNER_X, _NET_Y,         _CENTER_X,       _NEAR_SERVICE_Y),
    "near_right": (_CENTER_X,     _NET_Y,         _RIGHT_INNER_X,  _NEAR_SERVICE_Y),
}


@dataclass
class TennisPoint:
    """A discrete tennis point segmented from the event stream."""
    point_idx: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    serve_frame: int | None               # frame of the first hit (the serve)
    serve_player: str | None              # player who served
    first_bounce_frame: int | None
    first_bounce_court_xy: tuple[float, float] | None
    serve_zone: str | None                # FR-28: service box name
    serve_fault_type: str | None          # FR-29: "wide", "long", "net", or None
    end_reason: str                       # FR-26: OUT, DOUBLE_BOUNCE, NET, BALL_LOST
    rally_hit_count: int                  # total hits in the point
    bounce_count: int
    bounce_frames: list[int]
    events: list[TennisEvent]
    confidence: float                     # avg event score
    low_confidence_homography: bool = False  # NFR-R03: flagged due to sustained homography drop


def _find_low_homography_ranges(
    homographies: list,
    threshold: float = HOMOGRAPHY_CONF_THRESHOLD,
    min_run: int = HOMOGRAPHY_DROP_FRAMES,
) -> list[tuple[int, int]]:
    """NFR-R03: Find frame ranges where homography confidence is below threshold
    for at least `min_run` consecutive frames."""
    ranges: list[tuple[int, int]] = []
    run_start: int | None = None

    for i, h in enumerate(homographies):
        below = h.confidence < threshold
        if below:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= min_run:
                ranges.append((run_start, i - 1))
            run_start = None

    if run_start is not None and (len(homographies) - run_start) >= min_run:
        ranges.append((run_start, len(homographies) - 1))

    return ranges


# ─── FR-28: Serve Zone Classification ───────────────────────────────────────

def classify_serve_zone(court_x: float, court_y: float) -> str | None:
    """Map a bounce position to the service box it landed in, or None if outside all boxes."""
    for name, (x_min, y_min, x_max, y_max) in SERVICE_BOXES.items():
        if x_min <= court_x <= x_max and y_min <= court_y <= y_max:
            return name
    return None


# ─── FR-29: Serve Fault Type ────────────────────────────────────────────────

def classify_fault_type(
    court_x: float,
    court_y: float,
    target_side: str,
) -> str:
    """Classify a serve fault as wide, long, or net.

    target_side: "far" or "near" — the side the serve should land on.
    """
    if target_side == "far":
        if court_y < _FAR_SERVICE_Y:
            return "long"
        if court_y > _NET_Y:
            return "net"
        if court_x < _LEFT_INNER_X or court_x > _RIGHT_INNER_X:
            return "wide"
        return "long"
    else:
        if court_y > _NEAR_SERVICE_Y:
            return "long"
        if court_y < _NET_Y:
            return "net"
        if court_x < _LEFT_INNER_X or court_x > _RIGHT_INNER_X:
            return "wide"
        return "long"


# ─── FR-26: End Reason Classification ───────────────────────────────────────

def _determine_end_reason(events: list[TennisEvent], total_frames: int) -> str:
    """Determine why a point ended based on its event sequence."""
    if not events:
        return "BALL_LOST"

    bounces = [e for e in events if e.event_type == "bounce"]

    for b in reversed(bounces):
        if b.in_out == "out":
            return "OUT"

    for i in range(1, len(bounces)):
        if bounces[i].court_side == bounces[i - 1].court_side:
            hits_between = [
                e for e in events
                if e.event_type == "hit"
                and bounces[i - 1].frame_idx < e.frame_idx < bounces[i].frame_idx
            ]
            if not hits_between:
                return "DOUBLE_BOUNCE"

    last_event = events[-1]
    frames_after_last = total_frames - last_event.frame_idx
    if frames_after_last > INACTIVITY_GAP_FRAMES:
        return "BALL_LOST"

    return "BALL_LOST"


# ─── FR-25: Point Segmentation State Machine ────────────────────────────────

def _group_events_into_rallies(
    events: list[TennisEvent],
    gap_threshold: int = INACTIVITY_GAP_FRAMES,
) -> list[list[TennisEvent]]:
    """Split event stream into rally groups separated by inactivity gaps."""
    if not events:
        return []

    groups: list[list[TennisEvent]] = []
    current: list[TennisEvent] = [events[0]]

    for i in range(1, len(events)):
        frame_gap = events[i].frame_idx - events[i - 1].frame_idx
        if frame_gap > gap_threshold:
            groups.append(current)
            current = [events[i]]
        else:
            current.append(events[i])

    if current:
        groups.append(current)

    return groups


def segment_points(
    events: list[TennisEvent],
    fps: float,
    total_frames: int,
    start_sec: float = 0.0,
    min_events: int = MIN_EVENTS_FOR_POINT,
    homographies: list | None = None,
) -> list[TennisPoint]:
    """
    FR-25: Segment events into discrete tennis points.

    Args:
        events: sorted list of TennisEvent from detect_events()
        fps: video frame rate
        total_frames: total frames in the segment
        start_sec: segment start time for absolute timestamps
        min_events: minimum events to qualify as a point
        homographies: per-frame homography results for NFR-R03 flagging

    Returns:
        List of TennisPoint, one per detected point.
    """
    low_hom_ranges = _find_low_homography_ranges(homographies) if homographies else []

    rallies = _group_events_into_rallies(events)

    points: list[TennisPoint] = []

    for group in rallies:
        if len(group) < min_events:
            logger.debug(
                f"Skipping rally group with {len(group)} events "
                f"(frames {group[0].frame_idx}-{group[-1].frame_idx})"
            )
            continue

        all_hits = [e for e in group if e.event_type == "hit"]

        if not all_hits:
            continue

        serve_frame = all_hits[0].frame_idx
        serve_player = all_hits[0].player
        serve_side = all_hits[0].court_side

        point_events = [e for e in group if e.frame_idx >= serve_frame]

        if len(point_events) < min_events:
            continue

        hits = [e for e in point_events if e.event_type == "hit"]
        bounces = [e for e in point_events if e.event_type == "bounce"]

        first_bounce = bounces[0] if bounces else None
        first_bounce_frame = first_bounce.frame_idx if first_bounce else None
        first_bounce_xy = first_bounce.court_xy if first_bounce else None

        serve_zone = None
        serve_fault_type = None
        if first_bounce_xy is not None:
            serve_zone = classify_serve_zone(first_bounce_xy[0], first_bounce_xy[1])
            if serve_zone is None:
                target_side = "far" if serve_side == "near" else "near"
                serve_fault_type = classify_fault_type(
                    first_bounce_xy[0], first_bounce_xy[1], target_side,
                )

        end_reason = _determine_end_reason(point_events, total_frames)

        scores = [e.score for e in point_events]
        confidence = sum(scores) / len(scores) if scores else 0.0

        # NFR-R03: check if this point's frame range overlaps any sustained homography drop
        pt_start = point_events[0].frame_idx
        pt_end = point_events[-1].frame_idx
        hom_flagged = False
        hom_penalty = 1.0
        for drop_start, drop_end in low_hom_ranges:
            if pt_start <= drop_end and pt_end >= drop_start:
                hom_flagged = True
                overlap = min(pt_end, drop_end) - max(pt_start, drop_start) + 1
                pt_span = pt_end - pt_start + 1
                hom_penalty = max(0.3, 1.0 - (overlap / pt_span) * 0.5)
                break

        if hom_flagged:
            confidence = confidence * hom_penalty

        point = TennisPoint(
            point_idx=len(points),
            start_frame=pt_start,
            end_frame=pt_end,
            start_sec=round(start_sec + pt_start / fps, 3),
            end_sec=round(start_sec + pt_end / fps, 3),
            serve_frame=serve_frame,
            serve_player=serve_player,
            first_bounce_frame=first_bounce_frame,
            first_bounce_court_xy=first_bounce_xy,
            serve_zone=serve_zone,
            serve_fault_type=serve_fault_type,
            end_reason=end_reason,
            rally_hit_count=len(hits),
            bounce_count=len(bounces),
            bounce_frames=[b.frame_idx for b in bounces],
            events=point_events,
            confidence=round(confidence, 3),
            low_confidence_homography=hom_flagged,
        )
        points.append(point)

    n_points = len(points)
    end_reasons = {}
    for p in points:
        end_reasons[p.end_reason] = end_reasons.get(p.end_reason, 0) + 1

    logger.info(
        f"Point segmentation: {n_points} points from {len(events)} events. "
        f"End reasons: {end_reasons}"
    )

    return points
