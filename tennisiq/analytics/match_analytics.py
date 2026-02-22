"""
Match analytics engine — computes statistical analysis from shot and point data.

Produces per-player stats, shot patterns, rally analysis, serve metrics,
fatigue signals, and match flow data that feed the coaching intelligence layer.
"""
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

from tennisiq.analytics.shots import ShotEvent
from tennisiq.analytics.events import TennisEvent
from tennisiq.analytics.points import TennisPoint
from tennisiq.cv.ball.inference import BallPhysics
from tennisiq.cv.players.inference import FramePlayers

logger = logging.getLogger(__name__)


@dataclass
class PlayerStats:
    """Per-player statistical summary."""
    label: str  # "player_a" or "player_b"
    total_shots: int = 0
    shot_type_counts: dict[str, int] = field(default_factory=dict)
    shot_type_pcts: dict[str, float] = field(default_factory=dict)
    shot_direction_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    shot_direction_pcts: dict[str, dict[str, float]] = field(default_factory=dict)
    error_by_shot_type: dict[str, int] = field(default_factory=dict)
    error_rate_by_shot_type: dict[str, float] = field(default_factory=dict)
    error_by_rally_length: dict[str, int] = field(default_factory=dict)
    error_rate_by_rally_length: dict[str, float] = field(default_factory=dict)
    avg_shot_speed: float = 0.0
    total_distance_covered: float = 0.0
    center_of_gravity: tuple[float, float] = (0.0, 0.0)
    first_serve_pct: float = 0.0
    second_serve_pct: float = 0.0
    ace_count: int = 0
    double_fault_count: int = 0
    serve_zone_win_rate: dict[str, float] = field(default_factory=dict)
    serve_placement_counts: dict[str, int] = field(default_factory=dict)
    points_won: int = 0
    points_lost: int = 0


@dataclass
class MatchAnalytics:
    """Complete match analytics payload."""
    player_a: PlayerStats = field(default_factory=lambda: PlayerStats(label="player_a"))
    player_b: PlayerStats = field(default_factory=lambda: PlayerStats(label="player_b"))
    rally_length_distribution: dict[str, int] = field(default_factory=dict)
    rally_length_avg: float = 0.0
    total_points: int = 0
    total_shots: int = 0
    momentum_data: list[dict] = field(default_factory=list)
    match_flow: list[dict] = field(default_factory=list)
    shot_pattern_dominance: dict[str, list[dict]] = field(default_factory=dict)


RALLY_BUCKETS = {"1-3": (1, 3), "4-6": (4, 6), "7-9": (7, 9), "10+": (10, 999)}


def compute_match_analytics(
    shot_events: list[ShotEvent],
    shot_directions: dict[int, str],
    points: list[TennisPoint],
    events: list[TennisEvent],
    ball_physics: list[BallPhysics],
    player_results: list[FramePlayers],
    fps: float,
) -> MatchAnalytics:
    """Compute full match analytics from pipeline outputs."""
    analytics = MatchAnalytics()
    analytics.total_points = len(points)
    analytics.total_shots = len(shot_events)

    _compute_shot_stats(analytics, shot_events, shot_directions)
    _compute_error_analysis(analytics, shot_events, shot_directions, points, events)
    _compute_serve_stats(analytics, shot_events, points)
    _compute_rally_distribution(analytics, points)
    _compute_court_coverage(analytics, player_results)
    _compute_momentum(analytics, points, shot_events)
    _compute_shot_pattern_dominance(analytics, shot_events, shot_directions)

    logger.info(
        f"Match analytics: {analytics.total_shots} shots, {analytics.total_points} points, "
        f"A={analytics.player_a.total_shots} B={analytics.player_b.total_shots}"
    )
    return analytics


def _compute_shot_stats(
    analytics: MatchAnalytics,
    shots: list[ShotEvent],
    directions: dict[int, str],
) -> None:
    """Compute per-player shot type and direction distributions."""
    for player_stats in [analytics.player_a, analytics.player_b]:
        label = player_stats.label
        player_shots = [s for s in shots if s.owner == label]
        player_stats.total_shots = len(player_shots)

        type_counts: dict[str, int] = defaultdict(int)
        dir_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total_speed = 0.0
        speed_count = 0

        for s in player_shots:
            stype = s.shot_type or "unknown"
            type_counts[stype] += 1

            direction = directions.get(s.frame_idx, "unknown")
            dir_counts[stype][direction] += 1

            if s.speed_m_s is not None and s.speed_m_s > 0:
                total_speed += s.speed_m_s
                speed_count += 1

        player_stats.shot_type_counts = dict(type_counts)
        total = max(player_stats.total_shots, 1)
        player_stats.shot_type_pcts = {k: round(v / total * 100, 1) for k, v in type_counts.items()}

        player_stats.shot_direction_counts = {k: dict(v) for k, v in dir_counts.items()}
        player_stats.shot_direction_pcts = {}
        for stype, dirs in dir_counts.items():
            type_total = max(sum(dirs.values()), 1)
            player_stats.shot_direction_pcts[stype] = {
                d: round(c / type_total * 100, 1) for d, c in dirs.items()
            }

        player_stats.avg_shot_speed = round(total_speed / max(speed_count, 1), 1)


def _compute_error_analysis(
    analytics: MatchAnalytics,
    shots: list[ShotEvent],
    directions: dict[int, str],
    points: list[TennisPoint],
    events: list[TennisEvent],
) -> None:
    """Compute error rates by shot type and rally length."""
    point_end_shots = _map_last_shots_to_points(shots, points)

    for player_stats in [analytics.player_a, analytics.player_b]:
        label = player_stats.label
        error_by_type: dict[str, int] = defaultdict(int)
        shots_by_type: dict[str, int] = defaultdict(int)
        error_by_rally: dict[str, int] = defaultdict(int)
        shots_by_rally: dict[str, int] = defaultdict(int)

        for s in shots:
            if s.owner != label:
                continue
            stype = s.shot_type or "unknown"
            shots_by_type[stype] += 1

        for pt in points:
            rally_bucket = _rally_bucket(pt.rally_hit_count)
            shots_by_rally[rally_bucket] += 1

            if pt.end_reason in ("OUT", "NET") and pt.point_idx in point_end_shots:
                last_shot = point_end_shots[pt.point_idx]
                if last_shot.owner == label:
                    stype = last_shot.shot_type or "unknown"
                    error_by_type[stype] += 1
                    error_by_rally[rally_bucket] += 1

        player_stats.error_by_shot_type = dict(error_by_type)
        player_stats.error_rate_by_shot_type = {
            k: round(v / max(shots_by_type.get(k, 1), 1) * 100, 1)
            for k, v in error_by_type.items()
        }

        player_stats.error_by_rally_length = dict(error_by_rally)
        player_stats.error_rate_by_rally_length = {
            k: round(v / max(shots_by_rally.get(k, 1), 1) * 100, 1)
            for k, v in error_by_rally.items()
        }


def _compute_serve_stats(
    analytics: MatchAnalytics,
    shots: list[ShotEvent],
    points: list[TennisPoint],
) -> None:
    """Compute serve-related statistics per player."""
    for player_stats in [analytics.player_a, analytics.player_b]:
        label = player_stats.label
        serve_points = [pt for pt in points if pt.serve_player == label]

        total_serves = len(serve_points)
        if total_serves == 0:
            continue

        first_serves = [pt for pt in serve_points if pt.serve_fault_type is None]
        faults = [pt for pt in serve_points if pt.serve_fault_type is not None]

        player_stats.first_serve_pct = round(len(first_serves) / max(total_serves, 1) * 100, 1)

        double_faults = 0
        for i, pt in enumerate(serve_points):
            if pt.serve_fault_type and i + 1 < len(serve_points):
                next_pt = serve_points[i + 1]
                if next_pt.serve_fault_type:
                    double_faults += 1

        player_stats.double_fault_count = double_faults

        zone_wins: dict[str, int] = defaultdict(int)
        zone_total: dict[str, int] = defaultdict(int)
        placement_counts: dict[str, int] = defaultdict(int)

        for pt in serve_points:
            zone = pt.serve_zone or "unknown"
            placement_counts[zone] += 1
            zone_total[zone] += 1
            if pt.end_reason != "BALL_LOST" and _did_server_win(pt, label):
                zone_wins[zone] += 1

        player_stats.serve_placement_counts = dict(placement_counts)
        player_stats.serve_zone_win_rate = {
            z: round(zone_wins.get(z, 0) / max(zone_total[z], 1) * 100, 1)
            for z in zone_total
        }


def _compute_rally_distribution(analytics: MatchAnalytics, points: list[TennisPoint]) -> None:
    """Compute rally length histogram."""
    dist: dict[str, int] = defaultdict(int)
    total_hits = 0
    for pt in points:
        bucket = _rally_bucket(pt.rally_hit_count)
        dist[bucket] += 1
        total_hits += pt.rally_hit_count

    analytics.rally_length_distribution = dict(dist)
    analytics.rally_length_avg = round(total_hits / max(len(points), 1), 1)


def _compute_court_coverage(analytics: MatchAnalytics, player_results: list[FramePlayers]) -> None:
    """Compute distance traveled and center of gravity per player."""
    for player_stats, attr in [(analytics.player_a, "player_a"), (analytics.player_b, "player_b")]:
        positions: list[tuple[float, float]] = []
        for pr in player_results:
            pl = getattr(pr, attr)
            if pl and pl.foot_court[0] is not None:
                positions.append((pl.foot_court[0], pl.foot_court[1]))

        if len(positions) < 2:
            continue

        total_dist = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            total_dist += math.hypot(dx, dy)

        player_stats.total_distance_covered = round(total_dist, 1)

        avg_x = sum(p[0] for p in positions) / len(positions)
        avg_y = sum(p[1] for p in positions) / len(positions)
        player_stats.center_of_gravity = (round(avg_x, 1), round(avg_y, 1))


def _compute_momentum(
    analytics: MatchAnalytics,
    points: list[TennisPoint],
    shots: list[ShotEvent],
) -> None:
    """Compute momentum tracking — rolling window of point outcomes."""
    if not points:
        return

    window = 3
    momentum_data = []
    match_flow = []

    for i, pt in enumerate(points):
        winner = _determine_point_winner(pt, shots)

        # Rolling momentum (last `window` points)
        start_idx = max(0, i - window + 1)
        recent = points[start_idx:i + 1]
        a_wins = sum(1 for rpt in recent if _determine_point_winner(rpt, shots) == "player_a")
        b_wins = len(recent) - a_wins

        momentum_data.append({
            "point_idx": pt.point_idx,
            "timestamp_sec": pt.start_sec,
            "winner": winner,
            "a_momentum": a_wins,
            "b_momentum": b_wins,
            "rally_length": pt.rally_hit_count,
        })

        match_flow.append({
            "point_idx": pt.point_idx,
            "timestamp_sec": pt.start_sec,
            "rally_length": pt.rally_hit_count,
            "end_reason": pt.end_reason,
            "duration_sec": round(pt.end_sec - pt.start_sec, 1),
        })

    analytics.momentum_data = momentum_data
    analytics.match_flow = match_flow


def _compute_shot_pattern_dominance(
    analytics: MatchAnalytics,
    shots: list[ShotEvent],
    directions: dict[int, str],
) -> None:
    """Identify dominant shot patterns per player."""
    for label in ["player_a", "player_b"]:
        player_shots = [s for s in shots if s.owner == label]
        if not player_shots:
            analytics.shot_pattern_dominance[label] = []
            continue

        pattern_counts: dict[str, int] = defaultdict(int)
        total = len(player_shots)

        for s in player_shots:
            stype = s.shot_type or "unknown"
            direction = directions.get(s.frame_idx, "unknown")
            pattern = f"{stype}_{direction}"
            pattern_counts[pattern] += 1

        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])
        dominance = []
        for pattern, count in sorted_patterns[:5]:
            parts = pattern.split("_", 1)
            dominance.append({
                "pattern": pattern,
                "shot_type": parts[0],
                "direction": parts[1] if len(parts) > 1 else "unknown",
                "count": count,
                "pct": round(count / max(total, 1) * 100, 1),
            })

        analytics.shot_pattern_dominance[label] = dominance


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _rally_bucket(hit_count: int) -> str:
    for label, (lo, hi) in RALLY_BUCKETS.items():
        if lo <= hit_count <= hi:
            return label
    return "10+"


def _map_last_shots_to_points(
    shots: list[ShotEvent],
    points: list[TennisPoint],
) -> dict[int, ShotEvent]:
    """Map each point to the last shot that occurred within its frame range."""
    result = {}
    for pt in points:
        last = None
        for s in shots:
            if pt.start_frame <= s.frame_idx <= pt.end_frame:
                last = s
        if last:
            result[pt.point_idx] = last
    return result


def _did_server_win(pt: TennisPoint, server: str) -> bool:
    """Heuristic: server wins if point doesn't end with their error."""
    if pt.end_reason == "OUT" or pt.end_reason == "NET":
        return False
    return True


def _determine_point_winner(pt: TennisPoint, shots: list[ShotEvent]) -> str:
    """
    Determine which player won a point.

    Uses the last shot's owner + end reason to infer:
    - If OUT/NET → last hitter lost
    - Otherwise → last hitter won (rally continuation assumption)
    """
    last_shot = None
    for s in shots:
        if pt.start_frame <= s.frame_idx <= pt.end_frame:
            last_shot = s

    if last_shot is None:
        return "player_a"

    if pt.end_reason in ("OUT", "NET"):
        return "player_b" if last_shot.owner == "player_a" else "player_a"

    return last_shot.owner


def analytics_to_dict(analytics: MatchAnalytics) -> dict:
    """Serialize MatchAnalytics to a JSON-compatible dictionary."""
    def player_to_dict(ps: PlayerStats) -> dict:
        return {
            "label": ps.label,
            "total_shots": ps.total_shots,
            "shot_type_counts": ps.shot_type_counts,
            "shot_type_pcts": ps.shot_type_pcts,
            "shot_direction_counts": ps.shot_direction_counts,
            "shot_direction_pcts": ps.shot_direction_pcts,
            "error_by_shot_type": ps.error_by_shot_type,
            "error_rate_by_shot_type": ps.error_rate_by_shot_type,
            "error_by_rally_length": ps.error_by_rally_length,
            "error_rate_by_rally_length": ps.error_rate_by_rally_length,
            "avg_shot_speed_m_s": ps.avg_shot_speed,
            "total_distance_covered": ps.total_distance_covered,
            "center_of_gravity": list(ps.center_of_gravity),
            "first_serve_pct": ps.first_serve_pct,
            "double_fault_count": ps.double_fault_count,
            "serve_zone_win_rate": ps.serve_zone_win_rate,
            "serve_placement_counts": ps.serve_placement_counts,
            "points_won": ps.points_won,
            "points_lost": ps.points_lost,
        }

    return {
        "player_a": player_to_dict(analytics.player_a),
        "player_b": player_to_dict(analytics.player_b),
        "rally_length_distribution": analytics.rally_length_distribution,
        "rally_length_avg": analytics.rally_length_avg,
        "total_points": analytics.total_points,
        "total_shots": analytics.total_shots,
        "momentum_data": analytics.momentum_data,
        "match_flow": analytics.match_flow,
        "shot_pattern_dominance": analytics.shot_pattern_dominance,
    }
