"""
Coaching intelligence engine — generates data-backed insights from match analytics.

Produces:
    - Player cards: plain-English tendency summaries per player
    - Weakness reports: top 2-3 exploitable patterns with data
    - Match flow: rally length drift, momentum shifts, fatigue signals
    - Enhanced coaching cards: per-point shot-by-shot breakdowns with context
"""
import logging
from dataclasses import dataclass, field

from tennisiq.analytics.match_analytics import MatchAnalytics, PlayerStats, _determine_point_winner
from tennisiq.analytics.shots import ShotEvent
from tennisiq.analytics.points import TennisPoint

logger = logging.getLogger(__name__)


@dataclass
class PlayerCard:
    """Plain-English tendency summary for a player."""
    label: str
    tendencies: list[str]
    serve_summary: str
    shot_distribution_summary: str
    coverage_summary: str
    exploit_plan: str = ""   # single top-priority actionable sentence for the coach


@dataclass
class WeaknessItem:
    """A single exploitable weakness with supporting data."""
    description: str
    data_point: str
    points_cost: int
    severity: float  # 0-1


@dataclass
class WeaknessReport:
    """Top weaknesses for a player."""
    label: str
    weaknesses: list[WeaknessItem]


@dataclass
class MatchFlowInsight:
    """An insight about how the match evolved over time."""
    description: str
    timestamp_range: tuple[float, float] | None = None


@dataclass
class EnhancedCoachingCard:
    """Data-backed per-point coaching card."""
    point_idx: int
    start_sec: float
    end_sec: float
    summary: str
    shot_sequence: list[dict]
    pattern_context: str
    suggestion: str
    end_reason: str
    rally_hit_count: int
    confidence: float


@dataclass
class CoachingIntelligence:
    """Complete coaching intelligence output."""
    player_a_card: PlayerCard = field(default_factory=lambda: PlayerCard("player_a", [], "", "", ""))
    player_b_card: PlayerCard = field(default_factory=lambda: PlayerCard("player_b", [], "", "", ""))
    player_a_weaknesses: WeaknessReport = field(default_factory=lambda: WeaknessReport("player_a", []))
    player_b_weaknesses: WeaknessReport = field(default_factory=lambda: WeaknessReport("player_b", []))
    match_flow_insights: list[MatchFlowInsight] = field(default_factory=list)
    coaching_cards: list[EnhancedCoachingCard] = field(default_factory=list)


def generate_coaching_intelligence(
    analytics: MatchAnalytics,
    points: list[TennisPoint],
    shots: list[ShotEvent],
    shot_directions: dict[int, str],
) -> CoachingIntelligence:
    """Generate complete coaching intelligence from analytics data."""
    ci = CoachingIntelligence()

    ci.player_a_card = _generate_player_card(analytics.player_a, analytics)
    ci.player_b_card = _generate_player_card(analytics.player_b, analytics)

    ci.player_a_weaknesses = _generate_weakness_report(analytics.player_a, analytics, points, shots)
    ci.player_b_weaknesses = _generate_weakness_report(analytics.player_b, analytics, points, shots)

    ci.match_flow_insights = _generate_match_flow_insights(analytics, points, shots)
    ci.coaching_cards = _generate_enhanced_cards(points, shots, shot_directions, analytics)

    logger.info(
        f"Coaching intelligence: {len(ci.coaching_cards)} cards, "
        f"{len(ci.player_a_weaknesses.weaknesses)} A weaknesses, "
        f"{len(ci.player_b_weaknesses.weaknesses)} B weaknesses, "
        f"{len(ci.match_flow_insights)} flow insights"
    )
    return ci


def _generate_player_card(player: PlayerStats, analytics: MatchAnalytics) -> PlayerCard:
    """Generate plain-English tendency summary."""
    label = player.label
    name = "Player A" if label == "player_a" else "Player B"
    tendencies = []

    # Dominant shot type
    if player.shot_type_pcts:
        sorted_types = sorted(player.shot_type_pcts.items(), key=lambda x: -x[1])
        dominant = sorted_types[0]
        if dominant[0] != "neutral" and dominant[1] > 30:
            tendencies.append(
                f"{name} primarily uses {dominant[0]}s ({dominant[1]}% of all shots)."
            )

    # Dominant direction per shot type
    for stype, dirs in player.shot_direction_pcts.items():
        if stype == "neutral" or stype == "serve":
            continue
        sorted_dirs = sorted(dirs.items(), key=lambda x: -x[1])
        if sorted_dirs and sorted_dirs[0][1] > 50 and sorted_dirs[0][0] != "unknown":
            tendencies.append(
                f"{name}'s {stype} goes {sorted_dirs[0][0].replace('_', ' ')} "
                f"{sorted_dirs[0][1]}% of the time."
            )

    # Shot pattern dominance
    patterns = analytics.shot_pattern_dominance.get(label, [])
    if patterns and patterns[0]["pct"] > 25:
        p = patterns[0]
        tendencies.append(
            f"{name}'s most frequent pattern is {p['shot_type']} "
            f"{p['direction'].replace('_', ' ')} ({p['pct']}% of shots)."
        )

    # Serve summary
    serve_summary = ""
    if player.first_serve_pct > 0:
        serve_summary = f"{name} has a {player.first_serve_pct}% first serve rate."
        if player.serve_zone_win_rate:
            best_zone = max(player.serve_zone_win_rate.items(), key=lambda x: x[1])
            if best_zone[1] > 0:
                serve_summary += (
                    f" Most effective serve zone: {best_zone[0].replace('_', ' ')} "
                    f"({best_zone[1]}% win rate)."
                )

    # Shot distribution summary
    shot_dist = ""
    if player.shot_type_counts:
        parts = [f"{k}: {v}" for k, v in sorted(player.shot_type_counts.items(), key=lambda x: -x[1])]
        shot_dist = f"Shot breakdown: {', '.join(parts)} (total: {player.total_shots})"

    # Coverage summary
    coverage = ""
    if player.total_distance_covered > 0:
        coverage = (
            f"{name} covered {player.total_distance_covered:.0f} court units. "
            f"Average position: ({player.center_of_gravity[0]:.0f}, {player.center_of_gravity[1]:.0f})."
        )

    if not tendencies:
        tendencies.append(f"Insufficient shot data to determine {name}'s tendencies.")

    # Build exploit plan — single top-priority actionable sentence
    exploit_plan = _build_exploit_plan(player, analytics, name)

    return PlayerCard(
        label=label,
        tendencies=tendencies,
        serve_summary=serve_summary,
        shot_distribution_summary=shot_dist,
        coverage_summary=coverage,
        exploit_plan=exploit_plan,
    )


def _build_exploit_plan(player: PlayerStats, analytics: MatchAnalytics, name: str) -> str:
    """Single most actionable insight for defeating this player."""
    opponent = "Player B" if player.label == "player_a" else "Player A"

    # Fatigue weakness is the strongest signal
    short_rate = player.error_rate_by_rally_length.get("1-3", 0)
    mid_rate = player.error_rate_by_rally_length.get("4-6", 0)
    long_rate = max(
        player.error_rate_by_rally_length.get("7-9", 0),
        player.error_rate_by_rally_length.get("10+", 0),
    )
    long_errors = (
        player.error_by_rally_length.get("7-9", 0) +
        player.error_by_rally_length.get("10+", 0)
    )
    baseline = max(short_rate, mid_rate)
    if long_rate > baseline + 15 and long_errors >= 2:
        return (
            f"Extend rallies past 7 shots — {name}'s error rate jumps from "
            f"{baseline:.0f}% to {long_rate:.0f}% in long exchanges "
            f"({long_errors} points surrendered this match)."
        )

    # Shot type weakness
    non_neutral = {k: v for k, v in player.error_rate_by_shot_type.items() if k not in ("neutral", "unknown")}
    if non_neutral:
        worst_type, worst_rate = max(non_neutral.items(), key=lambda x: x[1])
        count = player.error_by_shot_type.get(worst_type, 0)
        if worst_rate > 25 and count >= 2:
            return (
                f"Attack {name}'s {worst_type} — {worst_rate:.0f}% error rate "
                f"({count} errors this match). Force them to that wing in key moments."
            )

    # Predictable direction
    for stype, dirs in player.shot_direction_pcts.items():
        if stype in ("neutral", "serve", "unknown"):
            continue
        sorted_dirs = sorted(dirs.items(), key=lambda x: -x[1])
        if sorted_dirs and sorted_dirs[0][1] > 70 and sorted_dirs[0][0] not in ("unknown", ""):
            dir_str = sorted_dirs[0][0].replace("_", " ")
            return (
                f"Position early for {name}'s {stype} — {sorted_dirs[0][1]:.0f}% go {dir_str}. "
                f"Reading the pattern gives {opponent} an easy setup."
            )

    return ""


def _generate_weakness_report(
    player: PlayerStats,
    analytics: MatchAnalytics,
    points: list[TennisPoint],
    shots: list[ShotEvent],
) -> WeaknessReport:
    """Identify the top 2-3 exploitable weaknesses with actionable, plain-English descriptions."""
    name = "Player A" if player.label == "player_a" else "Player B"
    opponent = "Player B" if player.label == "player_a" else "Player A"
    weaknesses: list[WeaknessItem] = []

    # Weakness 1: Error rate spikes in long rallies (fatigue / pressure signal)
    short_rate = player.error_rate_by_rally_length.get("1-3", 0)
    mid_rate = player.error_rate_by_rally_length.get("4-6", 0)
    long_rate = player.error_rate_by_rally_length.get("7-9", 0)
    longer_rate = player.error_rate_by_rally_length.get("10+", 0)
    fatigue_rate = max(long_rate, longer_rate)
    threshold = max(short_rate, mid_rate)

    if fatigue_rate > threshold + 15 and fatigue_rate > 25:
        long_errors = (
            player.error_by_rally_length.get("7-9", 0) +
            player.error_by_rally_length.get("10+", 0)
        )
        rate_label = max(("7-9", long_rate), ("10+", longer_rate), key=lambda x: x[1])
        weaknesses.append(WeaknessItem(
            description=(
                f"Against {name}: extend rallies past 7 shots. "
                f"Their error rate triples — from {threshold:.0f}% in short rallies "
                f"to {fatigue_rate:.0f}% in rallies longer than 7 shots. "
                f"This pattern cost {long_errors} point{'s' if long_errors != 1 else ''} in this match."
            ),
            data_point=(
                f"{long_errors} unforced errors in rallies 7+ shots "
                f"vs {threshold:.0f}% error rate in rallies under 6 shots"
            ),
            points_cost=long_errors,
            severity=min((fatigue_rate - threshold) / 50.0, 1.0),
        ))

    # Weakness 2: Shot type with highest error rate (minimum threshold for relevance)
    if player.error_rate_by_shot_type:
        non_neutral = {k: v for k, v in player.error_rate_by_shot_type.items() if k not in ("neutral", "unknown")}
        if non_neutral:
            worst_type, worst_rate = max(non_neutral.items(), key=lambda x: x[1])
            count = player.error_by_shot_type.get(worst_type, 0)
            total = player.shot_type_counts.get(worst_type, 0)
            if worst_rate > 20 and count >= 2:
                weaknesses.append(WeaknessItem(
                    description=(
                        f"{name}'s {worst_type} breaks down under pressure — "
                        f"{worst_rate:.0f}% error rate ({count} errors from {total} attempts). "
                        f"Target their {worst_type} side in critical points."
                    ),
                    data_point=f"{count} errors / {total} {worst_type}s ({worst_rate:.0f}% error rate)",
                    points_cost=count,
                    severity=min(worst_rate / 100.0, 1.0),
                ))

    # Weakness 3: Highly predictable shot direction (>70% same direction)
    for stype, dirs in player.shot_direction_pcts.items():
        if stype in ("neutral", "serve", "unknown"):
            continue
        sorted_dirs = sorted(dirs.items(), key=lambda x: -x[1])
        if sorted_dirs and sorted_dirs[0][1] > 70 and sorted_dirs[0][0] not in ("unknown", ""):
            dominant_dir = sorted_dirs[0][0].replace("_", " ")
            dominant_pct = sorted_dirs[0][1]
            weaknesses.append(WeaknessItem(
                description=(
                    f"{name} telegraphs their {stype} — {dominant_pct:.0f}% go {dominant_dir}. "
                    f"{opponent} can position early and attack the predictable pattern."
                ),
                data_point=f"{dominant_pct:.0f}% of {stype}s go {dominant_dir}",
                points_cost=0,
                severity=min(dominant_pct / 100.0, 0.8),
            ))
        break  # only one direction weakness to avoid redundancy

    # Sort by severity and keep top 3
    weaknesses.sort(key=lambda w: -w.severity)
    return WeaknessReport(label=player.label, weaknesses=weaknesses[:3])


def _generate_match_flow_insights(
    analytics: MatchAnalytics,
    points: list[TennisPoint],
    shots: list[ShotEvent],
) -> list[MatchFlowInsight]:
    """Generate insights about how the match evolved."""
    insights = []

    if not points:
        return insights

    # Momentum shifts (3+ consecutive points by same player)
    streak = 0
    streak_player = None
    for i, pt in enumerate(points):
        winner = _determine_point_winner(pt, shots)
        if winner == streak_player:
            streak += 1
        else:
            if streak >= 3 and streak_player:
                name = "Player A" if streak_player == "player_a" else "Player B"
                insights.append(MatchFlowInsight(
                    description=f"{name} won {streak} consecutive points — a momentum surge.",
                    timestamp_range=(points[max(0, i - streak)].start_sec, points[i - 1].end_sec),
                ))
            streak = 1
            streak_player = winner

    if streak >= 3 and streak_player:
        name = "Player A" if streak_player == "player_a" else "Player B"
        insights.append(MatchFlowInsight(
            description=f"{name} won the last {streak} consecutive points.",
            timestamp_range=(points[-streak].start_sec, points[-1].end_sec),
        ))

    # Rally length trend (early vs late)
    if len(points) >= 6:
        half = len(points) // 2
        early_avg = sum(pt.rally_hit_count for pt in points[:half]) / max(half, 1)
        late_avg = sum(pt.rally_hit_count for pt in points[half:]) / max(len(points) - half, 1)

        if abs(late_avg - early_avg) > 1.5:
            direction = "increased" if late_avg > early_avg else "decreased"
            insights.append(MatchFlowInsight(
                description=(
                    f"Rally length {direction} from {early_avg:.1f} shots (first half) "
                    f"to {late_avg:.1f} shots (second half) — "
                    f"{'suggesting fatigue or more conservative play' if direction == 'decreased' else 'suggesting longer exchanges and more baseline play'}."
                ),
            ))

    # Overall summary
    avg_rally = analytics.rally_length_avg
    if avg_rally > 0:
        if avg_rally < 3:
            insights.append(MatchFlowInsight(
                description=f"Short rallies dominate (avg {avg_rally:.1f} shots) — serve-heavy or aggressive play."
            ))
        elif avg_rally > 7:
            insights.append(MatchFlowInsight(
                description=f"Long rallies dominate (avg {avg_rally:.1f} shots) — baseline-heavy match."
            ))

    return insights


def _generate_enhanced_cards(
    points: list[TennisPoint],
    shots: list[ShotEvent],
    shot_directions: dict[int, str],
    analytics: MatchAnalytics,
) -> list[EnhancedCoachingCard]:
    """Generate data-backed coaching cards with shot-by-shot breakdowns."""
    cards = []
    error_tracker: dict[str, int] = {}

    for pt in points:
        point_shots = [
            s for s in shots
            if pt.start_frame <= s.frame_idx <= pt.end_frame
        ]

        shot_sequence = []
        for s in point_shots:
            direction = shot_directions.get(s.frame_idx, "unknown")
            owner_name = "A" if s.owner == "player_a" else "B"
            shot_sequence.append({
                "owner": s.owner,
                "owner_short": owner_name,
                "shot_type": s.shot_type or "unknown",
                "direction": direction,
                "speed_m_s": s.speed_m_s,
                "court_side": s.court_side,
            })

        summary = _build_point_summary(pt, point_shots, shot_directions)

        # Track pattern context (consecutive errors of same type)
        pattern_context = ""
        if pt.end_reason in ("OUT", "NET") and point_shots:
            last = point_shots[-1]
            key = f"{last.owner}_{last.shot_type}_{shot_directions.get(last.frame_idx, 'unk')}"
            error_tracker[key] = error_tracker.get(key, 0) + 1
            if error_tracker[key] > 1:
                name = "Player A" if last.owner == "player_a" else "Player B"
                pattern_context = (
                    f"This is the {_ordinal(error_tracker[key])} "
                    f"{last.shot_type or ''} {shot_directions.get(last.frame_idx, '')} "
                    f"error by {name}."
                )

        suggestion = _build_suggestion(pt, point_shots, shot_directions, analytics)

        cards.append(EnhancedCoachingCard(
            point_idx=pt.point_idx,
            start_sec=pt.start_sec,
            end_sec=pt.end_sec,
            summary=summary,
            shot_sequence=shot_sequence,
            pattern_context=pattern_context,
            suggestion=suggestion,
            end_reason=pt.end_reason,
            rally_hit_count=pt.rally_hit_count,
            confidence=pt.confidence,
        ))

    return cards


def _build_point_summary(
    pt: TennisPoint,
    point_shots: list[ShotEvent],
    shot_directions: dict[int, str],
) -> str:
    """Build a detailed summary of what happened in a point."""
    duration = pt.end_sec - pt.start_sec
    parts = []

    if point_shots:
        shot_desc = []
        for s in point_shots[:6]:
            owner = "A" if s.owner == "player_a" else "B"
            stype = s.shot_type or "shot"
            direction = shot_directions.get(s.frame_idx, "")
            speed = f" ({s.speed_m_s:.0f} m/s)" if s.speed_m_s else ""
            shot_desc.append(f"{owner} {stype} {direction}{speed}")

        if len(point_shots) > 6:
            shot_desc.append(f"... +{len(point_shots) - 6} more")

        parts.append(f"{len(point_shots)}-shot rally ({duration:.1f}s): {' → '.join(shot_desc)}")
    else:
        parts.append(f"Point lasted {duration:.1f}s with {pt.rally_hit_count} detected hits.")

    # Exclusive terminal classification — avoid contradictory labels
    is_serve_fault = pt.serve_fault_type and pt.rally_hit_count <= 1
    if is_serve_fault:
        fault = pt.serve_fault_type
        if fault == "net":
            parts.append("Serve into the net")
        elif fault == "long":
            parts.append("Serve long")
        elif fault == "wide":
            parts.append("Serve wide")
        else:
            parts.append(f"Serve fault ({fault})")
    elif pt.end_reason == "OUT":
        parts.append("Ball out")
    elif pt.end_reason == "NET":
        parts.append("Ball into the net")
    elif pt.end_reason == "DOUBLE_BOUNCE":
        parts.append("Double bounce")
    elif pt.end_reason == "BALL_LOST":
        parts.append("Ball went out of tracking range")
    else:
        parts.append(f"End: {pt.end_reason}")

    if pt.serve_zone and not is_serve_fault:
        parts.append(f"Serve landed {pt.serve_zone.replace('_', ' ')}")

    return ". ".join(parts) + "."


def _build_suggestion(
    pt: TennisPoint,
    point_shots: list[ShotEvent],
    shot_directions: dict[int, str],
    analytics: MatchAnalytics,
) -> str:
    """Generate a plain-English, data-backed coaching suggestion."""
    if not point_shots:
        return ""

    last_shot = point_shots[-1]
    last_owner = last_shot.owner
    name = "Player A" if last_owner == "player_a" else "Player B"
    opponent = "Player B" if last_owner == "player_a" else "Player A"
    player = analytics.player_a if last_owner == "player_a" else analytics.player_b

    # Long rally ended badly — fatigue/pressure weakness
    if pt.rally_hit_count >= 7:
        short_rate = player.error_rate_by_rally_length.get("1-3", 0)
        mid_rate = player.error_rate_by_rally_length.get("4-6", 0)
        long_rate = max(
            player.error_rate_by_rally_length.get("7-9", 0),
            player.error_rate_by_rally_length.get("10+", 0),
        )
        baseline = max(short_rate, mid_rate)
        if long_rate > baseline + 15:
            return (
                f"{opponent}'s strategy is working — {name} collapses in long rallies "
                f"({long_rate:.0f}% error rate vs {baseline:.0f}% in short rallies). "
                f"Keep pushing the pace past shot 7."
            )

    if pt.end_reason == "OUT":
        stype = last_shot.shot_type or "shot"
        direction = shot_directions.get(last_shot.frame_idx, "")
        err_rate = player.error_rate_by_shot_type.get(stype, 0)
        if err_rate > 20 and stype not in ("neutral", "unknown"):
            return (
                f"{name}'s {stype} is breaking down — {err_rate:.0f}% error rate this match. "
                f"This is an exploitable pattern. Keep forcing them to that wing."
            )
        if direction and direction not in ("unknown", ""):
            dir_str = direction.replace("_", " ")
            return (
                f"{name} went for the {dir_str} and missed long. "
                f"Under pressure, this is their go-to and it's costing them."
            )
        return f"{name} pushed for a winner and paid for it — unforced error out of bounds."

    if pt.end_reason == "NET":
        stype = last_shot.shot_type or "shot"
        return (
            f"{name}'s {stype} clipped the net — likely rushed or off-balance. "
            f"Force them wide to create more situations like this."
        )

    # Repetitive direction pattern
    if len(point_shots) >= 3:
        owner_shots = [s for s in point_shots if s.owner == last_owner]
        if len(owner_shots) >= 3:
            dirs = [shot_directions.get(s.frame_idx, "") for s in owner_shots[-3:]]
            if len(set(dirs)) == 1 and dirs[0] and dirs[0] not in ("unknown", ""):
                dir_str = dirs[0].replace("_", " ")
                return (
                    f"{name} went {dir_str} {len(owner_shots)} times in a row — "
                    f"a predictable pattern {opponent} can start anticipating."
                )

    # Solid point
    if pt.rally_hit_count >= 5:
        return (
            f"{pt.rally_hit_count}-shot rally — good baseline consistency from both players."
        )

    return ""


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def coaching_to_dict(ci: CoachingIntelligence) -> dict:
    """Serialize CoachingIntelligence to JSON-compatible dict."""
    def player_card_dict(pc: PlayerCard) -> dict:
        return {
            "label": pc.label,
            "exploit_plan": pc.exploit_plan,
            "tendencies": pc.tendencies,
            "serve_summary": pc.serve_summary,
            "shot_distribution_summary": pc.shot_distribution_summary,
            "coverage_summary": pc.coverage_summary,
        }

    def weakness_dict(wr: WeaknessReport) -> dict:
        return {
            "label": wr.label,
            "weaknesses": [
                {
                    "description": w.description,
                    "data_point": w.data_point,
                    "points_cost": w.points_cost,
                    "severity": w.severity,
                }
                for w in wr.weaknesses
            ],
        }

    def card_dict(c: EnhancedCoachingCard) -> dict:
        return {
            "point_idx": c.point_idx,
            "start_sec": c.start_sec,
            "end_sec": c.end_sec,
            "summary": c.summary,
            "shot_sequence": c.shot_sequence,
            "pattern_context": c.pattern_context,
            "suggestion": c.suggestion,
            "end_reason": c.end_reason,
            "rally_hit_count": c.rally_hit_count,
            "confidence": c.confidence,
        }

    def flow_dict(f: MatchFlowInsight) -> dict:
        return {
            "description": f.description,
            "timestamp_range": list(f.timestamp_range) if f.timestamp_range else None,
        }

    return {
        "player_a_card": player_card_dict(ci.player_a_card),
        "player_b_card": player_card_dict(ci.player_b_card),
        "player_a_weaknesses": weakness_dict(ci.player_a_weaknesses),
        "player_b_weaknesses": weakness_dict(ci.player_b_weaknesses),
        "match_flow_insights": [flow_dict(f) for f in ci.match_flow_insights],
        "coaching_cards": [card_dict(c) for c in ci.coaching_cards],
    }
