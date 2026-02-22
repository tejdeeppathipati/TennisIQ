from .events import detect_events, TennisEvent
from .points import segment_points, TennisPoint
from .shots import detect_shots, ShotEvent
from .match_analytics import compute_match_analytics, MatchAnalytics
from .coaching_intelligence import generate_coaching_intelligence, CoachingIntelligence

__all__ = [
    "detect_events", "TennisEvent",
    "segment_points", "TennisPoint",
    "detect_shots", "ShotEvent",
    "compute_match_analytics", "MatchAnalytics",
    "generate_coaching_intelligence", "CoachingIntelligence",
]
