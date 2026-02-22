"""
Stage 07: Decision tree for tennis — determines next action after each eval cycle.

Decision tree actions:
  1. mAP below floor + low frames -> increase fps, mine hard frames (net approaches, rallies, serves)
  2. FP rate high -> tighten crowd/ball-person exclusion, re-label affected shards
  3. Ball detection lagging -> focus ball-specific augmentation and label refinement
  4. mAP plateau + low frame count -> mine diverse game situations
  5. All criteria met -> exit loop

Hard cap: 3 iterations maximum. Exits with best available model after cap.
"""
import logging

logger = logging.getLogger(__name__)

PLAYER_MAP_FLOOR = 0.80
BALL_MAP_FLOOR = 0.70
COURT_LINES_MAP_FLOOR = 0.75
NET_MAP_FLOOR = 0.75
FP_RATE_CEILING = 0.10
MAP_PLATEAU_DELTA = 0.03
MIN_FRAMES = 500


def decide(
    iteration: int,
    eval_metrics: dict,
    previous_eval: dict | None,
    current_config: dict,
    max_iterations: int = 3,
) -> dict:
    """
    Evaluate metrics and determine next pipeline action.

    Returns:
        dict with:
          'action': one of 'retrain', 'relabel', 'mine_frames', 'exit'
          'justification': plain-English explanation (no ML jargon)
          'config_updates': dict of config changes for next iteration
          'exit_reason': plain-English if action is 'exit'
    """
    player_map = eval_metrics.get("player_map", 0)
    ball_map = eval_metrics.get("ball_map", 0)
    court_lines_map = eval_metrics.get("court_lines_map", 0)
    net_map = eval_metrics.get("net_map", 0)
    fp_rate = eval_metrics.get("fp_rate", 1.0)
    frame_count = eval_metrics.get("frame_count", 0)
    criteria_met = eval_metrics.get("criteria_met", False)
    unmet = eval_metrics.get("unmet_criteria", [])

    prev_player_map = previous_eval.get("player_map", 0) if previous_eval else 0

    if criteria_met:
        return {
            "action": "exit",
            "justification": (
                f"All accuracy targets met after {iteration} training pass{'es' if iteration != 1 else ''}. "
                f"Player: {player_map:.0%}, Ball: {ball_map:.0%}, "
                f"Court lines: {court_lines_map:.0%}, Net: {net_map:.0%}, "
                f"False positive rate: {fp_rate:.0%}. Pipeline complete."
            ),
            "config_updates": {},
            "exit_reason": None,
        }

    if iteration >= max_iterations:
        unmet_plain = "; ".join(unmet) if unmet else "accuracy targets not fully reached"
        return {
            "action": "exit",
            "justification": (
                f"Reached the maximum of {max_iterations} training passes. "
                f"Delivering best available model. Targets not fully met: {unmet_plain}."
            ),
            "config_updates": {},
            "exit_reason": unmet_plain,
        }

    map_plateau = (
        previous_eval is not None and
        abs(player_map - prev_player_map) < MAP_PLATEAU_DELTA
    )

    if ball_map < BALL_MAP_FLOOR and player_map >= PLAYER_MAP_FLOOR * 0.9:
        return {
            "action": "retrain",
            "justification": (
                f"Ball detection lagging at {ball_map:.0%} — below the {BALL_MAP_FLOOR:.0%} target. "
                f"Player detection is strong at {player_map:.0%}. "
                f"Focusing on ball-specific label refinement and augmentation for the next pass."
            ),
            "config_updates": {
                "focus_ball_augmentation": True,
                "ball_label_refinement": True,
            },
        }

    if player_map < PLAYER_MAP_FLOOR and frame_count < MIN_FRAMES:
        new_fps = min(current_config.get("fps", 30) * 2, 30)
        return {
            "action": "mine_frames",
            "justification": (
                f"Player detection at {player_map:.0%} — below the {PLAYER_MAP_FLOOR:.0%} target. "
                f"Only {frame_count} training frames available (need {MIN_FRAMES}+). "
                f"Increasing video sampling to {new_fps} frames per second "
                f"and focusing on high-action moments: net approaches, baseline rallies, and serve sequences."
            ),
            "config_updates": {
                "fps": new_fps,
                "mine_hard_frames": True,
                "hard_frame_types": ["net_approach", "baseline_rally", "serve"],
            },
        }

    if fp_rate > FP_RATE_CEILING:
        return {
            "action": "relabel",
            "justification": (
                f"Too many incorrect detections: {fp_rate:.0%} false positive rate "
                f"(target is below {FP_RATE_CEILING:.0%}). "
                f"Tightening rules for excluding ball persons, umpires, and crowd. "
                f"Re-running label review on the most affected frames."
            ),
            "config_updates": {
                "strict_crowd_exclusion": True,
                "rerun_labeling": True,
            },
        }

    any_below = (
        player_map < PLAYER_MAP_FLOOR or
        ball_map < BALL_MAP_FLOOR or
        court_lines_map < COURT_LINES_MAP_FLOOR or
        net_map < NET_MAP_FLOOR
    )

    if any_below:
        new_fps = min(current_config.get("fps", 30) + 5, 30)
        reason_parts = []
        if player_map < PLAYER_MAP_FLOOR:
            reason_parts.append(f"player at {player_map:.0%} (target {PLAYER_MAP_FLOOR:.0%})")
        if ball_map < BALL_MAP_FLOOR:
            reason_parts.append(f"ball at {ball_map:.0%} (target {BALL_MAP_FLOOR:.0%})")
        if court_lines_map < COURT_LINES_MAP_FLOOR:
            reason_parts.append(f"court lines at {court_lines_map:.0%} (target {COURT_LINES_MAP_FLOOR:.0%})")
        if net_map < NET_MAP_FLOOR:
            reason_parts.append(f"net at {net_map:.0%} (target {NET_MAP_FLOOR:.0%})")

        if map_plateau:
            return {
                "action": "mine_frames",
                "justification": (
                    f"Accuracy has plateaued — {' and '.join(reason_parts)}. "
                    f"Adding more diverse match situations: deuce-court rallies, ad-court serves, net volleys. "
                    f"Increasing sampling to {new_fps} frames per second."
                ),
                "config_updates": {
                    "fps": new_fps,
                    "mine_hard_frames": True,
                    "hard_frame_types": ["deuce_court", "ad_court_serve", "net_volley"],
                },
            }
        else:
            return {
                "action": "retrain",
                "justification": (
                    f"Making progress but not there yet — {' and '.join(reason_parts)}. "
                    f"Running another training pass with the current frame set and improved labels."
                ),
                "config_updates": {
                    "fps": new_fps,
                },
            }

    return {
        "action": "retrain",
        "justification": (
            f"Continuing training. Current: player {player_map:.0%}, ball {ball_map:.0%}, "
            f"court lines {court_lines_map:.0%}, net {net_map:.0%}. Unmet: {'; '.join(unmet)}."
        ),
        "config_updates": {},
    }
