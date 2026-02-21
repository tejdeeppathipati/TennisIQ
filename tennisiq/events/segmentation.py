from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


Point = Tuple[Optional[float], Optional[float]]


@dataclass
class PointEvent:
    frame_idx: int
    score: float


def _visible(frame: Dict) -> bool:
    if "ball_visible" in frame:
        return bool(frame["ball_visible"])
    bx, by = frame.get("ball_xy", (None, None))
    return bx is not None and by is not None


def _same_side(p1: Point, p2: Point, net_y: float) -> bool:
    if p1[1] is None or p2[1] is None:
        return False
    return (float(p1[1]) - net_y) * (float(p2[1]) - net_y) > 0


def _point_confidence(frames: Sequence[Dict], start: int, end: int, event_scores: List[float]) -> float:
    span = max(1, end - start + 1)
    segment = frames[start : end + 1]
    visible_ratio = sum(1 for f in segment if _visible(f)) / span

    hom = [float(f.get("homography_confidence", 0.0)) for f in segment]
    hom_conf = sum(hom) / max(1, len(hom))

    ev_conf = sum(event_scores) / max(1, len(event_scores)) if event_scores else 0.3

    conf = 0.4 * ev_conf + 0.3 * visible_ratio + 0.3 * hom_conf
    return float(max(0.0, min(1.0, conf)))


def run_point_state_machine(
    frames: Sequence[Dict],
    bounce_events: Sequence[Dict],
    hit_events: Sequence[Dict],
    fps: int,
    inactivity_frames: int = 24,
    ball_lost_frames: int = 12,
    serve_speed_thresh: float = 600.0,
    net_y: float = 1748.0,
) -> List[Dict]:
    bounce_by_idx = {int(e["frame_idx"]): e for e in bounce_events}
    hit_by_idx = {int(e["frame_idx"]): e for e in hit_events}

    points: List[Dict] = []
    state = "IDLE"

    missing = inactivity_frames
    curr: Dict | None = None

    for i, frame in enumerate(frames):
        visible = _visible(frame)
        speed = float(frame.get("ball_speed", 0.0))

        if state == "IDLE":
            if visible and missing >= inactivity_frames:
                curr = {
                    "point_id": len(points) + 1,
                    "start_frame": i,
                    "serve_frame": None,
                    "bounces": [],
                    "hits": [],
                    "event_scores": [],
                }
                state = "SERVE_SETUP"
                missing = 0
            else:
                missing = missing + 1 if not visible else 0
            continue

        # Point active states.
        if visible:
            missing = 0
        else:
            missing += 1

        assert curr is not None

        if i in hit_by_idx:
            curr["hits"].append(i)
            curr["event_scores"].append(float(hit_by_idx[i].get("score", 0.5)))
            if curr["serve_frame"] is None:
                curr["serve_frame"] = i

        if curr["serve_frame"] is None and speed >= serve_speed_thresh:
            curr["serve_frame"] = i

        if i in bounce_by_idx:
            b = bounce_by_idx[i]
            inout = str(frame.get("ball_inout", "unknown"))
            p = tuple(frame.get("ball_court_xy", (None, None)))
            curr["bounces"].append({"frame_idx": i, "inout": inout, "point": p, "score": float(b.get("score", 0.5))})
            curr["event_scores"].append(float(b.get("score", 0.5)))

            # Transition toward rally once first bounce/hit appears.
            if state in {"SERVE_SETUP", "SERVE_FLIGHT"}:
                state = "RALLY"

            # End reasons by bounce classification.
            if inout == "out":
                end_reason = "OUT"
            else:
                end_reason = None

            # Double bounce on same side.
            if end_reason is None and len(curr["bounces"]) >= 2:
                p1 = curr["bounces"][-2]["point"]
                p2 = curr["bounces"][-1]["point"]
                if _same_side(p1, p2, net_y=net_y):
                    end_reason = "DOUBLE_BOUNCE"

            # Net fault cue: bounce/termination near net with no clear in/out.
            if end_reason is None and p[1] is not None and abs(float(p[1]) - net_y) <= 80 and inout in {"unknown", "out"}:
                if state in {"SERVE_SETUP", "SERVE_FLIGHT"}:
                    end_reason = "NET"

            if end_reason is not None:
                end_frame = i
                conf = _point_confidence(frames, curr["start_frame"], end_frame, curr["event_scores"])
                first_bounce = curr["bounces"][0] if curr["bounces"] else None
                points.append(
                    {
                        "point_id": curr["point_id"],
                        "start_frame": int(curr["start_frame"]),
                        "end_frame": int(end_frame),
                        "start_sec": float(curr["start_frame"] / max(fps, 1)),
                        "end_sec": float(end_frame / max(fps, 1)),
                        "end_reason": end_reason,
                        "serve_frame": curr["serve_frame"],
                        "first_bounce_frame": first_bounce["frame_idx"] if first_bounce else None,
                        "first_bounce_court_xy": first_bounce["point"] if first_bounce else (None, None),
                        "serve_zone": None,
                        "rally_hit_count": int(len(curr["hits"])),
                        "confidence": conf,
                        "bounces": [b["frame_idx"] for b in curr["bounces"]],
                    }
                )
                state = "IDLE"
                curr = None
                missing = 0
                continue

        # Ball-lost termination.
        if missing > ball_lost_frames:
            end_frame = max(curr["start_frame"], i - missing)
            conf = _point_confidence(frames, curr["start_frame"], end_frame, curr["event_scores"])
            first_bounce = curr["bounces"][0] if curr["bounces"] else None
            points.append(
                {
                    "point_id": curr["point_id"],
                    "start_frame": int(curr["start_frame"]),
                    "end_frame": int(end_frame),
                    "start_sec": float(curr["start_frame"] / max(fps, 1)),
                    "end_sec": float(end_frame / max(fps, 1)),
                    "end_reason": "BALL_LOST",
                    "serve_frame": curr["serve_frame"],
                    "first_bounce_frame": first_bounce["frame_idx"] if first_bounce else None,
                    "first_bounce_court_xy": first_bounce["point"] if first_bounce else (None, None),
                    "serve_zone": None,
                    "rally_hit_count": int(len(curr["hits"])),
                    "confidence": conf,
                    "bounces": [b["frame_idx"] for b in curr["bounces"]],
                }
            )
            state = "IDLE"
            curr = None
            missing = 0

    if curr is not None:
        end_frame = len(frames) - 1
        conf = _point_confidence(frames, curr["start_frame"], end_frame, curr["event_scores"])
        first_bounce = curr["bounces"][0] if curr["bounces"] else None
        points.append(
            {
                "point_id": curr["point_id"],
                "start_frame": int(curr["start_frame"]),
                "end_frame": int(end_frame),
                "start_sec": float(curr["start_frame"] / max(fps, 1)),
                "end_sec": float(end_frame / max(fps, 1)),
                "end_reason": "BALL_LOST",
                "serve_frame": curr["serve_frame"],
                "first_bounce_frame": first_bounce["frame_idx"] if first_bounce else None,
                "first_bounce_court_xy": first_bounce["point"] if first_bounce else (None, None),
                "serve_zone": None,
                "rally_hit_count": int(len(curr["hits"])),
                "confidence": conf,
                "bounces": [b["frame_idx"] for b in curr["bounces"]],
            }
        )

    return points
