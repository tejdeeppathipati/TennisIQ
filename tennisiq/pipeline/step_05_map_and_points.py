from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from tennisiq.events.bounces import detect_bounces
from tennisiq.events.inout import classify_in_out
from tennisiq.geometry.court_reference import CourtReference
from tennisiq.geometry.homography import get_trans_matrix
from tennisiq.io.schemas import FrameRecord


Point = Tuple[Optional[float], Optional[float]]


def _project_point(point: Point, h_img_to_court: Optional[np.ndarray]) -> Point:
    if h_img_to_court is None:
        return (None, None)
    x, y = point
    if x is None or y is None:
        return (None, None)
    src = np.array([[[float(x), float(y)]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, h_img_to_court)
    px, py = dst[0][0]
    return (float(px), float(py))


def _bbox_footpoint(bbox) -> Point:
    if bbox is None:
        return (None, None)
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) / 2.0), float(y2))


def _service_zone_label(point: Point, ref: CourtReference) -> Optional[str]:
    x, y = point
    if x is None or y is None:
        return None

    left_x = ref.left_inner_line[0][0]
    mid_x = ref.middle_line[0][0]
    right_x = ref.right_inner_line[0][0]
    top_y = ref.top_inner_line[0][1]
    net_y = ref.net[0][1]
    bottom_y = ref.bottom_inner_line[0][1]

    if top_y <= y <= net_y:
        if left_x <= x <= mid_x:
            return "top_left"
        if mid_x < x <= right_x:
            return "top_right"
    if net_y < y <= bottom_y:
        if left_x <= x <= mid_x:
            return "bottom_left"
        if mid_x < x <= right_x:
            return "bottom_right"
    return None


def _classify_serve_miss(point: Point, ref: CourtReference) -> str:
    x, y = point
    if x is None or y is None:
        return "unknown"

    left_x = ref.left_inner_line[0][0]
    right_x = ref.right_inner_line[0][0]
    top_service_y = ref.top_inner_line[0][1]
    bottom_service_y = ref.bottom_inner_line[0][1]

    if y < top_service_y or y > bottom_service_y:
        return "long"
    if x < left_x or x > right_x:
        return "wide"
    return "net_or_body"


def _segment_points(frames: List[Dict], bounce_idxs: List[int], fps: int, inactivity_frames: int = 24, ball_lost_frames: int = 12):
    points = []
    in_point = False
    point_start = None
    missing = inactivity_frames

    bounce_set = set(bounce_idxs)

    for i, frame in enumerate(frames):
        ball = frame.get("ball_xy", (None, None))
        visible = ball[0] is not None and ball[1] is not None

        if not in_point:
            if visible and missing >= inactivity_frames:
                in_point = True
                point_start = i
                missing = 0
            else:
                missing = missing + 1 if not visible else 0
            continue

        # in_point branch
        if visible:
            missing = 0
        else:
            missing += 1

        end_reason = None
        end_idx = None
        if i in bounce_set and frame.get("ball_inout") == "out":
            end_reason = "OUT"
            end_idx = i
        elif missing > ball_lost_frames:
            end_reason = "BALL LOST"
            end_idx = i - missing

        if end_reason is not None and point_start is not None:
            end_idx = max(end_idx, point_start)
            point_bounces = [b for b in bounce_idxs if point_start <= b <= end_idx]
            points.append(
                {
                    "point_id": len(points) + 1,
                    "start_frame": int(point_start),
                    "end_frame": int(end_idx),
                    "start_sec": float(point_start / max(fps, 1)),
                    "end_sec": float(end_idx / max(fps, 1)),
                    "end_reason": end_reason,
                    "bounces": point_bounces,
                }
            )
            in_point = False
            point_start = None
            missing = 0

    if in_point and point_start is not None:
        end_idx = len(frames) - 1
        point_bounces = [b for b in bounce_idxs if point_start <= b <= end_idx]
        points.append(
            {
                "point_id": len(points) + 1,
                "start_frame": int(point_start),
                "end_frame": int(end_idx),
                "start_sec": float(point_start / max(fps, 1)),
                "end_sec": float(end_idx / max(fps, 1)),
                "end_reason": "VIDEO END",
                "bounces": point_bounces,
            }
        )

    return points


def run_step_05_map_and_points(frame_records: List[FrameRecord], fps: int) -> Dict[str, object]:
    ref = CourtReference()
    singles_court_poly = [ref.left_inner_line[0], ref.right_inner_line[0], ref.right_inner_line[1], ref.left_inner_line[1]]

    mapped_frames: List[Dict] = []
    ball_track = [fr.ball_xy for fr in frame_records]
    bounce_idxs = detect_bounces(ball_track)

    for fr in frame_records:
        row = fr.to_dict()
        h_ref_to_img = get_trans_matrix(row["court_keypoints"])
        h_img_to_court = None
        if h_ref_to_img is not None:
            try:
                h_img_to_court = np.linalg.inv(h_ref_to_img)
            except np.linalg.LinAlgError:
                h_img_to_court = None

        row["homography_ok"] = h_img_to_court is not None
        row["ball_court_xy"] = _project_point(tuple(row["ball_xy"]), h_img_to_court)

        player_a_foot = _bbox_footpoint(row.get("playerA_bbox"))
        player_b_foot = _bbox_footpoint(row.get("playerB_bbox"))
        row["playerA_court_xy"] = _project_point(player_a_foot, h_img_to_court)
        row["playerB_court_xy"] = _project_point(player_b_foot, h_img_to_court)
        row["ball_inout"] = classify_in_out(tuple(row["ball_court_xy"]), singles_court_poly)
        mapped_frames.append(row)

    points = _segment_points(mapped_frames, bounce_idxs=bounce_idxs, fps=fps)

    # Serve stats and maps from first bounce in each point.
    serve_total = 0
    serve_in = 0
    serve_points = []
    serve_miss_types: List[str] = []

    error_points = []
    for idx in bounce_idxs:
        p = mapped_frames[idx]["ball_court_xy"]
        if p[0] is None or p[1] is None:
            continue
        if mapped_frames[idx]["ball_inout"] == "out":
            error_points.append(p)

    for p in points:
        if not p["bounces"]:
            continue
        first_bounce_idx = p["bounces"][0]
        first_bounce = mapped_frames[first_bounce_idx]["ball_court_xy"]
        zone = _service_zone_label(first_bounce, ref)
        p["first_bounce_frame"] = first_bounce_idx
        p["first_bounce_court_xy"] = first_bounce
        p["serve_zone"] = zone

        if first_bounce[0] is not None and first_bounce[1] is not None:
            serve_total += 1
            if zone is not None:
                serve_in += 1
            else:
                serve_miss_types.append(_classify_serve_miss(first_bounce, ref))
            serve_points.append({"point_id": p["point_id"], "serve_bounce": first_bounce, "zone": zone})

    serve_in_pct = (100.0 * serve_in / serve_total) if serve_total else 0.0

    # Two simple textual insights.
    miss_counter = Counter(serve_miss_types)
    top_miss = miss_counter.most_common(1)[0][0] if miss_counter else "unknown"

    mid_x = ref.middle_line[0][0]
    deuce = sum(1 for x, _ in error_points if x is not None and x >= mid_x)
    ad = sum(1 for x, _ in error_points if x is not None and x < mid_x)
    side = "deuce" if deuce >= ad else "ad"

    insights = [
        f"Your serves miss mostly {top_miss}.",
        f"Your errors cluster on the {side} side.",
    ]

    tracks = {
        "ball_projected": [row["ball_court_xy"] for row in mapped_frames],
        "playerA_projected": [row["playerA_court_xy"] for row in mapped_frames],
        "playerB_projected": [row["playerB_court_xy"] for row in mapped_frames],
    }

    stats = {
        "num_frames": len(mapped_frames),
        "num_points": len(points),
        "num_bounces": len(bounce_idxs),
        "serve_total": serve_total,
        "serve_in": serve_in,
        "serve_in_pct": serve_in_pct,
    }

    visuals = {
        "serve_placement_points": serve_points,
        "error_heatmap_points": error_points,
    }

    return {
        "frames": mapped_frames,
        "tracks": tracks,
        "points": points,
        "stats": stats,
        "visuals": visuals,
        "insights": insights,
    }
