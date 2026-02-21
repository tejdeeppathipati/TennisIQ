from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from tennisiq.events.bounces import detect_bounce_candidates, score_bounce_candidates
from tennisiq.events.features import compute_track_kinematics
from tennisiq.events.hits import detect_hit_events
from tennisiq.events.segmentation import run_point_state_machine
from tennisiq.geometry.court_reference import CourtReference
from tennisiq.geometry.homography import get_trans_matrix, refer_kps
from tennisiq.geometry.polygons import (
    build_court_geometry,
    classify_point_in_out_line,
    service_box_label,
    side_label,
)
from tennisiq.io.schemas import FrameRecord


Point = Tuple[Optional[float], Optional[float]]


def _safe_point(p) -> Point:
    if p is None:
        return (None, None)
    x, y = p
    if x is None or y is None:
        return (None, None)
    return (float(x), float(y))


def _project_point(point: Point, h_img_to_court: Optional[np.ndarray]) -> Point:
    if h_img_to_court is None or point[0] is None or point[1] is None:
        return (None, None)
    src = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, h_img_to_court)
    px, py = dst[0][0]
    return (float(px), float(py))


def _bbox_footpoint(bbox) -> Point:
    if bbox is None:
        return (None, None)
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) / 2.0), float(y2))


def _median_filter_points(points: Sequence[Point], window: int = 3) -> List[Point]:
    if window <= 1:
        return list(points)
    half = window // 2
    out: List[Point] = []
    for i in range(len(points)):
        xs, ys = [], []
        for j in range(max(0, i - half), min(len(points), i + half + 1)):
            x, y = points[j]
            if x is not None and y is not None:
                xs.append(float(x))
                ys.append(float(y))
        if xs:
            out.append((float(np.median(xs)), float(np.median(ys))))
        else:
            out.append((None, None))
    return out


def _homography_confidence(pred_points: Sequence[Point], h_ref_to_img: Optional[np.ndarray], prev_h: Optional[np.ndarray]) -> float:
    if h_ref_to_img is None:
        return 0.0

    pred = cv2.perspectiveTransform(refer_kps, h_ref_to_img)
    errs = []
    for i in range(min(len(pred_points), pred.shape[0])):
        p = pred_points[i]
        if p[0] is None or p[1] is None:
            continue
        x_ref, y_ref = pred[i][0]
        errs.append(float(np.hypot(float(p[0]) - x_ref, float(p[1]) - y_ref)))

    if errs:
        err_mean = float(np.mean(errs))
        conf_err = float(np.exp(-err_mean / 45.0))
    else:
        conf_err = 0.15

    if prev_h is not None:
        denom = max(1e-6, float(np.linalg.norm(prev_h)))
        diff = float(np.linalg.norm(h_ref_to_img - prev_h) / denom)
        conf_temp = float(np.exp(-2.0 * diff))
    else:
        conf_temp = 1.0

    return float(max(0.0, min(1.0, 0.7 * conf_err + 0.3 * conf_temp)))


def _serve_miss_type(point: Point, ref: CourtReference) -> str:
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


def _point_card(point: Dict) -> Dict[str, str | int]:
    reason = point.get("end_reason", "BALL_LOST")
    zone = point.get("serve_zone")
    conf = float(point.get("confidence", 0.0))

    if reason == "OUT":
        why = "Point ended on an out bounce after rally pressure."
        tip = "Add more net clearance and target deeper margin instead of the line."
    elif reason == "DOUBLE_BOUNCE":
        why = "Second bounce on same side ended the point."
        tip = "Recover to center faster and prioritize the first reachable contact."
    elif reason == "NET":
        why = "Serve/rally trajectory likely clipped the net zone."
        tip = "Increase launch angle and contact slightly higher in front."
    else:
        why = "Ball tracking was lost before a clear terminal bounce."
        tip = "Use a safer tempo shot to keep the rally pattern stable."

    if zone:
        why += f" Serve target zone: {zone}."

    if conf < 0.45:
        why += " (Low-confidence event classification.)"

    return {"point_id": int(point["point_id"]), "why": why, "try_instead": tip}


def run_step_05_map_and_points(
    frame_records: List[FrameRecord],
    fps: int,
    event_model_path: str | None = None,
    event_threshold: float = 0.5,
    line_margin_px: float = 12.0,
    serve_speed_thresh: float = 600.0,
    inactivity_frames: int = 24,
    ball_lost_frames: int = 12,
) -> Dict[str, object]:
    ref = CourtReference()
    geometry = build_court_geometry(ref)
    net_y = float(ref.net[0][1])

    mapped_frames: List[Dict] = []

    # Homography solve with temporal stabilization.
    last_reliable_h_img_to_court: Optional[np.ndarray] = None
    last_reliable_h_ref_to_img: Optional[np.ndarray] = None
    last_reliable_conf = 0.0
    last_reliable_idx = -9999

    homography_series = []

    for i, fr in enumerate(tqdm(frame_records, desc="Step 5/6 Mapping + events", unit="frame")):
        row = fr.to_dict()
        kps = [tuple(p) for p in row["court_keypoints"]]

        h_ref_to_img = get_trans_matrix(kps)
        conf = _homography_confidence(kps, h_ref_to_img, last_reliable_h_ref_to_img)

        h_img_to_court = None
        carried = False
        if h_ref_to_img is not None and conf >= 0.18:
            try:
                h_img_to_court = np.linalg.inv(h_ref_to_img)
            except np.linalg.LinAlgError:
                h_img_to_court = None

        if h_img_to_court is None and last_reliable_h_img_to_court is not None and (i - last_reliable_idx) <= 5:
            # Carry forward last reliable homography for short occlusion windows.
            h_img_to_court = last_reliable_h_img_to_court
            conf = max(0.0, last_reliable_conf * (0.85 ** (i - last_reliable_idx)))
            carried = True

        if h_img_to_court is not None and not carried:
            last_reliable_h_img_to_court = h_img_to_court
            last_reliable_h_ref_to_img = h_ref_to_img
            last_reliable_conf = conf
            last_reliable_idx = i

        row["homography_ok"] = h_img_to_court is not None
        row["homography_confidence"] = float(conf)

        ball_xy = _safe_point(row.get("ball_xy", (None, None)))
        row["ball_xy"] = ball_xy
        row["ball_visible"] = ball_xy[0] is not None and ball_xy[1] is not None

        row["ball_court_xy"] = _project_point(ball_xy, h_img_to_court)

        player_a_foot = _bbox_footpoint(row.get("playerA_bbox"))
        player_b_foot = _bbox_footpoint(row.get("playerB_bbox"))
        row["playerA_court_xy"] = _project_point(player_a_foot, h_img_to_court)
        row["playerB_court_xy"] = _project_point(player_b_foot, h_img_to_court)

        row["event_candidates"] = {"bounce": False, "hit": False}
        row["event_scores"] = {"bounce": 0.0, "hit": 0.0}
        row["event_reasons"] = []

        homography_series.append({"frame_idx": i, "confidence": float(conf), "carried": carried, "ok": bool(row["homography_ok"])})
        mapped_frames.append(row)

    # Ball smoothing in court coordinates.
    ball_court_raw = [tuple(r["ball_court_xy"]) for r in mapped_frames]
    ball_court_smooth = _median_filter_points(ball_court_raw, window=3)

    for i, p in enumerate(ball_court_smooth):
        mapped_frames[i]["ball_court_xy"] = p

    # Kinematics.
    kin = compute_track_kinematics(ball_court_smooth, fps=fps)
    for i, k in enumerate(kin):
        mapped_frames[i]["ball_speed"] = float(k.get("speed", 0.0)) if np.isfinite(float(k.get("speed", np.nan))) else 0.0
        mapped_frames[i]["ball_accel"] = float(k.get("accel", 0.0)) if np.isfinite(float(k.get("accel", np.nan))) else 0.0

    # Calibrated in/out per frame.
    for i, row in enumerate(mapped_frames):
        if not row["homography_ok"]:
            row["ball_inout"] = "unknown"
            row["line_info"] = {"closest_line": "", "line_distance": None}
            continue
        label, meta = classify_point_in_out_line(row["ball_court_xy"], geometry, line_margin=line_margin_px)
        ld = meta.get("line_distance")
        if ld is None or not np.isfinite(float(ld)):
            meta["line_distance"] = None
        else:
            meta["line_distance"] = float(ld)
        row["ball_inout"] = label
        row["line_info"] = meta

    # Hybrid event detection.
    bounce_candidates = detect_bounce_candidates(
        ball_track=ball_court_smooth,
        ball_court_track=ball_court_smooth,
        net_y=net_y,
        fps=fps,
    )
    bounce_events = score_bounce_candidates(
        candidates=bounce_candidates,
        ball_track=ball_court_smooth,
        mapped_frames=mapped_frames,
        net_y=net_y,
        fps=fps,
        model_path=event_model_path,
        threshold=event_threshold,
        nms_window=3,
    )
    hit_events = detect_hit_events(ball_court_smooth, min_turn_angle_deg=50.0, min_speed=6.0, nms_window=3)

    bounce_by_idx = {int(e["frame_idx"]): e for e in bounce_events}
    hit_by_idx = {int(e["frame_idx"]): e for e in hit_events}

    for i, row in enumerate(mapped_frames):
        if i in bounce_by_idx:
            row["event_candidates"]["bounce"] = True
            row["event_scores"]["bounce"] = float(bounce_by_idx[i].get("score", 0.0))
            row["event_reasons"] += list(bounce_by_idx[i].get("reasons", []))
        if i in hit_by_idx:
            row["event_candidates"]["hit"] = True
            row["event_scores"]["hit"] = float(hit_by_idx[i].get("score", 0.0))
            row["event_reasons"].append("direction_change")

    points = run_point_state_machine(
        frames=mapped_frames,
        bounce_events=bounce_events,
        hit_events=hit_events,
        fps=fps,
        inactivity_frames=inactivity_frames,
        ball_lost_frames=ball_lost_frames,
        serve_speed_thresh=serve_speed_thresh,
        net_y=net_y,
    )

    # Add serve zone labels.
    for p in points:
        fb = p.get("first_bounce_court_xy", (None, None))
        p["serve_zone"] = service_box_label(fb, geometry) if fb[0] is not None and fb[1] is not None else None

    # Stats and visuals.
    serve_total = 0
    serve_in = 0
    serve_miss_types = []
    serve_placement_points = []
    error_heatmap_points = []

    for p in points:
        fb = p.get("first_bounce_court_xy", (None, None))
        if fb[0] is None or fb[1] is None:
            continue
        serve_total += 1
        zone = p.get("serve_zone")
        if zone is not None:
            serve_in += 1
        else:
            serve_miss_types.append(_serve_miss_type(fb, ref))
        serve_placement_points.append({"point_id": p["point_id"], "serve_bounce": fb, "zone": zone})

    for e in bounce_events:
        idx = int(e["frame_idx"])
        label = mapped_frames[idx].get("ball_inout", "unknown")
        pt = mapped_frames[idx].get("ball_court_xy", (None, None))
        if label == "out" and pt[0] is not None and pt[1] is not None:
            error_heatmap_points.append(pt)

    serve_in_pct = (100.0 * serve_in / serve_total) if serve_total else 0.0

    miss_counter = Counter(serve_miss_types)
    top_miss = miss_counter.most_common(1)[0][0] if miss_counter else "unknown"

    error_sides = [side_label(p, geometry.center_x) for p in error_heatmap_points]
    side_counts = Counter(error_sides)
    top_side = side_counts.most_common(1)[0][0] if side_counts else "deuce"

    insights = [
        f"Your serves miss mostly {top_miss}.",
        f"Your errors cluster on the {top_side} side.",
    ]

    point_cards = [_point_card(p) for p in points]

    hom_conf_vals = [float(x["confidence"]) for x in homography_series]
    tracks = {
        "ball_projected": ball_court_raw,
        "playerA_projected": [row["playerA_court_xy"] for row in mapped_frames],
        "playerB_projected": [row["playerB_court_xy"] for row in mapped_frames],
        "ball_smoothed": ball_court_smooth,
        "homography_series": {
            "series": homography_series,
            "mean_confidence": float(np.mean(hom_conf_vals)) if hom_conf_vals else 0.0,
            "valid_ratio": float(sum(1 for x in homography_series if x["ok"]) / max(1, len(homography_series))),
        },
    }

    stats = {
        "num_frames": len(mapped_frames),
        "num_points": len(points),
        "num_bounces": len(bounce_events),
        "num_hits": len(hit_events),
        "serve_total": serve_total,
        "serve_in": serve_in,
        "serve_in_pct": serve_in_pct,
        "end_reason_distribution": Counter(p["end_reason"] for p in points),
    }

    visuals = {
        "serve_placement_points": serve_placement_points,
        "error_heatmap_points": error_heatmap_points,
    }

    return {
        "frames": mapped_frames,
        "tracks": tracks,
        "points": points,
        "stats": stats,
        "visuals": visuals,
        "insights": insights,
        "point_cards": point_cards,
    }
