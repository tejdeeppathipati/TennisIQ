"""
Structured output writer for TennisIQ pipeline results.

Produces:
  outputs/{job_id}/
    run.json                  ← run-level metadata, config, model info
    frames.jsonl              ← one JSON object per line: ball, players, court, homography
    events.json               ← detected bounce + hit events with in/out classification
    points.json               ← segmented tennis points with serve zone, end reason
    coaching_cards.json       ← FR-40: per-point plain-English coaching cards
    timeseries/
      ball_court.json         ← [{t, x, y, speed_m_s, accel_m_s2}, ...]
      player_a_court.json     ← [{t, x, y}, ...]
      player_b_court.json     ← [{t, x, y}, ...]
    stats.json                ← aggregated stats + insights text
    visuals/
      ball_heatmap.json       ← 2D grid of ball positions for court heatmap
      speed_histogram.json    ← speed distribution bins for chart
      serve_placement.json    ← FR-36: per-serve dots (green=in, red=fault)
      error_heatmap.json      ← FR-37: 2D histogram of out-bounce positions
      player_a_heatmap.json   ← FR-38: 2D histogram of Player A court coverage
      player_b_heatmap.json   ← FR-38: 2D histogram of Player B court coverage
      player_coverage.json    ← raw player positions
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tennisiq.cv.ball.inference import MAX_PLAUSIBLE_SPEED_M_S
from tennisiq.analytics.events import LINE_MARGIN_UNITS

logger = logging.getLogger(__name__)


def _safe_round(val, decimals=2):
    if val is None:
        return None
    if isinstance(val, float) and (np.isinf(val) or np.isnan(val)):
        return None
    return round(val, decimals)


def write_outputs(
    job_id: str,
    output_dir: str,
    video_path: str,
    fps: float,
    start_sec: float,
    end_sec: float,
    ball_physics: list,
    homographies: list,
    player_results: list,
    court_keypoints: list,
    timing: dict,
    events: list | None = None,
    points: list | None = None,
    shot_events: list | None = None,
    shot_directions: dict | None = None,
    analytics=None,
    coaching=None,
) -> dict:
    """
    Write all structured output files for a pipeline run.

    Returns a dict of output file paths.
    """
    base = Path(output_dir) / job_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "timeseries").mkdir(exist_ok=True)
    (base / "visuals").mkdir(exist_ok=True)

    paths = {}

    paths["run"] = _write_run_metadata(
        base, job_id, video_path, fps, start_sec, end_sec, timing,
        len(ball_physics),
    )
    paths["frames"] = _write_frames_jsonl(
        base, ball_physics, homographies, player_results, court_keypoints, fps, start_sec,
    )
    paths["timeseries"] = _write_timeseries(
        base, ball_physics, player_results, fps, start_sec,
    )
    paths["stats"] = _write_stats(
        base, ball_physics, homographies, player_results, timing, fps, events, points,
    )
    paths["visuals"] = _write_visuals(
        base, ball_physics, player_results,
    )
    if events is not None:
        paths["events"] = _write_events(base, events)
        paths["error_heatmap"] = _write_error_heatmap(base, events)
    if points is not None:
        paths["points"] = _write_points(base, points)
        paths["serve_placement"] = _write_serve_placement(base, points)
    if player_results:
        paths["player_heatmaps"] = _write_player_heatmaps(base, player_results)

    # ── New analytics outputs ─────────────────────────────────────────────
    if shot_events is not None:
        paths["shots"] = _write_shots(base, shot_events, shot_directions or {})
    if analytics is not None:
        paths["analytics"] = _write_analytics(base, analytics)
    if coaching is not None:
        paths["coaching_cards"] = _write_enhanced_coaching_cards(base, coaching)
        paths["player_a_card"] = _write_player_card(base, coaching, "player_a")
        paths["player_b_card"] = _write_player_card(base, coaching, "player_b")
        paths["match_flow"] = _write_match_flow(base, coaching)
    elif points is not None:
        paths["coaching_cards"] = _write_coaching_cards(base, points)

    logger.info(f"Outputs written to {base}")
    return paths


def _write_run_metadata(base, job_id, video_path, fps, start_sec, end_sec, timing, n_frames):
    data = {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video": {
            "source_path": video_path,
            "fps": fps,
            "segment_start_sec": start_sec,
            "segment_end_sec": end_sec,
            "frames_processed": n_frames,
        },
        "models": {
            "court": {"name": "ResNet50 Keypoint Regression", "checkpoint": "checkpoints/court_resnet/keypoints_model.pth", "input_size": "224x224", "output": "28 values → 14 (x,y) keypoints"},
            "ball": {"name": "YOLOv5L6u (tennis ball)", "checkpoint": "checkpoints/ball_yolo5/models_best.pt", "input_size": "640", "output": "bbox + confidence per frame"},
            "player": {"name": "YOLOv8n", "checkpoint": "yolov8n.pt (auto-download)", "tracker": "ByteTrack"},
        },
        "config": {
            "court_batch_size": 32,
            "person_conf_threshold": 0.5,
            "court_margin_units": 200,
            "outlier_max_dist": 100,
            "homography_confidence_threshold": 0.7,
            "max_carry_forward_frames": 5,
        },
        "timing": timing,
    }
    path = str(base / "run.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def _write_frames_jsonl(base, ball_physics, homographies, player_results, court_keypoints, fps, start_sec):
    """One JSON object per line — the core per-frame data."""
    path = str(base / "frames.jsonl")
    n = len(ball_physics)

    with open(path, "w") as f:
        for i in range(n):
            t = start_sec + i / fps

            frame = {
                "frame_idx": i,
                "timestamp_sec": _safe_round(t, 3),
            }

            bp = ball_physics[i]
            frame["ball"] = {
                "pixel_xy": list(bp.pixel_xy) if bp.pixel_xy[0] is not None else None,
                "court_xy": [_safe_round(bp.court_xy[0]), _safe_round(bp.court_xy[1])] if bp.court_xy[0] is not None else None,
                "speed_m_s": _safe_round(bp.speed_m_per_s),
                "accel_m_s2": _safe_round(bp.accel_m_per_s2),
            }

            if i < len(homographies):
                h = homographies[i]
                frame["homography"] = {
                    "detected_keypoints": h.detected_count,
                    "confidence": _safe_round(h.confidence, 3),
                    "reprojection_error_px": _safe_round(h.reprojection_error) if h.reprojection_error < float("inf") else None,
                    "reliable": h.reliable,
                    "carried_forward": h.carried_forward,
                }

            if i < len(court_keypoints):
                kps = court_keypoints[i]
                frame["court_keypoints"] = [
                    [_safe_round(x), _safe_round(y)] if x is not None else None
                    for x, y in kps
                ]

            if i < len(player_results):
                pr = player_results[i]
                players = {}
                for label, pl in [("player_a", pr.player_a), ("player_b", pr.player_b)]:
                    if pl:
                        players[label] = {
                            "bbox": [_safe_round(c) for c in pl.bbox],
                            "confidence": _safe_round(pl.confidence, 3),
                            "track_id": pl.track_id,
                            "foot_pixel": [_safe_round(pl.foot_pixel[0]), _safe_round(pl.foot_pixel[1])],
                            "foot_court": [_safe_round(pl.foot_court[0]), _safe_round(pl.foot_court[1])] if pl.foot_court[0] is not None else None,
                        }
                    else:
                        players[label] = None
                frame["players"] = players

            f.write(json.dumps(frame) + "\n")

    return path


def _write_timeseries(base, ball_physics, player_results, fps, start_sec):
    """Chart-ready time series for ball and player court positions."""
    paths = {}

    ball_ts = []
    for i, bp in enumerate(ball_physics):
        t = _safe_round(start_sec + i / fps, 3)
        entry = {"t": t, "frame_idx": i}
        if bp.court_xy[0] is not None:
            entry["x"] = _safe_round(bp.court_xy[0])
            entry["y"] = _safe_round(bp.court_xy[1])
            entry["speed_m_s"] = _safe_round(bp.speed_m_per_s)
            entry["speed_km_h"] = _safe_round(bp.speed_m_per_s * 3.6) if bp.speed_m_per_s else None
            entry["accel_m_s2"] = _safe_round(bp.accel_m_per_s2)
        ball_ts.append(entry)

    path = str(base / "timeseries" / "ball_court.json")
    with open(path, "w") as f:
        json.dump(ball_ts, f)
    paths["ball_court"] = path

    for label, attr in [("player_a", "player_a"), ("player_b", "player_b")]:
        ts = []
        for i, pr in enumerate(player_results):
            t = _safe_round(start_sec + i / fps, 3)
            entry = {"t": t, "frame_idx": i}
            pl = getattr(pr, attr)
            if pl and pl.foot_court[0] is not None:
                entry["x"] = _safe_round(pl.foot_court[0])
                entry["y"] = _safe_round(pl.foot_court[1])
                entry["foot_pixel_x"] = _safe_round(pl.foot_pixel[0])
                entry["foot_pixel_y"] = _safe_round(pl.foot_pixel[1])
            ts.append(entry)

        path = str(base / "timeseries" / f"{label}_court.json")
        with open(path, "w") as f:
            json.dump(ts, f)
        paths[f"{label}_court"] = path

    return paths


def _write_events(base, events):
    """Write detected bounce/hit events to events.json."""
    data = []
    for evt in events:
        entry = {
            "event_type": evt.event_type,
            "frame_idx": evt.frame_idx,
            "timestamp_sec": evt.timestamp_sec,
            "court_xy": [_safe_round(evt.court_xy[0]), _safe_round(evt.court_xy[1])],
            "speed_before_m_s": evt.speed_before_m_s,
            "speed_after_m_s": evt.speed_after_m_s,
            "direction_change_deg": evt.direction_change_deg,
            "score": evt.score,
            "court_side": evt.court_side,
        }
        if evt.event_type == "hit":
            entry["player"] = evt.player
            entry["player_distance"] = evt.player_distance
        if evt.event_type == "bounce":
            entry["in_out"] = evt.in_out
        data.append(entry)

    path = str(base / "events.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Wrote {len(data)} events to {path}")
    return path


def _write_points(base, points):
    """Write segmented tennis points to points.json."""
    data = []
    for pt in points:
        entry = {
            "point_idx": pt.point_idx,
            "start_frame": pt.start_frame,
            "end_frame": pt.end_frame,
            "start_sec": pt.start_sec,
            "end_sec": pt.end_sec,
            "serve_frame": pt.serve_frame,
            "serve_player": pt.serve_player,
            "first_bounce_frame": pt.first_bounce_frame,
            "first_bounce_court_xy": list(pt.first_bounce_court_xy) if pt.first_bounce_court_xy else None,
            "serve_zone": pt.serve_zone,
            "serve_fault_type": pt.serve_fault_type,
            "end_reason": pt.end_reason,
            "rally_hit_count": pt.rally_hit_count,
            "bounce_count": pt.bounce_count,
            "bounce_frames": pt.bounce_frames,
            "confidence": pt.confidence,
            "low_confidence_homography": pt.low_confidence_homography,
            "event_count": len(pt.events),
        }
        data.append(entry)

    path = str(base / "points.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Wrote {len(data)} points to {path}")
    return path


def _write_stats(base, ball_physics, homographies, player_results, timing, fps, events=None, points=None):
    """Aggregated statistics + human-readable insights."""
    n = len(ball_physics)

    ball_detected = sum(1 for bp in ball_physics if bp.pixel_xy[0] is not None)
    ball_projected = sum(1 for bp in ball_physics if bp.court_xy[0] is not None)
    speeds = [bp.speed_m_per_s for bp in ball_physics if bp.speed_m_per_s is not None]
    reliable = sum(1 for h in homographies if h.reliable)
    carried = sum(1 for h in homographies if h.carried_forward)
    confidences = [h.confidence for h in homographies]
    reproj = [h.reprojection_error for h in homographies if h.reprojection_error < float("inf")]
    pa_count = sum(1 for r in player_results if r.player_a is not None)
    pb_count = sum(1 for r in player_results if r.player_b is not None)
    both_count = sum(1 for r in player_results if r.player_a and r.player_b)
    raw_dets = sum(len(r.all_detections) for r in player_results)
    valid_dets = sum(sum(1 for d in r.all_detections if d.inside_court) for r in player_results)

    stats = {
        "frames": {
            "total": n,
            "duration_sec": _safe_round(n / fps, 2) if fps > 0 else 0,
        },
        "court": {
            "reliable_frames": reliable,
            "reliable_pct": _safe_round(reliable / n * 100, 1) if n else 0,
            "carried_forward_frames": carried,
            "avg_confidence": _safe_round(sum(confidences) / n, 3) if n else None,
            "avg_reprojection_error_px": _safe_round(sum(reproj) / len(reproj), 2) if reproj else None,
        },
        "ball": {
            "detected_frames": ball_detected,
            "detected_pct": _safe_round(ball_detected / n * 100, 1) if n else 0,
            "court_projected_frames": ball_projected,
            "avg_speed_m_s": _safe_round(sum(speeds) / len(speeds), 1) if speeds else None,
            "avg_speed_km_h": _safe_round(sum(speeds) / len(speeds) * 3.6, 1) if speeds else None,
            "max_speed_m_s": _safe_round(max(speeds), 1) if speeds else None,
            "max_speed_km_h": _safe_round(max(speeds) * 3.6, 1) if speeds else None,
            "median_speed_m_s": _safe_round(float(np.median(speeds)), 1) if speeds else None,
        },
        "players": {
            "player_a_frames": pa_count,
            "player_a_pct": _safe_round(pa_count / n * 100, 1) if n else 0,
            "player_b_frames": pb_count,
            "player_b_pct": _safe_round(pb_count / n * 100, 1) if n else 0,
            "both_visible_frames": both_count,
            "both_visible_pct": _safe_round(both_count / n * 100, 1) if n else 0,
            "raw_detections": raw_dets,
            "after_court_filter": valid_dets,
            "noise_removed_pct": _safe_round((raw_dets - valid_dets) / raw_dets * 100, 1) if raw_dets else 0,
        },
        "timing": timing,
    }

    if events:
        n_bounces = sum(1 for e in events if e.event_type == "bounce")
        n_hits = sum(1 for e in events if e.event_type == "hit")
        n_in = sum(1 for e in events if e.in_out == "in")
        n_out = sum(1 for e in events if e.in_out == "out")
        n_line = sum(1 for e in events if e.in_out == "line")
        n_hit_assigned = sum(1 for e in events if e.event_type == "hit" and e.player is not None)
        hit_scores = [e.score for e in events if e.event_type == "hit"]
        bounce_scores = [e.score for e in events if e.event_type == "bounce"]

        stats["events"] = {
            "total": len(events),
            "bounces": n_bounces,
            "hits": n_hits,
            "bounces_in": n_in,
            "bounces_out": n_out,
            "bounces_line": n_line,
            "hits_player_assigned": n_hit_assigned,
            "avg_bounce_score": _safe_round(sum(bounce_scores) / len(bounce_scores), 3) if bounce_scores else None,
            "avg_hit_score": _safe_round(sum(hit_scores) / len(hit_scores), 3) if hit_scores else None,
        }

    if points:
        end_reasons = {}
        serve_zones = {}
        for pt in points:
            end_reasons[pt.end_reason] = end_reasons.get(pt.end_reason, 0) + 1
            if pt.serve_zone:
                serve_zones[pt.serve_zone] = serve_zones.get(pt.serve_zone, 0) + 1
        rally_counts = [pt.rally_hit_count for pt in points]

        stats["points"] = {
            "total": len(points),
            "end_reasons": end_reasons,
            "serve_zones": serve_zones,
            "avg_rally_hits": _safe_round(sum(rally_counts) / len(rally_counts), 1) if rally_counts else None,
            "max_rally_hits": max(rally_counts) if rally_counts else None,
            "avg_confidence": _safe_round(sum(p.confidence for p in points) / len(points), 3) if points else None,
            "faults": sum(1 for p in points if p.serve_fault_type is not None),
        }

    insights = []
    if n > 0 and fps > 0:
        insights.append(f"Processed {n} frames ({_safe_round(n/fps, 1)}s of video) in {timing.get('total', '?')}s on GPU.")

        if reliable / n >= 0.85:
            avg_reproj = _safe_round(sum(reproj) / len(reproj), 1) if reproj else "N/A"
            insights.append(f"Court detection is strong: {reliable}/{n} frames ({_safe_round(reliable/n*100)}%) have reliable homography with {avg_reproj}px avg reprojection error.")
        else:
            insights.append(f"Court detection may need attention: only {reliable}/{n} frames ({_safe_round(reliable/n*100)}%) have reliable homography.")

        if speeds:
            avg_kmh = sum(speeds) / len(speeds) * 3.6
            max_kmh = max(speeds) * 3.6
            insights.append(f"Ball speed: avg {avg_kmh:.0f} km/h, max {max_kmh:.0f} km/h (capped at {MAX_PLAUSIBLE_SPEED_M_S} m/s to exclude artifacts).")

        if ball_detected / n < 0.5:
            insights.append(f"Ball detection rate is low ({_safe_round(ball_detected/n*100)}%). This segment may contain replays, close-ups, or poor visibility.")
        else:
            insights.append(f"Ball detected in {ball_detected}/{n} frames ({_safe_round(ball_detected/n*100)}%).")

        if pb_count / n < 0.3:
            insights.append(f"Far-side player (B) detected in only {_safe_round(pb_count/n*100)}% of frames — typical for broadcast camera angles where the far player appears small.")

        if raw_dets > valid_dets:
            insights.append(f"Court boundary filter removed {raw_dets - valid_dets} non-player detections ({_safe_round((raw_dets-valid_dets)/raw_dets*100)}% noise — spectators, ball boys, umpires).")

        if events:
            n_bounces = sum(1 for e in events if e.event_type == "bounce")
            n_hits = sum(1 for e in events if e.event_type == "hit")
            n_in = sum(1 for e in events if e.in_out == "in")
            n_out = sum(1 for e in events if e.in_out == "out")
            insights.append(f"Event detection found {n_bounces} bounces and {n_hits} hits in this segment.")
            if n_bounces > 0:
                insights.append(f"Bounce classification: {n_in} in, {n_out} out — using singles court polygon with {LINE_MARGIN_UNITS}-unit line margin.")

        if points:
            n_pts = len(points)
            avg_hits = sum(p.rally_hit_count for p in points) / n_pts if n_pts else 0
            insights.append(f"Point segmentation found {n_pts} point(s) with avg {avg_hits:.1f} hits per rally.")
            faults = sum(1 for p in points if p.serve_fault_type is not None)
            if faults:
                insights.append(f"Serve faults: {faults} ({', '.join(p.serve_fault_type for p in points if p.serve_fault_type)}).")

    stats["insights"] = insights

    path = str(base / "stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    return path


def _write_visuals(base, ball_physics, player_results):
    """Chart-ready data for frontend visualization."""
    paths = {}

    # Ball heatmap: 2D histogram of court positions
    from tennisiq.geometry.court_reference import CourtReference
    ref = CourtReference()

    ball_x = [bp.court_xy[0] for bp in ball_physics if bp.court_xy[0] is not None]
    ball_y = [bp.court_xy[1] for bp in ball_physics if bp.court_xy[0] is not None]

    if ball_x:
        x_bins = np.linspace(ref.baseline_top[0][0], ref.baseline_top[1][0], 20)
        y_bins = np.linspace(ref.baseline_top[0][1], ref.baseline_bottom[0][1], 30)
        hist, _, _ = np.histogram2d(ball_x, ball_y, bins=[x_bins, y_bins])
        heatmap_data = {
            "grid": hist.tolist(),
            "x_edges": x_bins.tolist(),
            "y_edges": y_bins.tolist(),
            "court_width": ref.court_width,
            "court_height": ref.court_height,
        }
    else:
        heatmap_data = {"grid": [], "x_edges": [], "y_edges": []}

    path = str(base / "visuals" / "ball_heatmap.json")
    with open(path, "w") as f:
        json.dump(heatmap_data, f)
    paths["ball_heatmap"] = path

    # Speed histogram
    speeds = [bp.speed_m_per_s * 3.6 for bp in ball_physics
              if bp.speed_m_per_s is not None and bp.speed_m_per_s < 200]
    if speeds:
        bins = list(range(0, 250, 10))
        counts, edges = np.histogram(speeds, bins=bins)
        speed_data = {
            "bin_edges_km_h": [float(e) for e in edges],
            "counts": [int(c) for c in counts],
            "mean_km_h": _safe_round(float(np.mean(speeds)), 1),
            "median_km_h": _safe_round(float(np.median(speeds)), 1),
            "p95_km_h": _safe_round(float(np.percentile(speeds, 95)), 1),
        }
    else:
        speed_data = {"bin_edges_km_h": [], "counts": []}

    path = str(base / "visuals" / "speed_histogram.json")
    with open(path, "w") as f:
        json.dump(speed_data, f)
    paths["speed_histogram"] = path

    # Player coverage: court positions for movement heatmaps
    coverage = {"player_a": [], "player_b": []}
    for pr in player_results:
        if pr.player_a and pr.player_a.foot_court[0] is not None:
            coverage["player_a"].append([
                _safe_round(pr.player_a.foot_court[0]),
                _safe_round(pr.player_a.foot_court[1]),
            ])
        if pr.player_b and pr.player_b.foot_court[0] is not None:
            coverage["player_b"].append([
                _safe_round(pr.player_b.foot_court[0]),
                _safe_round(pr.player_b.foot_court[1]),
            ])

    path = str(base / "visuals" / "player_coverage.json")
    with open(path, "w") as f:
        json.dump(coverage, f)
    paths["player_coverage"] = path

    return paths


def _write_serve_placement(base, points):
    """FR-36: Serve placement chart data — green dot for in, red for fault."""
    from tennisiq.analytics.points import SERVICE_BOXES

    serves = []
    for pt in points:
        if pt.first_bounce_court_xy is None:
            continue
        x, y = pt.first_bounce_court_xy
        serves.append({
            "point_idx": pt.point_idx,
            "court_x": _safe_round(x),
            "court_y": _safe_round(y),
            "serve_zone": pt.serve_zone,
            "is_fault": pt.serve_fault_type is not None,
            "fault_type": pt.serve_fault_type,
            "serve_player": pt.serve_player,
        })

    boxes = {}
    for name, (x_min, y_min, x_max, y_max) in SERVICE_BOXES.items():
        boxes[name] = {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}

    data = {"serves": serves, "service_boxes": boxes}
    path = str(base / "visuals" / "serve_placement.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Wrote serve placement chart: {len(serves)} serves to {path}")
    return path


def _write_error_heatmap(base, events):
    """FR-37: 2D histogram of all out-bounce positions in court-space."""
    from tennisiq.geometry.court_reference import CourtReference
    ref = CourtReference()

    out_x = [e.court_xy[0] for e in events if e.event_type == "bounce" and e.in_out == "out"]
    out_y = [e.court_xy[1] for e in events if e.event_type == "bounce" and e.in_out == "out"]

    if out_x:
        x_bins = np.linspace(ref.baseline_top[0][0] - 200, ref.baseline_top[1][0] + 200, 20)
        y_bins = np.linspace(ref.baseline_top[0][1] - 200, ref.baseline_bottom[0][1] + 200, 30)
        hist, _, _ = np.histogram2d(out_x, out_y, bins=[x_bins, y_bins])
        data = {
            "grid": hist.tolist(),
            "x_edges": x_bins.tolist(),
            "y_edges": y_bins.tolist(),
            "total_out_bounces": len(out_x),
            "positions": [
                {"x": _safe_round(x), "y": _safe_round(y)}
                for x, y in zip(out_x, out_y)
            ],
        }
    else:
        data = {"grid": [], "x_edges": [], "y_edges": [], "total_out_bounces": 0, "positions": []}

    path = str(base / "visuals" / "error_heatmap.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Wrote error heatmap: {len(out_x)} out-bounces to {path}")
    return path


def _write_player_heatmaps(base, player_results):
    """FR-38: Separate 2D histograms for Player A and Player B court coverage."""
    from tennisiq.geometry.court_reference import CourtReference
    ref = CourtReference()

    x_bins = np.linspace(ref.baseline_top[0][0], ref.baseline_top[1][0], 15)
    y_bins = np.linspace(ref.baseline_top[0][1], ref.baseline_bottom[0][1], 20)

    paths = {}
    for label, attr in [("player_a", "player_a"), ("player_b", "player_b")]:
        xs = []
        ys = []
        for pr in player_results:
            pl = getattr(pr, attr)
            if pl and pl.foot_court[0] is not None:
                xs.append(pl.foot_court[0])
                ys.append(pl.foot_court[1])

        if xs:
            hist, _, _ = np.histogram2d(xs, ys, bins=[x_bins, y_bins])
            data = {
                "grid": hist.tolist(),
                "x_edges": x_bins.tolist(),
                "y_edges": y_bins.tolist(),
                "total_frames": len(xs),
            }
        else:
            data = {"grid": [], "x_edges": [], "y_edges": [], "total_frames": 0}

        path = str(base / "visuals" / f"{label}_heatmap.json")
        with open(path, "w") as f:
            json.dump(data, f)
        paths[label] = path
        logger.info(f"Wrote {label} heatmap: {len(xs)} positions to {path}")

    return paths


def _write_coaching_cards(base, points):
    """FR-40: Per-point coaching card with plain-English explanation."""
    cards = []
    for pt in points:
        summary = _generate_point_summary(pt)
        suggestion = _generate_coaching_suggestion(pt)
        cards.append({
            "point_idx": pt.point_idx,
            "summary": summary,
            "suggestion": suggestion,
            "start_sec": pt.start_sec,
            "end_sec": pt.end_sec,
            "rally_hit_count": pt.rally_hit_count,
            "bounce_count": pt.bounce_count,
            "end_reason": pt.end_reason,
            "serve_zone": pt.serve_zone,
            "serve_fault_type": pt.serve_fault_type,
            "confidence": pt.confidence,
        })

    path = str(base / "coaching_cards.json")
    with open(path, "w") as f:
        json.dump(cards, f, indent=2)
    logger.info(f"Wrote {len(cards)} coaching cards to {path}")
    return path


def _generate_point_summary(pt) -> str:
    """Build a human-readable explanation of why this point ended."""
    duration = pt.end_sec - pt.start_sec
    parts = [f"Point {pt.point_idx}: {pt.rally_hit_count}-hit rally lasting {duration:.1f}s."]

    if pt.serve_zone:
        parts.append(f"Serve landed in the {pt.serve_zone.replace('_', ' ')} box.")
    elif pt.serve_fault_type:
        parts.append(f"Serve was a fault ({pt.serve_fault_type}).")

    reason_map = {
        "OUT": "The point ended with a ball landing out of bounds.",
        "DOUBLE_BOUNCE": "The point ended on a double bounce — the ball bounced twice on the same side before being returned.",
        "NET": "The ball hit the net, ending the point.",
        "BALL_LOST": "The ball went out of the camera's tracking range, likely after the rally ended.",
    }
    parts.append(reason_map.get(pt.end_reason, f"End reason: {pt.end_reason}."))

    return " ".join(parts)


def _generate_coaching_suggestion(pt) -> str:
    """Generate an actionable coaching suggestion grounded in the point data."""
    suggestions = []

    if pt.serve_fault_type == "wide":
        suggestions.append("The serve went wide — try aiming more toward the center of the service box to reduce fault risk.")
    elif pt.serve_fault_type == "long":
        suggestions.append("The serve was long — consider adding more topspin or reducing power to keep the ball inside the service line.")
    elif pt.serve_fault_type == "net":
        suggestions.append("The serve hit the net — try a higher ball toss or more upward swing path to clear the net with margin.")

    if pt.end_reason == "OUT":
        last_bounce = pt.events[-1] if pt.events else None
        if last_bounce and last_bounce.event_type == "bounce":
            if last_bounce.court_side == "near":
                suggestions.append("The final ball went out on the near side. Focus on depth control and keeping the ball inside the baseline.")
            else:
                suggestions.append("The final ball went out on the far side. Work on placement — aim for a target zone well inside the lines.")

    if pt.rally_hit_count >= 8:
        suggestions.append("This was a long rally. Good endurance, but look for opportunities to end the point earlier with an aggressive shot or approach to the net.")
    elif pt.rally_hit_count <= 2 and pt.end_reason != "BALL_LOST":
        suggestions.append("Very short rally — focus on extending rallies by improving consistency and shot selection.")

    if pt.end_reason == "DOUBLE_BOUNCE":
        suggestions.append("A double bounce means the player couldn't reach the ball in time. Work on court positioning and split-step timing to improve reaction speed.")

    if pt.end_reason == "NET":
        suggestions.append("Ball caught the net. Try hitting with more clearance over the net, especially on passing shots and returns.")

    if not suggestions:
        suggestions.append("Solid point. Continue working on consistency and shot placement.")

    return " ".join(suggestions)


# ─── New Analytics Output Writers ─────────────────────────────────────────────

def _write_shots(base, shot_events, shot_directions):
    """Write shots.json with all detected shot events."""
    data = []
    for s in shot_events:
        data.append({
            "frame_idx": s.frame_idx,
            "timestamp_sec": s.timestamp_sec,
            "owner": s.owner,
            "ball_court_xy": list(s.ball_court_xy),
            "shot_type": s.shot_type,
            "shot_type_confidence": _safe_round(s.shot_type_confidence, 3),
            "ball_direction_deg": _safe_round(s.ball_direction_deg, 1),
            "ball_direction_label": shot_directions.get(s.frame_idx, "unknown"),
            "speed_m_s": _safe_round(s.speed_m_s, 2),
            "court_side": s.court_side,
            "ownership_method": s.ownership_method,
        })

    path = str(base / "shots.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Wrote {len(data)} shots to {path}")
    return path


def _write_analytics(base, analytics):
    """Write analytics.json with full match analytics."""
    from tennisiq.analytics.match_analytics import analytics_to_dict

    data = analytics_to_dict(analytics)
    path = str(base / "analytics.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Wrote analytics to {path}")
    return path


def _write_enhanced_coaching_cards(base, coaching):
    """Write coaching_cards.json from coaching intelligence engine."""
    from tennisiq.analytics.coaching_intelligence import coaching_to_dict

    data = coaching_to_dict(coaching)
    cards = data.get("coaching_cards", [])

    path = str(base / "coaching_cards.json")
    with open(path, "w") as f:
        json.dump(cards, f, indent=2)
    logger.info(f"Wrote {len(cards)} enhanced coaching cards to {path}")
    return path


def _write_player_card(base, coaching, player_label):
    """Write player_a_card.json or player_b_card.json."""
    from tennisiq.analytics.coaching_intelligence import coaching_to_dict

    data = coaching_to_dict(coaching)
    card_key = f"{player_label}_card"
    weakness_key = f"{player_label}_weaknesses"

    card_data = {
        "card": data.get(card_key, {}),
        "weaknesses": data.get(weakness_key, {}),
    }

    path = str(base / f"{player_label}_card.json")
    with open(path, "w") as f:
        json.dump(card_data, f, indent=2)
    logger.info(f"Wrote {player_label} card to {path}")
    return path


def _write_match_flow(base, coaching):
    """Write match_flow.json with flow insights."""
    from tennisiq.analytics.coaching_intelligence import coaching_to_dict

    data = coaching_to_dict(coaching)
    flow_data = {
        "insights": data.get("match_flow_insights", []),
    }

    path = str(base / "match_flow.json")
    with open(path, "w") as f:
        json.dump(flow_data, f, indent=2)
    logger.info(f"Wrote match flow to {path}")
    return path
