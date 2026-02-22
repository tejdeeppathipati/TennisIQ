import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tennisiq.geometry.court_reference import CourtReference
except ModuleNotFoundError:  # backend runs from backend/ without repo root on sys.path
    class CourtReference:
        def __init__(self) -> None:
            self.baseline_top = ((286, 561), (1379, 561))
            self.baseline_bottom = ((286, 2935), (1379, 2935))
            self.net = ((286, 1748), (1379, 1748))
            self.left_court_line = ((286, 561), (286, 2935))
            self.right_court_line = ((1379, 561), (1379, 2935))
            self.left_inner_line = ((423, 561), (423, 2935))
            self.right_inner_line = ((1242, 561), (1242, 2935))
            self.middle_line = ((832, 1110), (832, 2386))
            self.top_inner_line = ((423, 1110), (1242, 1110))
            self.bottom_inner_line = ((423, 2386), (1242, 2386))
            self.court_width = 1117
            self.court_height = 2408
            self.real_width_m = 10.97
            self.real_length_m = 23.77
            self.meters_per_unit = self.real_width_m / self.court_width


def build_analysis(run_dir: Path, job_id: str | None = None) -> dict | None:
    frames_path = run_dir / "frames.jsonl"
    if not frames_path.exists():
        return None

    run = _load_json(run_dir / "run.json") or {}
    stats = _load_json(run_dir / "stats.json") or {}
    events = _load_json(run_dir / "events.json") or []
    points = _load_json(run_dir / "points.json") or []
    ball_ts = _load_json(run_dir / "timeseries" / "ball_court.json") or []
    player_a_ts = _load_json(run_dir / "timeseries" / "player_a_court.json") or []
    player_b_ts = _load_json(run_dir / "timeseries" / "player_b_court.json") or []

    total_frames = 0
    ball_detected_frames = 0
    ball_projected_frames = 0
    homography_reliable_frames = 0
    player_a_frames = 0
    player_b_frames = 0
    both_frames = 0
    first_ts = None
    last_ts = None

    with frames_path.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_frames += 1
            ts = obj.get("timestamp_sec")
            if isinstance(ts, (int, float)):
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

            ball = obj.get("ball") or {}
            if ball.get("pixel_xy") is not None:
                ball_detected_frames += 1
            if ball.get("court_xy") is not None:
                ball_projected_frames += 1

            hom = obj.get("homography")
            if isinstance(hom, dict) and hom.get("reliable") is True:
                homography_reliable_frames += 1

            players = obj.get("players") or {}
            has_a = players.get("player_a") is not None
            has_b = players.get("player_b") is not None
            if has_a:
                player_a_frames += 1
            if has_b:
                player_b_frames += 1
            if has_a and has_b:
                both_frames += 1

    fps = (
        _safe_get(run, "video", "fps")
        or stats.get("fps")
        or _infer_fps_from_timestamps(total_frames, first_ts, last_ts)
    )

    duration_sec = _infer_duration(run, stats, total_frames, fps, first_ts, last_ts)

    ref = CourtReference()
    meters_per_unit = ref.meters_per_unit

    serve_stats = _compute_serve_stats(points, ref, meters_per_unit)
    rally_stats = _compute_rally_stats(points, events)
    error_stats = _compute_error_stats(events, ref, meters_per_unit)
    player_a_stats = _compute_player_stats(player_a_ts, ref, meters_per_unit)
    player_b_stats = _compute_player_stats(player_b_ts, ref, meters_per_unit)
    ball_stats = _compute_ball_stats(ball_ts, events)

    event_timeline = _compute_event_timeline(events)

    analysis = {
        "meta": {
            "job_id": job_id or run.get("job_id"),
            "fps": _safe_round(fps, 3) if fps else None,
            "duration_sec": _safe_round(duration_sec, 2) if duration_sec else None,
            "meters_per_unit": _safe_round(meters_per_unit, 6),
            "court": {
                "width_units": ref.court_width,
                "height_units": ref.court_height,
            },
        },
        "quality": {
            "frames_total": total_frames,
            "ball_coverage_pct": _percent(ball_detected_frames, total_frames),
            "ball_projected_pct": _percent(ball_projected_frames, total_frames),
            "homography_reliable_pct": _percent(homography_reliable_frames, total_frames),
            "player_visibility": {
                "player_a_pct": _percent(player_a_frames, total_frames),
                "player_b_pct": _percent(player_b_frames, total_frames),
                "both_pct": _percent(both_frames, total_frames),
            },
            "events_total": len(events) if isinstance(events, list) else 0,
            "points_total": len(points) if isinstance(points, list) else 0,
        },
        "serve": serve_stats,
        "rally": rally_stats,
        "errors": error_stats,
        "players": {
            "player_a": player_a_stats,
            "player_b": player_b_stats,
        },
        "ball": ball_stats,
        "events": {
            "timeline": event_timeline,
        },
    }

    return analysis


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _safe_get(d: dict, *keys):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _safe_round(val: float | None, decimals: int = 2) -> float | None:
    if val is None or isinstance(val, str):
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return round(float(val), decimals)


def _percent(num: int, denom: int) -> float:
    if not denom:
        return 0.0
    return round(num / denom * 100, 1)


def _stat_summary(values: list[float], decimals: int = 2) -> dict | None:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return {
        "mean": _safe_round(float(arr.mean()), decimals),
        "median": _safe_round(float(np.median(arr)), decimals),
        "p90": _safe_round(float(np.percentile(arr, 90)), decimals),
        "p95": _safe_round(float(np.percentile(arr, 95)), decimals),
        "max": _safe_round(float(arr.max()), decimals),
    }


def _infer_fps_from_timestamps(total_frames: int, first_ts: float | None, last_ts: float | None) -> float | None:
    if total_frames < 2 or first_ts is None or last_ts is None:
        return None
    duration = last_ts - first_ts
    if duration <= 0:
        return None
    return (total_frames - 1) / duration


def _infer_duration(run: dict, stats: dict, total_frames: int, fps: float | None,
                    first_ts: float | None, last_ts: float | None) -> float | None:
    start = _safe_get(run, "video", "segment_start_sec")
    end = _safe_get(run, "video", "segment_end_sec")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
        return end - start
    if isinstance(stats, dict) and isinstance(stats.get("duration_sec"), (int, float)):
        return float(stats["duration_sec"])
    if first_ts is not None and last_ts is not None and last_ts >= first_ts:
        return last_ts - first_ts
    if fps and total_frames:
        return total_frames / fps
    return None


def _compute_serve_stats(points: list, ref: CourtReference, meters_per_unit: float) -> dict:
    zone_counts: dict[str, int] = {}
    depth_samples_m: list[float] = []
    width_samples_m: list[float] = []
    fault_count = 0
    total_serves = 0

    center_x = ref.middle_line[0][0]
    net_y = ref.net[0][1]
    far_service_y = ref.top_inner_line[0][1]
    near_service_y = ref.bottom_inner_line[0][1]

    for pt in points:
        if not isinstance(pt, dict):
            continue
        zone = pt.get("serve_zone")
        if zone:
            zone_counts[zone] = zone_counts.get(zone, 0) + 1

        bounce = pt.get("first_bounce_court_xy")
        if not bounce or len(bounce) != 2:
            continue
        x, y = bounce
        if x is None or y is None:
            continue

        total_serves += 1
        if pt.get("serve_fault_type") is not None:
            fault_count += 1

        side = "near" if (zone and zone.startswith("near")) or y >= net_y else "far"
        service_line_y = near_service_y if side == "near" else far_service_y
        depth_units = abs(service_line_y - y)
        width_units = abs(x - center_x)
        depth_samples_m.append(depth_units * meters_per_unit)
        width_samples_m.append(width_units * meters_per_unit)

    return {
        "zone_counts": zone_counts,
        "fault_rate": _safe_round(fault_count / total_serves, 3) if total_serves else None,
        "depth_stats": _stat_summary(depth_samples_m, decimals=2),
        "width_stats": _stat_summary(width_samples_m, decimals=2),
        "depth_samples_m": [_safe_round(v, 3) for v in depth_samples_m],
        "width_samples_m": [_safe_round(v, 3) for v in width_samples_m],
        "sample_count": total_serves,
    }


def _compute_rally_stats(points: list, events: list) -> dict:
    rally_hits: list[int] = []
    rally_durations: list[float] = []
    end_reason_counts: dict[str, int] = {}
    tempo_values: list[float] = []

    for pt in points:
        if not isinstance(pt, dict):
            continue
        hits = pt.get("rally_hit_count")
        if isinstance(hits, int):
            rally_hits.append(hits)
        start = pt.get("start_sec")
        end = pt.get("end_sec")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
            duration = float(end - start)
            rally_durations.append(duration)
            if isinstance(hits, int) and duration > 0:
                tempo_values.append(hits / duration)
        reason = pt.get("end_reason")
        if reason:
            end_reason_counts[reason] = end_reason_counts.get(reason, 0) + 1

    hit_times = [e.get("timestamp_sec") for e in events if isinstance(e, dict) and e.get("event_type") == "hit"]
    hit_times = [t for t in hit_times if isinstance(t, (int, float))]
    hit_times.sort()
    inter_hit = [hit_times[i] - hit_times[i - 1] for i in range(1, len(hit_times)) if hit_times[i] > hit_times[i - 1]]

    return {
        "rally_hits": rally_hits,
        "rally_durations_sec": [_safe_round(v, 3) for v in rally_durations],
        "end_reason_counts": end_reason_counts,
        "tempo_stats": {
            "mean_hits_per_sec": _safe_round(float(np.mean(tempo_values)), 2) if tempo_values else None,
            "mean_inter_hit_sec": _safe_round(float(np.mean(inter_hit)), 3) if inter_hit else None,
            "p95_inter_hit_sec": _safe_round(float(np.percentile(inter_hit, 95)), 3) if inter_hit else None,
        },
    }


def _compute_error_stats(events: list, ref: CourtReference, meters_per_unit: float) -> dict:
    out_positions = []
    out_distances_m = []

    left_x = ref.left_inner_line[0][0]
    right_x = ref.right_inner_line[0][0]
    top_y = ref.baseline_top[0][1]
    bottom_y = ref.baseline_bottom[0][1]

    for e in events:
        if not isinstance(e, dict):
            continue
        if e.get("event_type") != "bounce" or e.get("in_out") != "out":
            continue
        xy = e.get("court_xy")
        if not xy or len(xy) != 2:
            continue
        x, y = xy
        if x is None or y is None:
            continue
        out_positions.append({"x": _safe_round(x, 2), "y": _safe_round(y, 2)})

        dx = max(left_x - x, 0, x - right_x)
        dy = max(top_y - y, 0, y - bottom_y)
        dist_units = math.hypot(dx, dy)
        out_distances_m.append(dist_units * meters_per_unit)

    return {
        "out_count": len(out_positions),
        "out_distance_stats": _stat_summary(out_distances_m, decimals=2),
        "error_positions": out_positions,
    }


def _compute_player_stats(player_ts: list, ref: CourtReference, meters_per_unit: float) -> dict | None:
    if not player_ts:
        return None

    positions = []
    for row in player_ts:
        if not isinstance(row, dict):
            continue
        x = row.get("x")
        y = row.get("y")
        t = row.get("t")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(t, (int, float)):
            positions.append((float(t), float(x), float(y)))

    if len(positions) < 2:
        return {
            "distance_m": 0.0,
            "speed_stats": None,
            "zone_time_pct": None,
        }

    positions.sort(key=lambda p: p[0])

    total_dist_units = 0.0
    speed_samples = []
    max_gap_sec = 0.5

    net_y = ref.net[0][1]
    top_service_y = ref.top_inner_line[0][1]
    bottom_service_y = ref.bottom_inner_line[0][1]
    net_band = 200.0
    zone_counts = {"baseline": 0, "mid": 0, "net": 0}

    for i in range(1, len(positions)):
        t0, x0, y0 = positions[i - 1]
        t1, x1, y1 = positions[i]
        dt = t1 - t0
        if dt <= 0 or dt > max_gap_sec:
            continue
        dist = math.hypot(x1 - x0, y1 - y0)
        total_dist_units += dist
        speed_samples.append(dist / dt)

    for _, _, y in positions:
        if abs(y - net_y) <= net_band:
            zone_counts["net"] += 1
        elif y <= top_service_y or y >= bottom_service_y:
            zone_counts["baseline"] += 1
        else:
            zone_counts["mid"] += 1

    total_positions = sum(zone_counts.values())
    zone_time_pct = None
    if total_positions:
        zone_time_pct = {
            k: round(v / total_positions * 100, 1) for k, v in zone_counts.items()
        }

    speed_samples_m_s = [s * meters_per_unit for s in speed_samples]

    return {
        "distance_m": _safe_round(total_dist_units * meters_per_unit, 2),
        "speed_stats": _stat_summary(speed_samples_m_s, decimals=2),
        "zone_time_pct": zone_time_pct,
    }


def _compute_ball_stats(ball_ts: list, events: list) -> dict:
    speeds = []
    accels = []

    for row in ball_ts:
        if not isinstance(row, dict):
            continue
        s = row.get("speed_m_s")
        a = row.get("accel_m_s2")
        if isinstance(s, (int, float)):
            speeds.append(float(s))
        if isinstance(a, (int, float)):
            accels.append(float(a))

    hit_deltas = []
    for e in events:
        if not isinstance(e, dict) or e.get("event_type") != "hit":
            continue
        before = e.get("speed_before_m_s")
        after = e.get("speed_after_m_s")
        delta = None
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            delta = after - before
        hit_deltas.append({
            "t": e.get("timestamp_sec"),
            "before": before,
            "after": after,
            "delta": delta,
        })

    accel_stats = None
    if accels:
        accel_stats = {
            "mean": _safe_round(float(np.mean(accels)), 2),
            "p95_abs": _safe_round(float(np.percentile([abs(a) for a in accels], 95)), 2),
        }

    return {
        "speed_stats": _stat_summary(speeds, decimals=2),
        "accel_stats": accel_stats,
        "speed_samples_m_s": [_safe_round(v, 3) for v in speeds],
        "hit_speed_deltas": hit_deltas,
    }


def _compute_event_timeline(events: list) -> list[dict]:
    timeline = []
    for e in events:
        if not isinstance(e, dict):
            continue
        timeline.append({
            "t": e.get("timestamp_sec"),
            "type": e.get("event_type"),
            "side": e.get("court_side"),
            "in_out": e.get("in_out"),
            "speed_before_m_s": e.get("speed_before_m_s"),
            "speed_after_m_s": e.get("speed_after_m_s"),
            "direction_change_deg": e.get("direction_change_deg"),
            "player": e.get("player"),
        })
    return timeline


def write_analysis_bundle(run_dir: Path, analysis: dict | None) -> None:
    if analysis is None:
        return

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    try:
        (analysis_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    except OSError:
        return

    files_to_copy = [
        "stats.json",
        "events.json",
        "points.json",
        "coaching_cards.json",
        "run.json",
    ]

    for rel in files_to_copy:
        src = run_dir / rel
        if src.exists():
            try:
                shutil.copyfile(src, analysis_dir / src.name)
            except OSError:
                pass

    timeseries_src = run_dir / "timeseries"
    if timeseries_src.is_dir():
        timeseries_dst = analysis_dir / "timeseries"
        timeseries_dst.mkdir(exist_ok=True)
        for ts_file in timeseries_src.glob("*.json"):
            try:
                shutil.copyfile(ts_file, timeseries_dst / ts_file.name)
            except OSError:
                pass

    visuals_src = run_dir / "visuals"
    if visuals_src.is_dir():
        visuals_dst = analysis_dir / "visuals"
        visuals_dst.mkdir(exist_ok=True)
        for vis_file in visuals_src.glob("*.json"):
            try:
                shutil.copyfile(vis_file, visuals_dst / vis_file.name)
            except OSError:
                pass

    logic_src = Path(__file__).resolve()
    if logic_src.exists():
        try:
            shutil.copyfile(logic_src, analysis_dir / "analysis_logic.py")
        except OSError:
            pass

    readme_path = analysis_dir / "README.txt"
    if not readme_path.exists():
        readme_path.write_text(
            "Analysis bundle contents:\\n"
            "- analysis.json: consolidated chart-ready analysis\\n"
            "- stats.json/events.json/points.json/coaching_cards.json/run.json\\n"
            "- timeseries/: ball and player time series\\n"
            "- visuals/: precomputed heatmaps and histograms\\n"
            "- analysis_logic.py: backend analysis builder\\n"
            "Note: frames.jsonl is intentionally omitted to keep the bundle lightweight.\\n"
        )
