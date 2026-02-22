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


def rebuild_analytics(run_dir: Path) -> dict | None:
    """Rebuild AnalyticsData from shots.json + points.json when analytics.json has zeros."""
    shots = _load_json(run_dir / "shots.json") or []
    points = _load_json(run_dir / "points.json") or []
    if not shots and not points:
        return None

    _enrich_points_with_last_owner(points, shots)

    a_shots = [s for s in shots if s.get("owner") == "player_a"]
    b_shots = [s for s in shots if s.get("owner") == "player_b"]

    player_a = _build_player_analytics("player_a", a_shots, points)
    player_b = _build_player_analytics("player_b", b_shots, points)

    rally_dist = _build_rally_distribution(points)
    rally_lengths = [p.get("rally_hit_count", 0) for p in points if isinstance(p, dict)]
    rally_avg = sum(rally_lengths) / len(rally_lengths) if rally_lengths else 0.0

    momentum = _build_momentum(points)
    match_flow_data = _build_match_flow_points(points)
    patterns = _build_shot_patterns(a_shots, b_shots)

    return {
        "player_a": player_a,
        "player_b": player_b,
        "rally_length_distribution": rally_dist,
        "rally_length_avg": round(rally_avg, 1),
        "total_points": len(points),
        "total_shots": len(shots),
        "momentum_data": momentum,
        "match_flow": match_flow_data,
        "shot_pattern_dominance": patterns,
    }


def _enrich_points_with_last_owner(points: list, shots: list) -> None:
    """Fill in last_hit_owner from shots data when it's missing in points."""
    sorted_shots = sorted(shots, key=lambda s: s.get("timestamp_sec", 0))
    for pt in points:
        if not isinstance(pt, dict):
            continue
        if pt.get("last_hit_owner"):
            continue
        start = pt.get("start_sec", 0)
        end = pt.get("end_sec", 0)
        last_owner = None
        for s in sorted_shots:
            ts = s.get("timestamp_sec", 0)
            if start <= ts <= end + 0.5:
                last_owner = s.get("owner")
        if last_owner:
            pt["last_hit_owner"] = last_owner


def rebuild_player_cards(run_dir: Path, analytics: dict | None = None) -> tuple[dict | None, dict | None]:
    """Rebuild player_a_card.json and player_b_card.json from analytics data."""
    if analytics is None:
        analytics = _load_json(run_dir / "analytics.json")
    if not analytics:
        return None, None

    pa = analytics.get("player_a", {})
    pb = analytics.get("player_b", {})
    patterns = analytics.get("shot_pattern_dominance", {})

    card_a = _build_player_card("player_a", pa, patterns.get("player_a", []))
    card_b = _build_player_card("player_b", pb, patterns.get("player_b", []))
    return card_a, card_b


def rebuild_match_flow(run_dir: Path) -> dict | None:
    """Rebuild match_flow.json from points.json + shots.json."""
    points = _load_json(run_dir / "points.json") or []
    shots = _load_json(run_dir / "shots.json") or []
    if not points:
        return None
    _enrich_points_with_last_owner(points, shots)

    insights = []
    rally_lengths = [p.get("rally_hit_count", 0) for p in points if isinstance(p, dict)]

    winners = []
    for p in points:
        er = p.get("end_reason", "")
        last_owner = p.get("last_hit_owner") or p.get("serve_player")
        if er in ("OUT", "NET", "NET_FAULT", "OUT_LONG", "OUT_WIDE"):
            winner = "player_b" if last_owner == "player_a" else "player_a"
        else:
            winner = last_owner or "player_a"
        winners.append(winner)

    max_streak = 0
    cur_streak = 0
    streak_player = None
    streak_start_idx = 0
    for i, w in enumerate(winners):
        if w == streak_player:
            cur_streak += 1
        else:
            if cur_streak >= 3:
                ts_start = points[streak_start_idx].get("start_sec")
                ts_end = points[i - 1].get("end_sec")
                player_label = "Player A" if streak_player == "player_a" else "Player B"
                insights.append({
                    "description": f"{player_label} won {cur_streak} consecutive points — a momentum surge.",
                    "timestamp_range": [ts_start, ts_end] if ts_start and ts_end else None,
                })
            streak_player = w
            cur_streak = 1
            streak_start_idx = i
        max_streak = max(max_streak, cur_streak)
    if cur_streak >= 3:
        ts_start = points[streak_start_idx].get("start_sec")
        ts_end = points[-1].get("end_sec")
        player_label = "Player A" if streak_player == "player_a" else "Player B"
        insights.append({
            "description": f"{player_label} won {cur_streak} consecutive points — a momentum surge.",
            "timestamp_range": [ts_start, ts_end] if ts_start and ts_end else None,
        })

    if len(rally_lengths) >= 4:
        mid = len(rally_lengths) // 2
        first_avg = sum(rally_lengths[:mid]) / mid if mid else 0
        second_avg = sum(rally_lengths[mid:]) / (len(rally_lengths) - mid) if (len(rally_lengths) - mid) else 0
        if abs(second_avg - first_avg) > 2:
            direction = "increased" if second_avg > first_avg else "decreased"
            insights.append({
                "description": f"Rally length {direction} from {first_avg:.1f} shots (first half) to {second_avg:.1f} shots (second half).",
                "timestamp_range": None,
            })

    overall_avg = sum(rally_lengths) / len(rally_lengths) if rally_lengths else 0
    if overall_avg > 8:
        insights.append({
            "description": f"Long rallies dominate (avg {overall_avg:.1f} shots) — baseline-heavy match.",
            "timestamp_range": None,
        })
    elif overall_avg < 4 and rally_lengths:
        insights.append({
            "description": f"Short rallies (avg {overall_avg:.1f} shots) — aggressive serve-and-volley or quick points.",
            "timestamp_range": None,
        })

    return {"insights": insights}


def _build_player_analytics(label: str, shots: list, points: list) -> dict:
    total = len(shots)
    type_counts: dict[str, int] = {}
    dir_counts: dict[str, dict[str, int]] = {}
    speeds: list[float] = []
    positions: list[tuple[float, float]] = []

    for s in shots:
        st = s.get("shot_type", "unknown")
        type_counts[st] = type_counts.get(st, 0) + 1

        d = s.get("ball_direction_label", "unknown")
        if st not in dir_counts:
            dir_counts[st] = {}
        dir_counts[st][d] = dir_counts[st].get(d, 0) + 1

        spd = s.get("speed_m_s")
        if isinstance(spd, (int, float)) and spd > 0:
            speeds.append(float(spd))

        xy = s.get("ball_court_xy")
        if xy and len(xy) == 2 and xy[0] is not None and xy[1] is not None:
            positions.append((float(xy[0]), float(xy[1])))

    type_pcts = {k: round(v / total * 100, 1) for k, v in type_counts.items()} if total else {}
    dir_pcts: dict[str, dict[str, float]] = {}
    for st, dirs in dir_counts.items():
        st_total = sum(dirs.values())
        dir_pcts[st] = {d: round(c / st_total * 100, 1) for d, c in dirs.items()} if st_total else {}

    error_by_shot: dict[str, int] = {}
    error_by_rally: dict[str, int] = {}
    total_by_rally: dict[str, int] = {}
    serve_zones: dict[str, int] = {}
    serve_wins: dict[str, int] = {}
    serve_total: dict[str, int] = {}
    double_faults = 0
    first_serve_attempts = 0
    first_serve_in = 0
    points_won = 0
    points_lost = 0

    for pt in points:
        if not isinstance(pt, dict):
            continue
        rally = pt.get("rally_hit_count", 0)
        bucket = "1-3" if rally <= 3 else "4-6" if rally <= 6 else "7-9" if rally <= 9 else "10+"
        total_by_rally[bucket] = total_by_rally.get(bucket, 0) + 1

        er = pt.get("end_reason", "")
        last_owner = pt.get("last_hit_owner") or pt.get("serve_player")
        is_error = er in ("OUT", "NET", "NET_FAULT", "OUT_LONG", "OUT_WIDE")
        loser = last_owner if is_error else None
        winner = None
        if is_error:
            winner = "player_b" if last_owner == "player_a" else "player_a"
        else:
            winner = last_owner or "player_a"

        if winner == label:
            points_won += 1
        elif loser == label or (winner and winner != label):
            points_lost += 1

        if loser == label:
            error_by_rally[bucket] = error_by_rally.get(bucket, 0) + 1

        sp = pt.get("serve_player")
        if sp == label:
            zone = pt.get("serve_zone", "unknown")
            serve_zones[zone] = serve_zones.get(zone, 0) + 1
            serve_total[zone] = serve_total.get(zone, 0) + 1
            if winner == label:
                serve_wins[zone] = serve_wins.get(zone, 0) + 1

            fault = pt.get("serve_fault_type")
            if fault == "double":
                double_faults += 1

    error_rate_by_shot: dict[str, float] = {}
    for s in shots:
        st = s.get("shot_type", "unknown")
        ts = s.get("timestamp_sec", 0)
        for pt in points:
            if not isinstance(pt, dict):
                continue
            end_sec = pt.get("end_sec", 0)
            er = pt.get("end_reason", "")
            last_owner = pt.get("last_hit_owner") or pt.get("serve_player")
            if (abs(ts - end_sec) < 1.5 and er in ("OUT", "NET", "NET_FAULT", "OUT_LONG", "OUT_WIDE")
                    and last_owner == label):
                error_by_shot[st] = error_by_shot.get(st, 0) + 1
                break

    for st, errs in error_by_shot.items():
        tc = type_counts.get(st, 0)
        if tc > 0:
            error_rate_by_shot[st] = round(errs / tc * 100, 1)

    error_rate_by_rally: dict[str, float] = {}
    for bucket, errs in error_by_rally.items():
        t = total_by_rally.get(bucket, 0)
        if t > 0:
            error_rate_by_rally[bucket] = round(errs / t * 100, 1)

    serve_zone_win_rate: dict[str, float] = {}
    for zone, t in serve_total.items():
        if t > 0:
            serve_zone_win_rate[zone] = round(serve_wins.get(zone, 0) / t * 100, 1)

    avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 0.0
    total_dist = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        total_dist += (dx ** 2 + dy ** 2) ** 0.5

    cx = round(sum(p[0] for p in positions) / len(positions), 1) if positions else 0.0
    cy = round(sum(p[1] for p in positions) / len(positions), 1) if positions else 0.0

    return {
        "label": label,
        "total_shots": total,
        "shot_type_counts": type_counts,
        "shot_type_pcts": type_pcts,
        "shot_direction_counts": dir_counts,
        "shot_direction_pcts": dir_pcts,
        "error_by_shot_type": error_by_shot,
        "error_rate_by_shot_type": error_rate_by_shot,
        "error_by_rally_length": error_by_rally,
        "error_rate_by_rally_length": error_rate_by_rally,
        "avg_shot_speed_m_s": avg_speed,
        "total_distance_covered": round(total_dist, 1),
        "center_of_gravity": [cx, cy],
        "first_serve_pct": round(first_serve_in / first_serve_attempts * 100, 1) if first_serve_attempts else 0.0,
        "double_fault_count": double_faults,
        "serve_zone_win_rate": serve_zone_win_rate,
        "serve_placement_counts": serve_zones,
        "points_won": points_won,
        "points_lost": points_lost,
    }


def _build_rally_distribution(points: list) -> dict[str, int]:
    dist: dict[str, int] = {}
    for p in points:
        if not isinstance(p, dict):
            continue
        r = p.get("rally_hit_count", 0)
        bucket = "1-3" if r <= 3 else "4-6" if r <= 6 else "7-9" if r <= 9 else "10+"
        dist[bucket] = dist.get(bucket, 0) + 1
    return dist


def _build_momentum(points: list) -> list[dict]:
    momentum = []
    a_m = 0
    b_m = 0
    for p in points:
        if not isinstance(p, dict):
            continue
        er = p.get("end_reason", "")
        last_owner = p.get("last_hit_owner") or p.get("serve_player")
        is_error = er in ("OUT", "NET", "NET_FAULT", "OUT_LONG", "OUT_WIDE", "DOUBLE_BOUNCE")
        if is_error:
            winner = "player_b" if last_owner == "player_a" else "player_a"
        else:
            winner = last_owner or "player_a"

        if winner == "player_a":
            a_m += 1
            b_m = max(0, b_m - 1)
        else:
            b_m += 1
            a_m = max(0, a_m - 1)

        momentum.append({
            "point_idx": p.get("point_idx", len(momentum)),
            "timestamp_sec": p.get("start_sec", 0),
            "winner": winner,
            "a_momentum": a_m,
            "b_momentum": b_m,
            "rally_length": p.get("rally_hit_count", 0),
        })
    return momentum


def _build_match_flow_points(points: list) -> list[dict]:
    flow = []
    for p in points:
        if not isinstance(p, dict):
            continue
        start = p.get("start_sec", 0)
        end = p.get("end_sec", 0)
        flow.append({
            "point_idx": p.get("point_idx", len(flow)),
            "timestamp_sec": start,
            "rally_length": p.get("rally_hit_count", 0),
            "end_reason": p.get("end_reason", "UNKNOWN"),
            "duration_sec": round(end - start, 1) if isinstance(end, (int, float)) and isinstance(start, (int, float)) else 0,
        })
    return flow


def _build_shot_patterns(a_shots: list, b_shots: list) -> dict[str, list[dict]]:
    def _patterns_for(shots: list) -> list[dict]:
        pat_counts: dict[str, dict] = {}
        for s in shots:
            st = s.get("shot_type", "unknown")
            d = s.get("ball_direction_label", "unknown")
            key = f"{st}_{d}"
            if key not in pat_counts:
                pat_counts[key] = {"pattern": key, "shot_type": st, "direction": d, "count": 0}
            pat_counts[key]["count"] += 1
        total = len(shots) or 1
        result = sorted(pat_counts.values(), key=lambda x: -x["count"])
        for p in result:
            p["pct"] = round(p["count"] / total * 100, 1)
        return result[:5]
    return {"player_a": _patterns_for(a_shots), "player_b": _patterns_for(b_shots)}


def _build_player_card(label: str, pa: dict, patterns: list) -> dict:
    total = pa.get("total_shots", 0)
    type_counts = pa.get("shot_type_counts", {})
    type_pcts = pa.get("shot_type_pcts", {})
    dir_pcts = pa.get("shot_direction_pcts", {})
    error_rates = pa.get("error_rate_by_shot_type", {})
    error_by_rally = pa.get("error_rate_by_rally_length", {})
    name = "Player A" if label == "player_a" else "Player B"

    tendencies = []
    if total > 0:
        dominant = max(type_counts, key=type_counts.get) if type_counts else None
        if dominant:
            tendencies.append(f"{name} primarily uses {dominant}s ({type_pcts.get(dominant, 0)}% of all shots).")
        for st, dirs in dir_pcts.items():
            if not dirs:
                continue
            top_dir = max(dirs, key=dirs.get)
            if dirs[top_dir] >= 55:
                tendencies.append(f"{name}'s {st} goes {top_dir.replace('_', ' ')} {dirs[top_dir]}% of the time.")
        if patterns:
            top = patterns[0]
            tendencies.append(f"{name}'s most frequent pattern is {top['shot_type']} {top['direction'].replace('_', ' ')} ({top['pct']}% of shots).")
    else:
        tendencies.append(f"Insufficient shot data to determine {name}'s tendencies.")

    exploit_plan = ""
    weaknesses: list[dict] = []

    for st, rate in error_rates.items():
        count = pa.get("error_by_shot_type", {}).get(st, 0)
        tc = type_counts.get(st, 0)
        if rate >= 30 and count >= 2:
            exploit_plan = f"Attack {name}'s {st} — {rate:.0f}% error rate ({count} errors this match). Force them to that wing in key moments."
            weaknesses.append({
                "description": f"{name}'s {st} breaks down under pressure — {rate:.0f}% error rate ({count} errors from {tc} attempts). Target their {st} side in critical points.",
                "data_point": f"{count} errors / {tc} {st}s ({rate:.0f}% error rate)",
                "points_cost": count,
                "severity": round(min(rate / 100, 1.0), 3),
            })

    for st, dirs in dir_pcts.items():
        if not dirs:
            continue
        top_dir = max(dirs, key=dirs.get)
        if dirs[top_dir] >= 75:
            pct = dirs[top_dir]
            weaknesses.append({
                "description": f"{name} telegraphs their {st} — {pct:.0f}% go {top_dir.replace('_', ' ')}. Opponent can position early and attack the predictable pattern.",
                "data_point": f"{pct:.0f}% of {st}s go {top_dir.replace('_', ' ')}",
                "points_cost": 0,
                "severity": round(min(pct / 100, 1.0), 3),
            })

    for bucket, rate in error_by_rally.items():
        errs = pa.get("error_by_rally_length", {}).get(bucket, 0)
        if rate >= 50 and errs >= 2:
            weaknesses.append({
                "description": f"{name}'s error rate spikes in {bucket}-shot rallies ({rate:.0f}%). Extend rallies to exploit this.",
                "data_point": f"{rate:.0f}% error rate in {bucket}-shot rallies",
                "points_cost": errs,
                "severity": round(min(rate / 100, 1.0), 3),
            })

    weaknesses.sort(key=lambda w: -w["severity"])

    dist_parts = ", ".join(f"{k}: {v}" for k, v in sorted(type_counts.items(), key=lambda x: -x[1]))
    shot_dist_summary = f"Shot breakdown: {dist_parts} (total: {total})" if total else ""
    cov = pa.get("total_distance_covered", 0)
    cog = pa.get("center_of_gravity", [0, 0])
    cov_summary = f"{name} covered {int(cov)} court units. Average position: ({int(cog[0])}, {int(cog[1])})." if cov else ""

    serve_summary = ""
    szwr = pa.get("serve_zone_win_rate", {})
    spc = pa.get("serve_placement_counts", {})
    if spc:
        parts = [f"{z}: {c}" for z, c in spc.items()]
        serve_summary = f"Serve zones: {', '.join(parts)}."
        if szwr:
            best = max(szwr, key=szwr.get) if szwr else None
            if best:
                serve_summary += f" Best zone: {best.replace('_', ' ')} ({szwr[best]}% win rate)."

    return {
        "card": {
            "label": label,
            "exploit_plan": exploit_plan,
            "tendencies": tendencies,
            "serve_summary": serve_summary,
            "shot_distribution_summary": shot_dist_summary,
            "coverage_summary": cov_summary,
        },
        "weaknesses": {
            "label": label,
            "weaknesses": weaknesses,
        },
    }


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
