"""
TennisIQ pipeline runner.

Sends each 10-second video segment to the Modal GPU function (tennisiq-court)
for inference. Accumulates results across all segments into merged output files.
Reports progress back to the FastAPI backend via /status/update.
"""
import os
import json
import math
import logging
import shutil
import tempfile
import threading
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", os.path.join(os.path.dirname(__file__), "..", "outputs"))
SEGMENT_DURATION = 10.0

JSON_MERGE_LIST_FILES = {"events.json", "points.json", "coaching_cards.json"}
JSON_MERGE_DICT_FILES = {"stats.json", "run.json"}


def _get_video_info(video_path: str) -> dict:
    """Extract FPS and duration from a video file via OpenCV."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"fps": fps, "duration": frames / fps if fps > 0 else 0}


def _download_youtube(url: str, output_dir: str) -> str:
    """Download a YouTube video to MP4 using yt-dlp Python API. Returns file path."""
    import yt_dlp

    outtmpl = os.path.join(output_dir, "video.%(ext)s")
    opts = {
        "format": "mp4/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    for fn in os.listdir(output_dir):
        if fn.startswith("video.") and fn.split(".")[-1] in {"mp4", "mkv", "webm", "mov"}:
            return os.path.join(output_dir, fn)
    raise RuntimeError("yt-dlp did not produce a video file")


def _resolve_video(footage_url: str) -> str:
    """Given a footage_url (YouTube URL or file:// path), return a local file path."""
    if footage_url.startswith("file://"):
        return footage_url[7:]
    tmpdir = tempfile.mkdtemp(prefix="tennisiq_yt_")
    return _download_youtube(footage_url, tmpdir)


def _post_status(backend_url: str, **kwargs):
    try:
        requests.post(f"{backend_url}/status/update", json=kwargs, timeout=10)
    except Exception as e:
        logger.warning(f"Status post failed: {e}")


def _run_segment_on_modal(
    video_bytes: bytes,
    fps: float,
    start_sec: float,
    end_sec: float,
) -> dict:
    """Run a single 10-second segment on Modal T4 GPU, with local CPU fallback."""
    try:
        import modal
        fn = modal.Function.from_name("tennisiq-court", "run_court_and_ball")
        return fn.remote(
            video_bytes=video_bytes,
            fps=fps,
            start_sec=start_sec,
            end_sec=end_sec,
        )
    except Exception as modal_err:
        logger.warning(f"Modal unavailable ({modal_err}), running locally on CPU...")
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.environ.setdefault("TENNISIQ_CHECKPOINT_ROOT",
                              os.path.join(project_root, "checkpoints"))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from tennisiq.modal_court import run_court_and_ball
        return run_court_and_ball.local(
            video_bytes=video_bytes,
            fps=fps,
            start_sec=start_sec,
            end_sec=end_sec,
        )


# ── Segment result accumulator ────────────────────────────────────────────────

class ResultAccumulator:
    """Collects results from multiple segments and merges them into unified outputs."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.all_events: list = []
        self.all_points: list = []
        self.all_coaching_cards: list = []
        self.all_shots: list = []
        self.all_frames_jsonl: list[str] = []
        self.overlay_parts: list[bytes] = []
        self.clip_counter = 0
        self.latest_stats: dict = {}
        self.latest_run: dict = {}
        self.latest_analytics: dict = {}
        self.latest_player_a_card: dict = {}
        self.latest_player_b_card: dict = {}
        self.latest_match_flow: dict = {}
        self.heatmap_accum: dict[str, list] = {}
        self.timeseries_accum: dict[str, list] = {}

    def ingest_segment(self, seg_idx: int, result: dict):
        """Process one segment's output_files + video_files into accumulated state."""
        output_files = result.get("output_files", {})
        video_files = result.get("video_files", {})

        for rel_path, content in output_files.items():
            fname = Path(rel_path).name

            if fname == "events.json":
                try:
                    items = json.loads(content)
                    if isinstance(items, list):
                        for e in items:
                            e["_segment"] = seg_idx
                        self.all_events.extend(items)
                except (json.JSONDecodeError, TypeError):
                    pass

            elif fname == "points.json":
                try:
                    items = json.loads(content)
                    if isinstance(items, list):
                        for p in items:
                            p["point_idx"] = len(self.all_points) + items.index(p)
                            p["_segment"] = seg_idx
                        self.all_points.extend(items)
                except (json.JSONDecodeError, TypeError):
                    pass

            elif fname == "coaching_cards.json":
                try:
                    items = json.loads(content)
                    if isinstance(items, list):
                        self.all_coaching_cards.extend(items)
                except (json.JSONDecodeError, TypeError):
                    pass

            elif fname == "shots.json":
                try:
                    items = json.loads(content)
                    if isinstance(items, list):
                        for s in items:
                            s["_segment"] = seg_idx
                        self.all_shots.extend(items)
                except (json.JSONDecodeError, TypeError):
                    pass

            elif fname == "analytics.json":
                try:
                    self.latest_analytics = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    pass

            elif fname == "player_a_card.json":
                try:
                    self.latest_player_a_card = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    pass

            elif fname == "player_b_card.json":
                try:
                    self.latest_player_b_card = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    pass

            elif fname == "match_flow.json":
                try:
                    self.latest_match_flow = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    pass

            elif fname == "frames.jsonl":
                self.all_frames_jsonl.append(content)

            elif fname == "stats.json":
                try:
                    self.latest_stats = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    pass

            elif fname == "run.json":
                try:
                    self.latest_run = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    pass

            elif rel_path.startswith("visuals/") or rel_path.startswith("visuals\\"):
                self._accumulate_visual(rel_path, content)

            elif rel_path.startswith("timeseries/") or rel_path.startswith("timeseries\\"):
                ts_name = Path(rel_path).name
                try:
                    items = json.loads(content)
                    if isinstance(items, list):
                        self.timeseries_accum.setdefault(ts_name, []).extend(items)
                except (json.JSONDecodeError, TypeError):
                    pass

        for rel_path, vbytes in video_files.items():
            if "overlay" in rel_path:
                self.overlay_parts.append(vbytes)
            elif "clips/" in rel_path or "clips\\" in rel_path:
                clip_name = f"point_{self.clip_counter}.mp4"
                self.clip_counter += 1
                fp = self.out_dir / "clips" / clip_name
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_bytes(vbytes)

    def _accumulate_visual(self, rel_path: str, content: str):
        """Merge heatmap/visual JSON data across segments."""
        fname = Path(rel_path).name
        if not fname.endswith(".json"):
            return

        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return

        if fname == "player_coverage.json":
            existing = self.heatmap_accum.get("player_coverage", {})
            if not existing:
                self.heatmap_accum["player_coverage"] = data
            else:
                for key in ("player_a", "player_b"):
                    if key in data:
                        existing.setdefault(key, []).extend(data[key])

        elif fname.endswith("_heatmap.json"):
            key = fname.replace(".json", "")
            if "grid" in data:
                self.heatmap_accum[key] = data
            elif isinstance(data, dict):
                self.heatmap_accum.setdefault(key, {}).update(data)

        elif fname == "serve_placement.json":
            existing = self.heatmap_accum.get("serve_placement")
            if not existing:
                self.heatmap_accum["serve_placement"] = data
            else:
                if "serves" in data and "serves" in existing:
                    existing["serves"].extend(data["serves"])

        else:
            self.heatmap_accum[fname.replace(".json", "")] = data

    def write_merged(self, fps: float, duration: float, n_segments: int, succeeded: int):
        """Write all accumulated results to disk as merged output files."""
        self.out_dir.mkdir(parents=True, exist_ok=True)

        _write_json(self.out_dir / "events.json", self.all_events)
        _reindex_points(self.all_points)
        _write_json(self.out_dir / "points.json", self.all_points)
        _reindex_cards(self.all_coaching_cards, self.all_points)
        _write_json(self.out_dir / "coaching_cards.json", self.all_coaching_cards)

        _write_json(self.out_dir / "shots.json", self.all_shots)

        stats = self._build_merged_stats(fps, duration, n_segments, succeeded)
        _write_json(self.out_dir / "stats.json", stats)

        if self.latest_analytics:
            _write_json(self.out_dir / "analytics.json", self.latest_analytics)
        if self.latest_player_a_card:
            _write_json(self.out_dir / "player_a_card.json", self.latest_player_a_card)
        if self.latest_player_b_card:
            _write_json(self.out_dir / "player_b_card.json", self.latest_player_b_card)
        if self.latest_match_flow:
            _write_json(self.out_dir / "match_flow.json", self.latest_match_flow)

        run_info = self.latest_run or {}
        run_info["segments_total"] = n_segments
        run_info["segments_succeeded"] = succeeded
        run_info["fps"] = fps
        run_info["duration_sec"] = round(duration, 2)
        _write_json(self.out_dir / "run.json", run_info)

        if self.all_frames_jsonl:
            (self.out_dir / "frames.jsonl").write_text(
                "\n".join(self.all_frames_jsonl), encoding="utf-8"
            )

        vis_dir = self.out_dir / "visuals"
        vis_dir.mkdir(parents=True, exist_ok=True)
        for key, data in self.heatmap_accum.items():
            _write_json(vis_dir / f"{key}.json", data)

        ts_dir = self.out_dir / "timeseries"
        ts_dir.mkdir(parents=True, exist_ok=True)
        for key, data in self.timeseries_accum.items():
            _write_json(ts_dir / key, data)

        if self.overlay_parts:
            self._write_merged_overlay(vis_dir / "overlay.mp4")

    def _write_merged_overlay(self, output_path: Path):
        """Concatenate segment overlay videos into a single overlay."""
        if len(self.overlay_parts) == 1:
            output_path.write_bytes(self.overlay_parts[0])
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmpdir = tempfile.mkdtemp(prefix="tennisiq_overlay_")
        list_path = os.path.join(tmpdir, "filelist.txt")
        part_paths = []

        for i, vbytes in enumerate(self.overlay_parts):
            part_path = os.path.join(tmpdir, f"part_{i:04d}.mp4")
            with open(part_path, "wb") as f:
                f.write(vbytes)
            part_paths.append(part_path)

        try:
            import subprocess
            with open(list_path, "w") as f:
                for p in part_paths:
                    f.write(f"file '{p}'\n")

            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", list_path, "-c", "copy", str(output_path)],
                capture_output=True, timeout=120,
            )
        except Exception as e:
            logger.warning(f"ffmpeg concat failed ({e}), writing first overlay only")
            output_path.write_bytes(self.overlay_parts[0])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _build_merged_stats(self, fps, duration, n_segments, succeeded):
        """Build merged stats from accumulated data."""
        n_bounces = sum(1 for e in self.all_events if e.get("event_type") == "bounce")
        n_hits = sum(1 for e in self.all_events if e.get("event_type") == "hit")
        avg_rally = 0
        avg_conf = 0
        if self.all_points:
            rallies = [p.get("rally_hit_count", 0) for p in self.all_points]
            confs = [p.get("confidence", 0) for p in self.all_points]
            avg_rally = sum(rallies) / len(rallies)
            avg_conf = sum(confs) / len(confs)

        return {
            "fps": fps,
            "duration_sec": round(duration, 2),
            "segments": {"total": n_segments, "succeeded": succeeded},
            "events": {
                "total": len(self.all_events),
                "bounces": n_bounces,
                "hits": n_hits,
            },
            "points": {
                "total": len(self.all_points),
                "avg_rally_hits": round(avg_rally, 1),
                "avg_confidence": round(avg_conf, 3),
            },
            "insights": self.latest_stats.get("insights", []),
        }


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _reindex_points(points: list):
    for i, p in enumerate(points):
        p["point_idx"] = i


def _reindex_cards(cards: list, points: list):
    for i, c in enumerate(cards):
        c["point_idx"] = i


# ── Post-merge analytics recomputation ────────────────────────────────────────

def _compute_post_merge_analytics(out_dir: Path, fps: float):
    """
    Recompute shot detection, match analytics, and coaching intelligence
    from merged output files. Runs locally after segment merging so analytics
    are available even when the Modal cloud function uses an older code version.
    """
    try:
        from tennisiq.cv.ball.inference import BallPhysics
        from tennisiq.cv.players.inference import PlayerDetection, FramePlayers
        from tennisiq.analytics.events import TennisEvent
        from tennisiq.analytics.points import TennisPoint
        from tennisiq.analytics.shots import classify_shot_direction, ShotEvent
        from tennisiq.analytics.match_analytics import compute_match_analytics, analytics_to_dict
        from tennisiq.analytics.coaching_intelligence import generate_coaching_intelligence, coaching_to_dict
    except ImportError as e:
        logger.warning(f"Cannot import analytics modules for post-merge computation: {e}")
        return

    frames_path = out_dir / "frames.jsonl"
    events_path = out_dir / "events.json"
    points_path = out_dir / "points.json"

    if not frames_path.exists():
        logger.warning("No frames.jsonl for post-merge analytics")
        return

    logger.info("Running post-merge analytics recomputation...")

    # Reconstruct ball_physics from frames.jsonl
    ball_physics: list[BallPhysics] = []
    player_results: list[FramePlayers] = []

    for line in frames_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            f = json.loads(line)
        except json.JSONDecodeError:
            continue

        ball = f.get("ball", {})
        pxy = ball.get("pixel_xy")
        cxy = ball.get("court_xy")
        ball_physics.append(BallPhysics(
            frame_idx=f.get("frame_idx", len(ball_physics)),
            pixel_xy=(pxy[0], pxy[1]) if pxy else (None, None),
            court_xy=(cxy[0], cxy[1]) if cxy else (None, None),
            speed_m_per_s=ball.get("speed_m_s"),
            accel_m_per_s2=ball.get("accel_m_s2"),
        ))

        players = f.get("players", {})
        pa_data = players.get("player_a")
        pb_data = players.get("player_b")
        pa = None
        pb = None
        if pa_data and isinstance(pa_data, dict):
            fc = pa_data.get("foot_court")
            if fc:
                pa = PlayerDetection(
                    bbox=(0, 0, 0, 0),
                    confidence=pa_data.get("confidence", 0.5),
                    track_id=None,
                    foot_pixel=(0, 0),
                    foot_court=(fc[0], fc[1]),
                    inside_court=True,
                )
        if pb_data and isinstance(pb_data, dict):
            fc = pb_data.get("foot_court")
            if fc:
                pb = PlayerDetection(
                    bbox=(0, 0, 0, 0),
                    confidence=pb_data.get("confidence", 0.5),
                    track_id=None,
                    foot_pixel=(0, 0),
                    foot_court=(fc[0], fc[1]),
                    inside_court=True,
                )
        player_results.append(FramePlayers(
            frame_idx=f.get("frame_idx", len(player_results)),
            all_detections=[d for d in [pa, pb] if d],
            player_a=pa,
            player_b=pb,
        ))

    if len(ball_physics) < 4:
        logger.warning(f"Too few frames ({len(ball_physics)}) for shot detection")
        return

    # Reconstruct events
    events_list: list[TennisEvent] = []
    if events_path.exists():
        try:
            raw_events = json.loads(events_path.read_text(encoding="utf-8"))
            for e in raw_events:
                events_list.append(TennisEvent(
                    event_type=e["event_type"],
                    frame_idx=e["frame_idx"],
                    timestamp_sec=e["timestamp_sec"],
                    court_xy=tuple(e["court_xy"]),
                    speed_before_m_s=e.get("speed_before_m_s"),
                    speed_after_m_s=e.get("speed_after_m_s"),
                    direction_change_deg=e.get("direction_change_deg"),
                    score=e.get("score", 0.5),
                    player=e.get("player"),
                    player_distance=e.get("player_distance"),
                    in_out=e.get("in_out"),
                    court_side=e.get("court_side"),
                ))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse events.json: {e}")

    # Reconstruct points
    points_list: list[TennisPoint] = []
    if points_path.exists():
        try:
            raw_points = json.loads(points_path.read_text(encoding="utf-8"))
            for p in raw_points:
                fb_xy = p.get("first_bounce_court_xy")
                pt_events = [ev for ev in events_list
                             if p["start_frame"] <= ev.frame_idx <= p["end_frame"]]
                points_list.append(TennisPoint(
                    point_idx=p["point_idx"],
                    start_frame=p["start_frame"],
                    end_frame=p["end_frame"],
                    start_sec=p["start_sec"],
                    end_sec=p["end_sec"],
                    serve_frame=p.get("serve_frame"),
                    serve_player=p.get("serve_player"),
                    first_bounce_frame=p.get("first_bounce_frame"),
                    first_bounce_court_xy=tuple(fb_xy) if fb_xy else None,
                    serve_zone=p.get("serve_zone"),
                    serve_fault_type=p.get("serve_fault_type"),
                    end_reason=p.get("end_reason", "BALL_LOST"),
                    rally_hit_count=p.get("rally_hit_count", 0),
                    bounce_count=p.get("bounce_count", 0),
                    bounce_frames=p.get("bounce_frames", []),
                    events=pt_events,
                    confidence=p.get("confidence", 0.5),
                ))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse points.json: {e}")

    # Phase 1: Contact detection from ball physics
    # Uses EMA-smoothed trajectory with angle reversal, speed gating, and cooldown.
    from tennisiq.analytics.shots import detect_contacts
    shot_events = detect_contacts(
        ball_physics=ball_physics,
        player_results=player_results,
        fps=fps,
        start_sec=0.0,
    )
    logger.info(f"Post-merge contact detection: {len(shot_events)} contacts")

    # Phase 2: Trajectory-based shot type classification
    from tennisiq.analytics.shot_classifier import classify_shot_type

    shot_directions: dict[int, str] = {}
    for i, shot in enumerate(shot_events):
        result = classify_shot_type(shot, i, shot_events, ball_physics)
        shot.shot_type = result.shot_type
        shot.shot_type_confidence = result.confidence
        shot_directions[shot.frame_idx] = classify_shot_direction(
            shot, shot.court_side or "near"
        )
    type_counts: dict[str, int] = {}
    for s in shot_events:
        type_counts[s.shot_type] = type_counts.get(s.shot_type, 0) + 1
    logger.info(f"Post-merge shot classification: {type_counts}")

    # Phase 3: Match analytics
    analytics = compute_match_analytics(
        shot_events=shot_events,
        shot_directions=shot_directions,
        points=points_list,
        events=events_list,
        ball_physics=ball_physics,
        player_results=player_results,
        fps=fps,
    )

    # Phase 4: Coaching intelligence
    coaching = generate_coaching_intelligence(
        analytics, points_list, shot_events, shot_directions
    )

    logger.info(f"Post-merge analytics complete: {analytics.total_shots} shots, "
                f"{analytics.total_points} points")

    # Write output files
    shots_data = []
    for s in shot_events:
        shots_data.append({
            "frame_idx": s.frame_idx,
            "timestamp_sec": s.timestamp_sec,
            "owner": s.owner,
            "ball_court_xy": list(s.ball_court_xy),
            "shot_type": s.shot_type,
            "shot_type_confidence": s.shot_type_confidence,
            "ball_direction_deg": s.ball_direction_deg,
            "ball_direction_label": shot_directions.get(s.frame_idx, "unknown"),
            "speed_m_s": s.speed_m_s,
            "court_side": s.court_side,
            "ownership_method": s.ownership_method,
        })
    _write_json(out_dir / "shots.json", shots_data)

    analytics_dict = analytics_to_dict(analytics)
    _write_json(out_dir / "analytics.json", analytics_dict)

    coaching_dict = coaching_to_dict(coaching)

    player_a_card = {
        "card": coaching_dict.get("player_a_card", {}),
        "weaknesses": coaching_dict.get("player_a_weaknesses", {}),
    }
    _write_json(out_dir / "player_a_card.json", player_a_card)

    player_b_card = {
        "card": coaching_dict.get("player_b_card", {}),
        "weaknesses": coaching_dict.get("player_b_weaknesses", {}),
    }
    _write_json(out_dir / "player_b_card.json", player_b_card)

    match_flow = {
        "insights": coaching_dict.get("match_flow_insights", []),
    }
    _write_json(out_dir / "match_flow.json", match_flow)

    # Also update coaching cards with enhanced data if available
    enhanced_cards = coaching_dict.get("coaching_cards", [])
    if enhanced_cards:
        _write_json(out_dir / "coaching_cards.json", enhanced_cards)

    logger.info(f"Post-merge analytics files written to {out_dir}")


# ── Main pipeline orchestrator ────────────────────────────────────────────────

def _run_pipeline(
    job_id: str,
    footage_url: str,
    config: dict,
    backend_url: str,
):
    """Orchestrate the full pipeline: resolve video -> segment -> Modal GPU -> merge."""
    try:
        _post_status(backend_url, job_id=job_id, stage="downloading",
                     description="Downloading and preparing video...", status="running")

        video_path = _resolve_video(footage_url)
        info = _get_video_info(video_path)
        fps = info["fps"]
        duration = info["duration"]

        if duration <= 0:
            _post_status(backend_url, job_id=job_id, stage="error",
                         description="Could not determine video duration.", status="error",
                         error_message="Video has zero duration or is unreadable.")
            return

        n_segments = max(1, math.ceil(duration / SEGMENT_DURATION))
        segments = []
        for i in range(n_segments):
            s = i * SEGMENT_DURATION
            e = min((i + 1) * SEGMENT_DURATION, duration)
            segments.append({"idx": i, "start_sec": s, "end_sec": e})

        _post_status(backend_url, job_id=job_id, stage="segmenting",
                     description=f"Video is {duration:.1f}s — splitting into {n_segments} segment(s) for GPU processing.",
                     status="running", segments=segments)

        out_dir = Path(os.path.abspath(OUTPUTS_DIR)) / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(video_path, "rb") as f:
            video_bytes = f.read()

        accum = ResultAccumulator(out_dir)
        succeeded = 0
        failed = 0
        segment_errors = []

        for seg in segments:
            seg_idx = seg["idx"]
            _post_status(
                backend_url, job_id=job_id,
                stage="inference",
                description=f"Running inference on segment {seg_idx + 1}/{n_segments} "
                            f"({seg['start_sec']:.0f}s–{seg['end_sec']:.0f}s) on Modal T4 GPU...",
                status="running",
            )

            try:
                result = _run_segment_on_modal(
                    video_bytes=video_bytes,
                    fps=fps,
                    start_sec=seg["start_sec"],
                    end_sec=seg["end_sec"],
                )

                accum.ingest_segment(seg_idx, result)
                succeeded += 1

                _post_status(
                    backend_url, job_id=job_id,
                    stage="inference",
                    description=f"Segment {seg_idx + 1}/{n_segments} complete "
                                f"({result.get('frames_processed', '?')} frames, "
                                f"{result.get('timing', {}).get('total', '?')}s on GPU).",
                    status="running",
                    segment_complete={"idx": seg_idx, "result_key": str(out_dir)},
                )

            except Exception as e:
                failed += 1
                err_msg = str(e)
                segment_errors.append(f"Segment {seg_idx + 1}: {err_msg}")
                logger.error(f"Segment {seg_idx} failed: {e}")
                _post_status(
                    backend_url, job_id=job_id,
                    stage="inference",
                    description=f"Segment {seg_idx + 1}/{n_segments} failed: {err_msg}",
                    status="running",
                )

        if succeeded > 0:
            _post_status(backend_url, job_id=job_id, stage="generating_outputs",
                         description="Merging segment results...", status="running")
            accum.write_merged(fps, duration, n_segments, succeeded)

            # Post-merge fallback recomputation:
            # Recompute only when merged analytics are missing/empty.
            # This preserves up-to-date Modal analytics (including pose-based
            # shot labels) while still fixing stale/older cloud outputs.
            analytics_ok = False
            analytics_path = out_dir / "analytics.json"
            if analytics_path.exists():
                try:
                    merged_analytics = json.loads(analytics_path.read_text(encoding="utf-8"))
                    total_shots = int(merged_analytics.get("total_shots", 0) or 0)
                    total_points = int(merged_analytics.get("total_points", 0) or 0)
                    analytics_ok = total_shots > 0 and total_points > 0
                except Exception:
                    analytics_ok = False

            if not analytics_ok:
                _post_status(backend_url, job_id=job_id, stage="match_analytics",
                             description="Computing match analytics & coaching intelligence...",
                             status="running")
                try:
                    _compute_post_merge_analytics(out_dir, fps)
                except Exception as e:
                    logger.error(f"Post-merge analytics failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

        # Copy raw video into outputs for the results page
        if footage_url.startswith("file://"):
            raw_src = footage_url[7:]
            raw_dst = out_dir / "raw_video.mp4"
            if not raw_dst.exists() and os.path.exists(raw_src):
                shutil.copy2(raw_src, str(raw_dst))

        # Determine final status
        if succeeded == 0:
            error_detail = f"All {n_segments} segment(s) failed on Modal GPU.\n" + "\n".join(segment_errors)
            _post_status(
                backend_url, job_id=job_id,
                stage="error",
                description="Pipeline failed — no segments completed successfully.",
                status="error",
                error_message=error_detail,
            )
        elif failed > 0:
            _post_status(
                backend_url, job_id=job_id,
                stage="complete",
                description=f"Pipeline complete with partial results ({succeeded}/{n_segments} segments, {failed} failed).",
                status="complete",
            )
        else:
            _post_status(
                backend_url, job_id=job_id,
                stage="complete",
                description=f"Pipeline complete — all {n_segments} segment(s) analyzed on GPU.",
                status="complete",
            )

    except Exception as e:
        import traceback
        logger.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
        _post_status(
            backend_url, job_id=job_id,
            stage="error",
            description=f"Pipeline failed: {e}",
            status="error",
            error_message=str(e),
        )


def spawn_pipeline(job_id: str, footage_url: str, config: dict, backend_url: str) -> None:
    """
    Spawn the TennisIQ pipeline asynchronously in a background thread.
    Returns immediately so FastAPI can respond with the job_id.
    """
    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, footage_url, config, backend_url),
        daemon=True,
    )
    thread.start()
    logger.info(f"Pipeline thread spawned for job {job_id}")
