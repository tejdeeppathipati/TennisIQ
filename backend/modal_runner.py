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
    """Run a single 10-second segment on Modal T4 GPU. Raises on failure."""
    import modal
    fn = modal.Function.from_name("tennisiq-court", "run_court_and_ball")
    return fn.remote(
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
        self.all_frames_jsonl: list[str] = []
        self.overlay_parts: list[bytes] = []
        self.clip_counter = 0
        self.latest_stats: dict = {}
        self.latest_run: dict = {}
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

        stats = self._build_merged_stats(fps, duration, n_segments, succeeded)
        _write_json(self.out_dir / "stats.json", stats)

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
