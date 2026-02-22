"""
TennisIQ FastAPI backend.
Endpoints: /ingest, /status, /checkpoint, /results, /sessions, /outputs
State managed in SQLite via db.py.
Modal pipeline spawned non-blocking via modal_runner.py.
"""
import os
import re
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

import db
import modal_runner

YOUTUBE_URL_RE = re.compile(
    r"^https?://(?:www\.|m\.)?(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/embed/)[\w-]+",
    re.IGNORECASE,
)
MIN_UPLOAD_BYTES = 10_240  # 10 KB


def is_valid_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_URL_RE.match(url))


def validate_video_file(file_path: str) -> tuple[bool, str]:
    """Open an uploaded MP4 with OpenCV and verify it contains readable video frames."""
    import cv2

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        cap.release()
        return False, "The uploaded file could not be opened as a video. It may be corrupted or not a valid MP4."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return False, "The uploaded video contains no frames. It may be corrupted or empty."

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False, "The uploaded video exists but no frames could be read. The file may be corrupted."

    return True, ""

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", os.path.join(os.path.dirname(__file__), "..", "outputs"))
DEFAULT_CONFIG = {
    "fps": 30,
    "max_iterations": 3,
    "train_split": 0.75,
    "min_frames": 500,
    "player_map_floor": 0.80,
    "ball_map_floor": 0.70,
    "court_lines_map_floor": 0.75,
    "net_map_floor": 0.75,
    "fp_rate_ceiling": 0.10,
    "phash_threshold": 10,
    "checkpoint_frames": 24,
    "shard_size": 25,
    "dead_frame_motion_threshold": 0.02,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    logger.info("TennisIQ backend started. Database initialized.")
    yield


app = FastAPI(
    title="TennisIQ API",
    description="Tennis vision intelligence pipeline for college coaches",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

outputs_abs = os.path.abspath(OUTPUTS_DIR)
os.makedirs(outputs_abs, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=outputs_abs), name="outputs")


# ── Request / Response Models ──────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    url: str
    config: Optional[dict] = None


class FrameFeedback(BaseModel):
    frame_index: int
    action: str  # "approve" | "reject"
    note: Optional[str] = None


class CheckpointSubmission(BaseModel):
    feedback: list[FrameFeedback]


class PointFeedback(BaseModel):
    """Coach feedback for a single detected point (FR-31)."""
    action: str  # "confirm" | "flag"
    note: Optional[str] = None


class StatusUpdate(BaseModel):
    """Received from Modal pipeline to update job state."""
    job_id: str
    stage: str
    description: str
    status: Optional[str] = "running"
    iteration: Optional[int] = None
    eval_metrics: Optional[dict] = None
    decision: Optional[dict] = None
    artifacts: Optional[list[dict]] = None
    checkpoint_frames: Optional[list[dict]] = None
    points_for_review: Optional[str] = None
    error_message: Optional[str] = None
    segment_complete: Optional[dict] = None  # NFR-R06: {"idx": 0, "result_key": "..."}
    segments: Optional[list[dict]] = None    # NFR-R06: register segments at start


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "tennisiq-backend"}


@app.post("/ingest")
async def ingest_url(request: IngestURLRequest):
    """
    Accept a YouTube URL, create a pipeline job, spawn Modal GPU function.
    Returns job_id immediately for frontend to begin polling /status.
    """
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required.")

    if not is_valid_youtube_url(url):
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid YouTube URL (e.g. https://www.youtube.com/watch?v=... or https://youtu.be/...).",
        )

    config = {**DEFAULT_CONFIG, **(request.config or {})}
    job_id = db.create_job(footage_url=url, footage_type="youtube", config=config)

    try:
        modal_runner.spawn_pipeline(
            job_id=job_id,
            footage_url=url,
            config=config,
            backend_url=BACKEND_URL,
        )
    except Exception as e:
        db.set_error(job_id, f"Failed to start pipeline: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline could not be started. Modal may be unavailable: {str(e)}"
        )

    return {"job_id": job_id, "status": "queued"}


@app.post("/ingest/upload")
async def ingest_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config: str = Form(default="{}"),
):
    """
    Accept an MP4 file upload as fallback when YouTube ingestion fails.
    Saves file, creates job, spawns Modal pipeline with local file path.
    """
    allowed_ext = (".mp4", ".mov", ".mkv", ".webm")
    if not file.filename or not file.filename.lower().endswith(allowed_ext):
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(allowed_ext)}")

    upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
    os.makedirs(upload_dir, exist_ok=True)

    try:
        parsed_config = {**DEFAULT_CONFIG, **json.loads(config)}
    except json.JSONDecodeError:
        parsed_config = DEFAULT_CONFIG

    import uuid
    safe_name = f"{uuid.uuid4()}.mp4"
    file_path = os.path.join(upload_dir, safe_name)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    file_size = os.path.getsize(file_path)
    if file_size < MIN_UPLOAD_BYTES:
        os.remove(file_path)
        raise HTTPException(
            status_code=400,
            detail=f"The uploaded file is too small ({file_size} bytes). Please upload a valid tennis match MP4.",
        )

    valid, reason = validate_video_file(file_path)
    if not valid:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=reason)

    job_id = db.create_job(
        footage_url=f"file://{os.path.abspath(file_path)}",
        footage_type="upload",
        config=parsed_config
    )

    try:
        modal_runner.spawn_pipeline(
            job_id=job_id,
            footage_url=f"file://{os.path.abspath(file_path)}",
            config=parsed_config,
            backend_url=BACKEND_URL,
        )
    except Exception as e:
        db.set_error(job_id, f"Failed to start pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Return current pipeline state. Polled every 5s by the frontend.
    Returns stage, description, decision log, latest eval metrics.
    """
    status = db.get_full_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found.")
    return status


@app.post("/status/update")
async def update_status(update: StatusUpdate):
    """
    Receive stage updates from the Modal pipeline function.
    Modal POSTs here after each stage completes.
    """
    job = db.get_job(update.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if update.error_message:
        db.set_error(update.job_id, update.error_message)
        return {"ok": True}

    db.update_stage(update.job_id, update.stage, update.description, update.status or "running")

    if update.iteration is not None:
        db.update_iteration(update.job_id, update.iteration)

    if update.eval_metrics:
        db.save_eval_result(update.job_id, update.iteration or 0, update.eval_metrics)

    if update.decision:
        db.save_decision(
            update.job_id,
            update.iteration or 0,
            update.decision.get("action", ""),
            update.decision.get("justification", ""),
        )

    if update.checkpoint_frames:
        db.save_checkpoint_frames(update.job_id, update.checkpoint_frames)
        db.update_stage(
            update.job_id,
            "awaiting_review",
            "Waiting for coach to review checkpoint frames.",
            "awaiting_review"
        )

    if update.artifacts:
        for artifact in update.artifacts:
            db.save_artifact(
                update.job_id,
                artifact.get("type", "unknown"),
                artifact.get("path", ""),
                artifact.get("metadata"),
            )

    if update.segments:
        db.create_segments(update.job_id, update.segments)

    if update.segment_complete:
        db.mark_segment_complete(
            update.job_id,
            update.segment_complete["idx"],
            update.segment_complete.get("result_key"),
        )

    if update.points_for_review:
        db.save_points_for_review(update.job_id, update.points_for_review)

    if update.status == "complete":
        db.set_complete(update.job_id)

    return {"ok": True}


@app.get("/checkpoint/{job_id}")
async def get_checkpoint(job_id: str):
    """Return the 24 checkpoint frames for coach review."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    frames = db.get_checkpoint_frames(job_id)
    if not frames:
        raise HTTPException(status_code=404, detail="No checkpoint frames available yet.")

    return {"job_id": job_id, "frames": frames}


@app.post("/checkpoint/{job_id}")
async def submit_checkpoint(job_id: str, submission: CheckpointSubmission):
    """
    Receive coach feedback for checkpoint frames.
    Written to SQLite instantly — Modal polls for this to resume the pipeline.
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    feedback = [f.model_dump() for f in submission.feedback]
    db.save_feedback(job_id, feedback)

    return {"ok": True, "feedback_count": len(feedback)}


@app.get("/segments/{job_id}")
async def get_segments(job_id: str):
    """NFR-R06: Return segment completion status for a job."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    all_segs = db.get_all_segments(job_id)
    incomplete = [s for s in all_segs if s["status"] != "complete"]
    return {
        "job_id": job_id,
        "total_segments": len(all_segs),
        "complete": len(all_segs) - len(incomplete),
        "incomplete": incomplete,
    }


@app.get("/checkpoint/{job_id}/points")
async def get_points_for_review(job_id: str):
    """FR-30/32: Return detected points + overlay video path for coach review.

    The pipeline pauses in 'awaiting_point_review' state after inference.
    The coach sees the point list and can play the overlay alongside each point.
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    points_json = db.get_points_for_review(job_id)
    if not points_json:
        raise HTTPException(
            status_code=404,
            detail="No points available for review. Pipeline may still be running.",
        )

    points = json.loads(points_json)

    existing_feedback = db.get_point_feedback(job_id)
    feedback_map = {pf["point_idx"]: pf for pf in existing_feedback}

    overlay_artifact = None
    artifacts = db.get_artifacts(job_id)
    for a in artifacts:
        if a["artifact_type"] == "overlay_video":
            overlay_artifact = a["artifact_path"]
            break

    for pt in points:
        idx = pt["point_idx"]
        if idx in feedback_map:
            pt["coach_action"] = feedback_map[idx]["action"]
            pt["coach_note"] = feedback_map[idx]["note"]
        else:
            pt["coach_action"] = None
            pt["coach_note"] = None

    return {
        "job_id": job_id,
        "status": job["status"],
        "points": points,
        "overlay_video_path": overlay_artifact,
        "review_complete": job["status"] != "awaiting_point_review",
    }


@app.post("/checkpoint/{job_id}/points/{point_idx}/feedback")
async def submit_point_feedback(job_id: str, point_idx: int, fb: PointFeedback):
    """FR-31/33: Coach confirms, flags, or annotates a single point.

    Written to SQLite instantly upon submission.
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if fb.action not in ("confirm", "flag"):
        raise HTTPException(
            status_code=400,
            detail="Action must be 'confirm' or 'flag'.",
        )

    db.save_point_feedback(job_id, point_idx, fb.action, fb.note)
    return {"ok": True, "point_idx": point_idx, "action": fb.action}


@app.post("/checkpoint/{job_id}/finalize")
async def finalize_review(job_id: str):
    """FR-34: Mark point review as complete and apply coach feedback.

    Flagged points get their confidence penalized and are marked in the
    final output. Confirmed points keep their original confidence.
    Returns the adjusted points list.
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] != "awaiting_point_review":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not awaiting review (current status: {job['status']}).",
        )

    points_json = db.get_points_for_review(job_id)
    if not points_json:
        raise HTTPException(status_code=404, detail="No points to finalize.")

    points = json.loads(points_json)
    feedback_list = db.get_point_feedback(job_id)
    feedback_map = {pf["point_idx"]: pf for pf in feedback_list}

    adjusted = apply_coach_feedback(points, feedback_map)

    db.save_artifact(
        job_id, "points_final", "inline",
        metadata=adjusted,
    )
    db.finalize_point_review(job_id)

    return {
        "ok": True,
        "job_id": job_id,
        "total_points": len(adjusted),
        "confirmed": sum(1 for p in adjusted if p.get("coach_action") == "confirm"),
        "flagged": sum(1 for p in adjusted if p.get("coach_action") == "flag"),
        "excluded": sum(1 for p in adjusted if p.get("excluded")),
    }


def apply_coach_feedback(points: list[dict], feedback_map: dict) -> list[dict]:
    """FR-34: Apply coach feedback to adjust confidence and exclusion flags.

    - Confirmed points keep original confidence, marked coach_action="confirm".
    - Flagged points get confidence halved and a flag marker.
    - If coach note contains "exclude", the point is excluded from final analytics.
    - Unreviewed points keep original values.
    """
    CONFIDENCE_PENALTY = 0.5

    adjusted = []
    for pt in points:
        pt = dict(pt)
        idx = pt["point_idx"]
        fb = feedback_map.get(idx)

        if fb:
            pt["coach_action"] = fb["action"]
            pt["coach_note"] = fb.get("note")

            if fb["action"] == "flag":
                pt["original_confidence"] = pt["confidence"]
                pt["confidence"] = round(pt["confidence"] * CONFIDENCE_PENALTY, 3)
                note_lower = (fb.get("note") or "").lower()
                pt["excluded"] = "exclude" in note_lower
            else:
                pt["excluded"] = False
        else:
            pt["coach_action"] = None
            pt["coach_note"] = None
            pt["excluded"] = False

        adjusted.append(pt)
    return adjusted


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Return final pipeline artifacts, eval metrics, and decision log."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] not in ("complete", "error"):
        raise HTTPException(status_code=202, detail="Pipeline still running.")

    artifacts = db.get_artifacts(job_id)
    eval_results = db.get_eval_results(job_id)
    decisions = db.get_decisions(job_id)
    feedback = db.get_feedback(job_id)
    point_feedback = db.get_point_feedback(job_id)

    artifacts_by_type: dict = {}
    for a in artifacts:
        t = a["artifact_type"]
        if t not in artifacts_by_type:
            artifacts_by_type[t] = []
        artifacts_by_type[t].append(a)

    return {
        "job_id": job_id,
        "status": job["status"],
        "footage_url": job["footage_url"],
        "artifacts": artifacts_by_type,
        "eval_results": eval_results,
        "decisions": decisions,
        "coach_feedback": feedback,
        "point_feedback": point_feedback,
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


@app.get("/results/{job_id}/data")
async def get_results_data(job_id: str):
    """Return all structured output data for the results dashboard.

    Reads actual JSON files from the outputs directory and returns them
    in the format the frontend components expect: points, coaching cards,
    serve placement, heatmaps, stats, events, and video URLs.
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] not in ("complete", "error"):
        raise HTTPException(status_code=202, detail="Pipeline still running.")

    run_dir = Path(outputs_abs) / job_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Output directory not found for job {job_id}.")

    api_base = f"/outputs/{job_id}"

    data: dict = {
        "job_id": job_id,
        "status": job["status"],
        "footage_url": job.get("footage_url"),
        "overlay_video_url": None,
        "raw_video_url": job.get("footage_url"),
        "points": [],
        "events": [],
        "coaching_cards": [],
        "serve_placement": None,
        "error_heatmap": None,
        "player_a_heatmap": None,
        "player_b_heatmap": None,
        "stats": None,
        "clips": [],
        "downloads": [],
        "shots": [],
        "analytics": None,
        "player_a_card": None,
        "player_b_card": None,
        "match_flow": None,
    }

    def _load_json(relpath: str):
        p = run_dir / relpath
        if p.exists():
            try:
                return json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                return None
        return None

    data["points"] = _load_json("points.json") or []
    data["events"] = _load_json("events.json") or []
    data["stats"] = _load_json("stats.json")

    raw_video_path = run_dir / "raw_video.mp4"
    if raw_video_path.exists():
        data["raw_video_url"] = f"{api_base}/raw_video.mp4"
        data["downloads"].append({"label": "Raw Video", "href": f"{api_base}/raw_video.mp4"})

    overlay_path = run_dir / "visuals" / "overlay.mp4"
    if overlay_path.exists():
        data["overlay_video_url"] = f"{api_base}/visuals/overlay.mp4"
        data["downloads"].append({"label": "Overlay Video", "href": f"{api_base}/visuals/overlay.mp4"})

    data["downloads"].append({"label": "Points JSON", "href": f"{api_base}/points.json"})
    data["downloads"].append({"label": "Events JSON", "href": f"{api_base}/events.json"})
    data["downloads"].append({"label": "Stats JSON", "href": f"{api_base}/stats.json"})

    data["coaching_cards"] = _load_json("coaching_cards.json") or []
    if not data["coaching_cards"] and data["points"]:
        data["coaching_cards"] = _generate_coaching_cards_from_points(data["points"])

    data["serve_placement"] = _load_json("visuals/serve_placement.json")
    if not data["serve_placement"] and data["points"]:
        data["serve_placement"] = _generate_serve_placement_from_points(data["points"])

    data["error_heatmap"] = _load_json("visuals/error_heatmap.json")
    if not data["error_heatmap"] and data["events"]:
        data["error_heatmap"] = _generate_error_heatmap_from_events(data["events"])

    pa_heatmap = _load_json("visuals/player_a_heatmap.json")
    pb_heatmap = _load_json("visuals/player_b_heatmap.json")
    if pa_heatmap:
        data["player_a_heatmap"] = pa_heatmap
    if pb_heatmap:
        data["player_b_heatmap"] = pb_heatmap

    if not pa_heatmap or not pb_heatmap:
        coverage = _load_json("visuals/player_coverage.json")
        if coverage:
            if not pa_heatmap and "player_a" in coverage:
                data["player_a_heatmap"] = _positions_to_heatmap(coverage["player_a"], "Player A")
            if not pb_heatmap and "player_b" in coverage:
                data["player_b_heatmap"] = _positions_to_heatmap(coverage["player_b"], "Player B")

    ball_heatmap = _load_json("visuals/ball_heatmap.json")
    if ball_heatmap:
        data["ball_heatmap"] = ball_heatmap

    # ── New analytics data ────────────────────────────────────────────────
    data["shots"] = _load_json("shots.json") or []
    data["analytics"] = _load_json("analytics.json")
    data["player_a_card"] = _load_json("player_a_card.json")
    data["player_b_card"] = _load_json("player_b_card.json")
    data["match_flow"] = _load_json("match_flow.json")

    if data["analytics"]:
        data["downloads"].append({"label": "Analytics JSON", "href": f"{api_base}/analytics.json"})
    if data["shots"]:
        data["downloads"].append({"label": "Shots JSON", "href": f"{api_base}/shots.json"})

    clips_dir = run_dir / "clips"
    if clips_dir.is_dir():
        for clip_file in sorted(clips_dir.glob("*.mp4")):
            data["clips"].append({
                "filename": clip_file.name,
                "url": f"{api_base}/clips/{clip_file.name}",
            })
            data["downloads"].append({"label": clip_file.stem, "href": f"{api_base}/clips/{clip_file.name}"})

    point_feedback = db.get_point_feedback(job_id)
    data["point_feedback"] = point_feedback

    return data


def _generate_coaching_cards_from_points(points: list) -> list:
    """Generate coaching cards from point data when coaching_cards.json is missing."""
    cards = []
    for pt in points:
        end = pt.get("end_reason", "UNKNOWN")
        hits = pt.get("rally_hit_count", 0)
        zone = pt.get("serve_zone")
        fault = pt.get("serve_fault_type")

        if end == "OUT":
            summary = f"Point ended with an out ball after {hits} hits."
            suggestion = "Focus on keeping the ball within the court lines during extended rallies."
        elif end == "BALL_LOST":
            summary = f"Ball tracking was lost after {hits} hits in this rally."
            suggestion = "This point may have ended due to a fast exchange or camera angle change."
        elif end == "NET":
            summary = f"Ball hit the net after {hits} hits."
            suggestion = "Try adding more height over the net on approach shots."
        elif end == "DOUBLE_BOUNCE":
            summary = f"Double bounce detected after {hits} hits."
            suggestion = "Work on court coverage to reach the ball before the second bounce."
        else:
            summary = f"Point ended ({end}) after {hits} hits."
            suggestion = "Review the video clip for this point to identify improvement areas."

        if fault:
            summary += f" Serve fault: {fault}."
            if fault == "wide":
                suggestion = "Aim serves more toward the center of the service box."
            elif fault == "long":
                suggestion = "Reduce serve power or add more topspin to keep it in the box."
            elif fault == "net":
                suggestion = "Increase serve trajectory height to clear the net."

        cards.append({
            "point_idx": pt.get("point_idx", 0),
            "summary": summary,
            "suggestion": suggestion,
            "start_sec": pt.get("start_sec", 0),
            "end_sec": pt.get("end_sec", 0),
            "rally_hit_count": hits,
            "bounce_count": pt.get("bounce_count", 0),
            "end_reason": end,
            "serve_zone": zone,
            "serve_fault_type": fault,
            "confidence": pt.get("confidence", 0),
        })
    return cards


def _generate_serve_placement_from_points(points: list) -> dict | None:
    """Generate serve placement data from points when serve_placement.json is missing."""
    serves = []
    for pt in points:
        court_xy = pt.get("first_bounce_court_xy")
        if not court_xy:
            continue
        serves.append({
            "point_idx": pt.get("point_idx", 0),
            "court_x": court_xy[0],
            "court_y": court_xy[1],
            "serve_zone": pt.get("serve_zone"),
            "is_fault": pt.get("serve_fault_type") is not None,
            "fault_type": pt.get("serve_fault_type"),
            "serve_player": pt.get("serve_player"),
        })

    if not serves:
        return None

    service_boxes = {
        "far_left": {"x_min": 0, "y_min": 0, "x_max": 685, "y_max": 1828},
        "far_right": {"x_min": 685, "y_min": 0, "x_max": 1370, "y_max": 1828},
        "near_left": {"x_min": 0, "y_min": 1828, "x_max": 685, "y_max": 3657},
        "near_right": {"x_min": 685, "y_min": 1828, "x_max": 1370, "y_max": 3657},
    }
    return {"serves": serves, "service_boxes": service_boxes}


def _generate_error_heatmap_from_events(events: list) -> dict | None:
    """Generate error heatmap from events when error_heatmap.json is missing."""
    out_positions = []
    for e in events:
        if e.get("event_type") == "bounce" and e.get("in_out") == "out":
            xy = e.get("court_xy")
            if xy:
                out_positions.append(xy)

    if not out_positions:
        return None

    import numpy as np
    xs = [p[0] for p in out_positions]
    ys = [p[1] for p in out_positions]
    grid, x_edges, y_edges = np.histogram2d(xs, ys, bins=[10, 15])
    return {
        "grid": grid.T.tolist(),
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
        "total_out_bounces": len(out_positions),
    }


def _positions_to_heatmap(positions: list, label: str) -> dict:
    """Convert raw [x,y] position arrays into HeatmapData grid format."""
    import numpy as np
    if not positions:
        return {"grid": [], "x_edges": [], "y_edges": [], "total_frames": 0}
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    grid, x_edges, y_edges = np.histogram2d(xs, ys, bins=[10, 15])
    return {
        "grid": grid.T.tolist(),
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
        "total_frames": len(positions),
    }


# ── Session Persistence (FR-47 / FR-48) ──────────────────────────────────────


class SessionSave(BaseModel):
    """FR-47: Manually save a session record with custom preferences."""
    coach_id: Optional[str] = "default"
    preferences: Optional[dict] = None


@app.post("/sessions/{job_id}/save")
async def save_session(job_id: str, body: SessionSave):
    """FR-47: Persist a session record after pipeline completion.

    Bundles footage metadata, coach feedback, detection summary, and
    optional preferences so they can be restored on the next run.
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Job must be complete to save a session (current: {job['status']}).",
        )

    point_feedback = db.get_point_feedback(job_id)

    detection_summary = _build_detection_summary(job_id)

    session_data = {
        "job_id": job_id,
        "coach_id": body.coach_id or "default",
        "footage_url": job["footage_url"],
        "footage_type": job["footage_type"],
        "fps": detection_summary.get("fps"),
        "frame_count": detection_summary.get("frame_count"),
        "duration_sec": detection_summary.get("duration_sec"),
        "total_points": detection_summary.get("total_points", 0),
        "total_events": detection_summary.get("total_events", 0),
        "coach_feedback": point_feedback,
        "detection_summary": detection_summary,
        "preferences": body.preferences or {},
    }

    session_id = db.save_session(session_data)
    return {"ok": True, "session_id": session_id, "job_id": job_id}


@app.get("/sessions")
async def list_sessions(coach_id: str = "default"):
    """FR-48: List all saved sessions for a coach, newest first."""
    sessions = db.get_sessions_for_coach(coach_id)
    return {"coach_id": coach_id, "sessions": sessions}


@app.get("/sessions/latest/{coach_id}")
async def get_latest_session(coach_id: str = "default"):
    """FR-48: Load the most recent session for a coach.

    Returns preferences and flagged corrections that can be applied
    as starting state for the next pipeline run.
    """
    session = db.get_latest_session_for_coach(coach_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="No previous sessions found for this coach.",
        )
    return session


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """FR-48: Retrieve a single session record with preferences and feedback."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


def _build_detection_summary(job_id: str) -> dict:
    """Gather run metadata and detection counts from stored artifacts."""
    artifacts = db.get_artifacts(job_id)

    summary: dict = {}
    for a in artifacts:
        if a["artifact_type"] == "run_metadata":
            try:
                meta = json.loads(a["metadata"]) if isinstance(a["metadata"], str) else a["metadata"]
                summary["fps"] = meta.get("fps")
                summary["frame_count"] = meta.get("frame_count")
                if summary.get("fps") and summary.get("frame_count"):
                    summary["duration_sec"] = round(summary["frame_count"] / summary["fps"], 2)
            except (json.JSONDecodeError, TypeError):
                pass
        elif a["artifact_type"] == "points_final":
            try:
                points = json.loads(a["metadata"]) if isinstance(a["metadata"], str) else a["metadata"]
                if isinstance(points, list):
                    summary["total_points"] = len(points)
            except (json.JSONDecodeError, TypeError):
                pass
        elif a["artifact_type"] == "events":
            try:
                events = json.loads(a["metadata"]) if isinstance(a["metadata"], str) else a["metadata"]
                if isinstance(events, list):
                    summary["total_events"] = len(events)
            except (json.JSONDecodeError, TypeError):
                pass

    return summary
