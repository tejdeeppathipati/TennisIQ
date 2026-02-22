"""
SQLite helpers for TennisIQ job state management.
Handles jobs, stage updates, point-level coach feedback, session persistence, and results.
"""
import sqlite3
import json
import os
import uuid
from datetime import datetime
from typing import Optional

DATABASE_PATH = os.getenv("DATABASE_PATH", "./courtai.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'queued',
                stage TEXT,
                stage_description TEXT,
                footage_url TEXT,
                footage_type TEXT DEFAULT 'youtube',
                config TEXT,
                iteration INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error_message TEXT
            );

            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                action TEXT NOT NULL,
                justification TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            );

            CREATE TABLE IF NOT EXISTS checkpoint_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                frame_path TEXT NOT NULL,
                overlay_path TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            );

            CREATE TABLE IF NOT EXISTS coach_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                action TEXT NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            );

            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                player_map REAL,
                ball_map REAL,
                court_lines_map REAL,
                net_map REAL,
                fp_rate REAL,
                generalization_score REAL,
                frame_count INTEGER,
                criteria_met INTEGER DEFAULT 0,
                details TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            );

            CREATE TABLE IF NOT EXISTS segment_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                segment_idx INTEGER NOT NULL,
                start_sec REAL NOT NULL,
                end_sec REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                result_key TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id),
                UNIQUE(job_id, segment_idx)
            );

            CREATE TABLE IF NOT EXISTS point_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                point_idx INTEGER NOT NULL,
                action TEXT NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id),
                UNIQUE(job_id, point_idx)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                coach_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                footage_url TEXT,
                footage_type TEXT,
                fps REAL,
                frame_count INTEGER,
                duration_sec REAL,
                total_points INTEGER,
                total_events INTEGER,
                coach_feedback TEXT,
                detection_summary TEXT,
                preferences TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            );
        """)


def create_job(footage_url: str, footage_type: str = "youtube", config: Optional[dict] = None) -> str:
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO jobs (id, status, stage, stage_description, footage_url, footage_type, config, created_at, updated_at)
               VALUES (?, 'queued', 'queued', 'Pipeline queued, waiting for GPU provisioning', ?, ?, ?, ?, ?)""",
            (job_id, footage_url, footage_type, json.dumps(config or {}), now, now)
        )
    return job_id


def update_stage(job_id: str, stage: str, description: str, status: str = "running") -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET stage=?, stage_description=?, status=?, updated_at=? WHERE id=?",
            (stage, description, status, now, job_id)
        )


def update_iteration(job_id: str, iteration: int) -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET iteration=?, updated_at=? WHERE id=?",
            (iteration, now, job_id)
        )


def set_error(job_id: str, message: str) -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET status='error', error_message=?, updated_at=? WHERE id=?",
            (message, now, job_id)
        )


def set_complete(job_id: str) -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET status='complete', stage='complete', stage_description='Pipeline complete. Results ready.', updated_at=? WHERE id=?",
            (now, job_id)
        )


def get_job(job_id: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None


def save_decision(job_id: str, iteration: int, action: str, justification: str) -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO decisions (job_id, iteration, action, justification, created_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, iteration, action, justification, now)
        )


def get_decisions(job_id: str) -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions WHERE job_id=? ORDER BY created_at ASC",
            (job_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def save_checkpoint_frames(job_id: str, frames: list[dict]) -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.executemany(
            "INSERT INTO checkpoint_frames (job_id, frame_index, frame_path, overlay_path, created_at) VALUES (?, ?, ?, ?, ?)",
            [(job_id, f["frame_index"], f["frame_path"], f.get("overlay_path"), now) for f in frames]
        )


def get_checkpoint_frames(job_id: str) -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM checkpoint_frames WHERE job_id=? ORDER BY frame_index ASC",
            (job_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def save_feedback(job_id: str, feedback: list[dict]) -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.executemany(
            "INSERT INTO coach_feedback (job_id, frame_index, action, note, created_at) VALUES (?, ?, ?, ?, ?)",
            [(job_id, f["frame_index"], f["action"], f.get("note"), now) for f in feedback]
        )
        conn.execute(
            "UPDATE jobs SET status='running', stage='awaiting_training', stage_description='Coach feedback received. Starting fine-tuning.', updated_at=? WHERE id=?",
            (now, job_id)
        )


def get_feedback(job_id: str) -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM coach_feedback WHERE job_id=? ORDER BY frame_index ASC",
            (job_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def is_checkpoint_approved(job_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return False
        return row["status"] not in ("awaiting_review",)


def save_eval_result(job_id: str, iteration: int, metrics: dict) -> None:
    now = datetime.utcnow().isoformat()
    criteria_met = 1 if metrics.get("criteria_met") else 0
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO eval_results
               (job_id, iteration, player_map, ball_map, court_lines_map, net_map,
                fp_rate, generalization_score, frame_count, criteria_met, details, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id, iteration,
                metrics.get("player_map"), metrics.get("ball_map"),
                metrics.get("court_lines_map"), metrics.get("net_map"),
                metrics.get("fp_rate"), metrics.get("generalization_score"),
                metrics.get("frame_count"), criteria_met,
                json.dumps(metrics), now
            )
        )


def get_eval_results(job_id: str) -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM eval_results WHERE job_id=? ORDER BY iteration ASC",
            (job_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def save_artifact(job_id: str, artifact_type: str, artifact_path: str, metadata=None) -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO artifacts (job_id, artifact_type, artifact_path, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, artifact_type, artifact_path, json.dumps({} if metadata is None else metadata), now)
        )


def get_artifacts(job_id: str) -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM artifacts WHERE job_id=? ORDER BY created_at ASC",
            (job_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def create_segments(job_id: str, segments: list[dict]) -> None:
    """NFR-R06: Register all segments for a job so we can track completion."""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO segment_status (job_id, segment_idx, start_sec, end_sec, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'pending', ?, ?)
               ON CONFLICT(job_id, segment_idx) DO NOTHING""",
            [(job_id, s["idx"], s["start_sec"], s["end_sec"], now, now) for s in segments],
        )


def mark_segment_complete(job_id: str, segment_idx: int, result_key: str | None = None) -> None:
    """NFR-R06: Mark a single segment as complete."""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE segment_status SET status='complete', result_key=?, updated_at=? WHERE job_id=? AND segment_idx=?",
            (result_key, now, job_id, segment_idx),
        )


def get_incomplete_segments(job_id: str) -> list[dict]:
    """NFR-R06: Return segments that haven't completed yet."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM segment_status WHERE job_id=? AND status != 'complete' ORDER BY segment_idx",
            (job_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_segments(job_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM segment_status WHERE job_id=? ORDER BY segment_idx",
            (job_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def save_point_feedback(job_id: str, point_idx: int, action: str, note: Optional[str] = None) -> None:
    """Save or update coach feedback for a single detected point (FR-31/33).

    action: "confirm" | "flag" | only used with a note
    Uses INSERT OR REPLACE so the coach can revise feedback.
    """
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO point_feedback (job_id, point_idx, action, note, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(job_id, point_idx) DO UPDATE SET
                   action=excluded.action,
                   note=excluded.note,
                   created_at=excluded.created_at""",
            (job_id, point_idx, action, note, now),
        )


def get_point_feedback(job_id: str) -> list:
    """Return all point feedback for a job, ordered by point index."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM point_feedback WHERE job_id=? ORDER BY point_idx ASC",
            (job_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_single_point_feedback(job_id: str, point_idx: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM point_feedback WHERE job_id=? AND point_idx=?",
            (job_id, point_idx),
        ).fetchone()
        return dict(row) if row else None


def save_points_for_review(job_id: str, points_json: str) -> None:
    """Store the raw points JSON so the checkpoint endpoint can serve it (FR-30)."""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO artifacts (job_id, artifact_type, artifact_path, metadata, created_at)
               VALUES (?, 'points_review', 'inline', ?, ?)""",
            (job_id, points_json, now),
        )
        conn.execute(
            "UPDATE jobs SET status='awaiting_point_review', stage='awaiting_point_review', "
            "stage_description='Inference complete. Waiting for coach to review detected points.', updated_at=? WHERE id=?",
            (now, job_id),
        )


def get_points_for_review(job_id: str) -> Optional[str]:
    """Retrieve stored points JSON for checkpoint review."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT metadata FROM artifacts WHERE job_id=? AND artifact_type='points_review' ORDER BY created_at DESC LIMIT 1",
            (job_id,),
        ).fetchone()
        return row["metadata"] if row else None


def finalize_point_review(job_id: str) -> None:
    """Mark point review as complete so the pipeline can finalize output (FR-30)."""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET status='running', stage='finalizing', "
            "stage_description='Coach review complete. Applying feedback and generating final output.', updated_at=? WHERE id=?",
            (now, job_id),
        )


def save_session(session: dict) -> str:
    """FR-47: Persist a complete session record after pipeline completion.

    The session bundles footage metadata, detection summary, coach feedback,
    and any preferences so they can be loaded for future runs (FR-48).
    """
    session_id = session.get("id") or str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO sessions
               (id, coach_id, job_id, footage_url, footage_type,
                fps, frame_count, duration_sec,
                total_points, total_events,
                coach_feedback, detection_summary, preferences, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                session.get("coach_id", "default"),
                session["job_id"],
                session.get("footage_url"),
                session.get("footage_type"),
                session.get("fps"),
                session.get("frame_count"),
                session.get("duration_sec"),
                session.get("total_points", 0),
                session.get("total_events", 0),
                json.dumps(session.get("coach_feedback", [])),
                json.dumps(session.get("detection_summary", {})),
                json.dumps(session.get("preferences", {})),
                now,
            ),
        )
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    """Return a single session record."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["coach_feedback"] = json.loads(d["coach_feedback"]) if d["coach_feedback"] else []
        d["detection_summary"] = json.loads(d["detection_summary"]) if d["detection_summary"] else {}
        d["preferences"] = json.loads(d["preferences"]) if d["preferences"] else {}
        return d


def get_sessions_for_coach(coach_id: str = "default") -> list[dict]:
    """FR-48: Return all sessions for a coach, newest first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE coach_id=? ORDER BY created_at DESC",
            (coach_id,),
        ).fetchall()
        sessions = []
        for row in rows:
            d = dict(row)
            d["coach_feedback"] = json.loads(d["coach_feedback"]) if d["coach_feedback"] else []
            d["detection_summary"] = json.loads(d["detection_summary"]) if d["detection_summary"] else {}
            d["preferences"] = json.loads(d["preferences"]) if d["preferences"] else {}
            sessions.append(d)
        return sessions


def get_latest_session_for_coach(coach_id: str = "default") -> Optional[dict]:
    """FR-48: Return the most recent session for loading preferences."""
    sessions = get_sessions_for_coach(coach_id)
    return sessions[0] if sessions else None


def get_full_status(job_id: str) -> Optional[dict]:
    """Return complete job status including decisions and latest eval for /status endpoint."""
    job = get_job(job_id)
    if not job:
        return None
    decisions = get_decisions(job_id)
    eval_results = get_eval_results(job_id)
    latest_eval = eval_results[-1] if eval_results else None
    return {
        "job_id": job_id,
        "status": job["status"],
        "stage": job["stage"],
        "stage_description": job["stage_description"],
        "iteration": job["iteration"],
        "error_message": job["error_message"],
        "decisions": decisions,
        "latest_eval": latest_eval,
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }
