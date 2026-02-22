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

_DB_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.getenv("DATABASE_PATH", os.path.join(_DB_DIR, "courtai.db"))


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
                error_message TEXT,
                output_dir TEXT
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

            CREATE TABLE IF NOT EXISTS coach_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                point_idx INTEGER NOT NULL,
                timestamp_sec REAL NOT NULL,
                note_text TEXT NOT NULL,
                player TEXT NOT NULL DEFAULT 'player_a',
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
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

            -- Historical player registry: one row per named player tracked over time
            CREATE TABLE IF NOT EXISTS players (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                coach_id TEXT NOT NULL DEFAULT 'default',
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- One row per (player, match/job) — the per-match stat snapshot
            CREATE TABLE IF NOT EXISTS player_match_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                match_label TEXT,
                match_date TEXT NOT NULL,
                -- Core computed stats stored as JSON blobs for flexibility
                shot_type_counts TEXT,         -- JSON: {forehand: N, backhand: N, ...}
                error_rate_by_shot TEXT,       -- JSON: {forehand: 0.36, ...}
                error_rate_by_rally TEXT,      -- JSON: {"1-3": 0.1, "4-6": 0.2, ...}
                avg_shot_speed_kmh REAL,
                first_serve_pct REAL,
                double_fault_count INTEGER,
                total_shots INTEGER,
                total_points INTEGER,
                points_won INTEGER,
                points_lost INTEGER,
                dominant_pattern TEXT,         -- top shot+direction combo string
                created_at TEXT NOT NULL,
                FOREIGN KEY (player_id) REFERENCES players(id),
                FOREIGN KEY (job_id) REFERENCES jobs(id),
                UNIQUE(player_id, job_id)
            );
        """)
        # Migration: add output_dir to jobs if missing
        cur = conn.execute("PRAGMA table_info(jobs)")
        columns = [row[1] for row in cur.fetchall()]
        if "output_dir" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN output_dir TEXT")

        # Migration: add player column to coach_notes if missing
        cur = conn.execute("PRAGMA table_info(coach_notes)")
        cn_columns = [row[1] for row in cur.fetchall()]
        if cn_columns and "player" not in cn_columns:
            conn.execute("ALTER TABLE coach_notes ADD COLUMN player TEXT NOT NULL DEFAULT 'player_a'")

        # Migration: ensure players / player_match_stats exist (already in CREATE IF NOT EXISTS above,
        # but belt-and-suspenders in case DB was created before this migration ran)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS players (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                coach_id TEXT NOT NULL DEFAULT 'default',
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS player_match_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                match_label TEXT,
                match_date TEXT NOT NULL,
                shot_type_counts TEXT,
                error_rate_by_shot TEXT,
                error_rate_by_rally TEXT,
                avg_shot_speed_kmh REAL,
                first_serve_pct REAL,
                double_fault_count INTEGER,
                total_shots INTEGER,
                total_points INTEGER,
                points_won INTEGER,
                points_lost INTEGER,
                dominant_pattern TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (player_id) REFERENCES players(id),
                FOREIGN KEY (job_id) REFERENCES jobs(id),
                UNIQUE(player_id, job_id)
            )
        """)


def create_job(footage_url: str, footage_type: str = "youtube", config: Optional[dict] = None) -> str:
    job_id = str(uuid.uuid4())
    now = datetime.utcnow()
    # Output directory name: date and time for easy identification (e.g. 2026-02-22_01-18-42_abc12def)
    output_dir = now.strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid.uuid4().hex[:8]
    now_iso = now.isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO jobs (id, status, stage, stage_description, footage_url, footage_type, config, created_at, updated_at, output_dir)
               VALUES (?, 'queued', 'queued', 'Pipeline queued, waiting for GPU provisioning', ?, ?, ?, ?, ?, ?)""",
            (job_id, footage_url, footage_type, json.dumps(config or {}), now_iso, now_iso, output_dir)
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


def save_coach_note(job_id: str, point_idx: int, timestamp_sec: float, note_text: str, player: str = "player_a") -> int:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO coach_notes (job_id, point_idx, timestamp_sec, note_text, player, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, point_idx, timestamp_sec, note_text, player, now),
        )
        return cur.lastrowid


def get_coach_notes(job_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM coach_notes WHERE job_id=? ORDER BY point_idx ASC, timestamp_sec ASC",
            (job_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_coach_note(note_id: int) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM coach_notes WHERE id=?", (note_id,))
        return cur.rowcount > 0


# ── Player history helpers ─────────────────────────────────────────────────────

def create_player(name: str, coach_id: str = "default", notes: str = "") -> str:
    """Create a new tracked player; return their ID."""
    player_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO players (id, name, coach_id, notes, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (player_id, name.strip(), coach_id, notes, now, now),
        )
    return player_id


def get_player(player_id: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM players WHERE id=?", (player_id,)).fetchone()
        return dict(row) if row else None


def get_players(coach_id: str = "default") -> list[dict]:
    """Return all players for a coach ordered by name."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM players WHERE coach_id=? ORDER BY name ASC",
            (coach_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_player(player_id: str, name: Optional[str] = None, notes: Optional[str] = None) -> bool:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM players WHERE id=?", (player_id,)).fetchone()
        if not row:
            return False
        new_name = name.strip() if name is not None else row["name"]
        new_notes = notes if notes is not None else row["notes"]
        conn.execute(
            "UPDATE players SET name=?, notes=?, updated_at=? WHERE id=?",
            (new_name, new_notes, now, player_id),
        )
        return True


def delete_player(player_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM players WHERE id=?", (player_id,))
        conn.execute("DELETE FROM player_match_stats WHERE player_id=?", (player_id,))
        return cur.rowcount > 0


def save_player_match_stats(
    player_id: str,
    job_id: str,
    analytics: dict,
    player_key: str = "player_a",
    match_label: str = "",
    match_date: Optional[str] = None,
) -> Optional[int]:
    """Snapshot per-match stats for a player from the analytics blob.

    analytics: the AnalyticsData dict from results.
    player_key: "player_a" or "player_b".
    Returns the new row id, or None if job already has a snapshot for this player.
    """
    pa = analytics.get(player_key) if analytics else None
    if not pa:
        return None

    now = datetime.utcnow().isoformat()
    date = match_date or now[:10]

    # Derive dominant pattern: highest-count entry in shot_type_counts
    stc = pa.get("shot_type_counts", {})
    dominant = max(stc, key=lambda k: stc[k]) if stc else None

    # Convert speeds from m/s → km/h
    avg_kmh = round(pa.get("avg_shot_speed_m_s", 0) * 3.6, 1)

    with get_connection() as conn:
        try:
            cur = conn.execute(
                """INSERT INTO player_match_stats
                   (player_id, job_id, match_label, match_date,
                    shot_type_counts, error_rate_by_shot, error_rate_by_rally,
                    avg_shot_speed_kmh, first_serve_pct, double_fault_count,
                    total_shots, total_points, points_won, points_lost,
                    dominant_pattern, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(player_id, job_id) DO UPDATE SET
                       match_label=excluded.match_label,
                       shot_type_counts=excluded.shot_type_counts,
                       error_rate_by_shot=excluded.error_rate_by_shot,
                       error_rate_by_rally=excluded.error_rate_by_rally,
                       avg_shot_speed_kmh=excluded.avg_shot_speed_kmh,
                       first_serve_pct=excluded.first_serve_pct,
                       double_fault_count=excluded.double_fault_count,
                       total_shots=excluded.total_shots,
                       total_points=excluded.total_points,
                       points_won=excluded.points_won,
                       points_lost=excluded.points_lost,
                       dominant_pattern=excluded.dominant_pattern""",
                (
                    player_id, job_id, match_label or "", date,
                    json.dumps(pa.get("shot_type_counts", {})),
                    json.dumps(pa.get("error_rate_by_shot_type", {})),
                    json.dumps(pa.get("error_rate_by_rally_length", {})),
                    avg_kmh,
                    pa.get("first_serve_pct", 0),
                    pa.get("double_fault_count", 0),
                    pa.get("total_shots", 0),
                    analytics.get("total_points", 0),
                    pa.get("points_won", 0),
                    pa.get("points_lost", 0),
                    dominant,
                    now,
                ),
            )
            return cur.lastrowid
        except Exception:
            return None


def get_player_match_history(player_id: str) -> list[dict]:
    """Return all per-match stat rows for a player, oldest first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM player_match_stats WHERE player_id=? ORDER BY match_date ASC, created_at ASC",
            (player_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            for field in ("shot_type_counts", "error_rate_by_shot", "error_rate_by_rally"):
                try:
                    d[field] = json.loads(d[field]) if d[field] else {}
                except (json.JSONDecodeError, TypeError):
                    d[field] = {}
            result.append(d)
        return result


def compute_player_trends(player_id: str) -> dict:
    """Compute trend analytics across all matches for a player.

    Returns a dict with:
      - match_count: int
      - avg_backhand_error_rate: float
      - backhand_error_trend: list of (match_date, rate) pairs
      - avg_forehand_error_rate: float
      - forehand_error_trend: list of (match_date, rate) pairs
      - avg_shot_speed_kmh_trend: list of (match_date, kmh)
      - serve_fault_trend: list of (match_date, first_serve_pct)
      - long_rally_error_trend: list of (match_date, rate)
      - dominant_patterns: dict of pattern -> count_of_matches_dominant
      - confirmed_weaknesses: list of {weakness, matches_count, avg_rate, trend}
    """
    history = get_player_match_history(player_id)
    n = len(history)
    if n == 0:
        return {"match_count": 0}

    def _trend(key: str, subkey: str) -> list[dict]:
        out = []
        for h in history:
            val = h.get(key, {}).get(subkey)
            if val is not None:
                out.append({"date": h["match_date"], "label": h.get("match_label") or h["match_date"], "value": round(val, 3)})
        return out

    def _scalar_trend(key: str) -> list[dict]:
        return [
            {"date": h["match_date"], "label": h.get("match_label") or h["match_date"], "value": h.get(key)}
            for h in history if h.get(key) is not None
        ]

    backhand_trend = _trend("error_rate_by_shot", "backhand")
    forehand_trend = _trend("error_rate_by_shot", "forehand")
    long_rally_trend = _trend("error_rate_by_rally", "7-9") or _trend("error_rate_by_rally", "10+")

    # Dominant patterns: count how many matches each was dominant
    pattern_counts: dict[str, int] = {}
    for h in history:
        dp = h.get("dominant_pattern")
        if dp:
            pattern_counts[dp] = pattern_counts.get(dp, 0) + 1

    # Confirmed weaknesses: any stat that appears consistently across ≥60% of matches
    confirmed_weaknesses = []
    if backhand_trend:
        avg_bh = sum(t["value"] for t in backhand_trend) / len(backhand_trend)
        high_count = sum(1 for t in backhand_trend if t["value"] >= 0.3)
        if avg_bh >= 0.25 or high_count >= max(1, round(len(backhand_trend) * 0.6)):
            confirmed_weaknesses.append({
                "weakness": "Backhand consistency",
                "matches_count": len(backhand_trend),
                "high_error_matches": high_count,
                "avg_rate": round(avg_bh, 3),
                "trend": backhand_trend,
                "confirmed": high_count >= max(1, round(len(backhand_trend) * 0.6)),
            })

    if long_rally_trend:
        avg_lr = sum(t["value"] for t in long_rally_trend) / len(long_rally_trend)
        high_count = sum(1 for t in long_rally_trend if t["value"] >= 0.5)
        if avg_lr >= 0.4:
            confirmed_weaknesses.append({
                "weakness": "Long rally endurance",
                "matches_count": len(long_rally_trend),
                "high_error_matches": high_count,
                "avg_rate": round(avg_lr, 3),
                "trend": long_rally_trend,
                "confirmed": high_count >= max(1, round(len(long_rally_trend) * 0.6)),
            })

    # Serve trend
    serve_trend = _scalar_trend("first_serve_pct")
    if serve_trend:
        avg_serve = sum(t["value"] for t in serve_trend) / len(serve_trend)
        if avg_serve < 0.6:
            confirmed_weaknesses.append({
                "weakness": "First serve reliability",
                "matches_count": len(serve_trend),
                "high_error_matches": sum(1 for t in serve_trend if t["value"] < 0.6),
                "avg_rate": round(1 - avg_serve, 3),
                "trend": [{"date": t["date"], "label": t["label"], "value": round(1 - t["value"], 3)} for t in serve_trend],
                "confirmed": sum(1 for t in serve_trend if t["value"] < 0.6) >= max(1, round(len(serve_trend) * 0.6)),
            })

    speed_trend = _scalar_trend("avg_shot_speed_kmh")
    avg_speed = round(sum(t["value"] for t in speed_trend) / len(speed_trend), 1) if speed_trend else None

    return {
        "match_count": n,
        "avg_shot_speed_kmh": avg_speed,
        "shot_speed_trend": speed_trend,
        "backhand_error_trend": backhand_trend,
        "forehand_error_trend": forehand_trend,
        "long_rally_error_trend": long_rally_trend,
        "serve_fault_trend": serve_trend,
        "dominant_patterns": pattern_counts,
        "confirmed_weaknesses": confirmed_weaknesses,
    }


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
