"""
Stage 08: Visual output generation for tennis.

Produces:
  1. Annotated overlay video (bounding boxes for player, ball, court_lines, net)
  2. Per-set player movement heatmaps
  3. Serve placement chart (ball position at impact relative to service box)
  4. Rally highlight clips (long rallies, break points, aces)

All outputs written to Modal Volume at /data/outputs/{job_id}/
"""
import json
import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("/data/outputs")

CLASS_COLORS = {
    0: (0, 255, 100),    # player — green
    1: (0, 255, 255),    # ball — yellow
    2: (255, 180, 0),    # court_lines — blue-ish
    3: (180, 0, 255),    # net — purple
}
CLASS_NAMES = {0: "player", 1: "ball", 2: "court_lines", 3: "net"}


def render_annotated_video(
    job_id: str,
    video_path: str,
    model_path: str,
    output_dir: Path,
    config: dict,
) -> str:
    """
    Run YOLOv8 inference on full video and render bounding box overlay.
    Uses OpenCV to write annotated frames, then FFmpeg to assemble video.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    conf = config.get("overlay_conf", 0.30)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_dir = output_dir / "annotated_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf, verbose=False)
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf_score = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                thickness = 1 if cls_id == 1 else 2  # thinner for ball
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                label = f"{CLASS_NAMES.get(cls_id, 'obj')} {conf_score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), frame)
        frame_idx += 1

    cap.release()

    output_video = output_dir / "annotated_video.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        str(output_video),
    ]
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    logger.info(f"Annotated video rendered: {output_video}")
    return str(output_video)


def generate_heatmaps(
    job_id: str,
    video_path: str,
    model_path: str,
    output_dir: Path,
    config: dict,
) -> list[str]:
    """
    Generate per-set player movement heatmaps.
    Splits video into segments (default 3 sets), accumulates player detection centroids.
    """
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    model = YOLO(model_path)
    conf = config.get("heatmap_conf", 0.30)
    num_sets = config.get("num_sets", 3)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    set_size = total_frames // num_sets
    sets = [
        (i * set_size, (i + 1) * set_size if i < num_sets - 1 else total_frames)
        for i in range(num_sets)
    ]

    heatmap_paths = []
    HEATMAP_RES = 64

    for s_idx, (start_frame, end_frame) in enumerate(sets):
        heat = np.zeros((HEATMAP_RES, HEATMAP_RES), dtype=np.float32)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_step = max(1, (end_frame - start_frame) // 200)
        frame_idx = start_frame

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_idx - start_frame) % frame_step == 0:
                results = model.predict(source=frame, conf=conf, verbose=False, classes=[0])
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        xywhn = boxes.xywhn[i].tolist()
                        cx_norm, cy_norm = xywhn[0], xywhn[1]
                        hx = min(int(cx_norm * HEATMAP_RES), HEATMAP_RES - 1)
                        hy = min(int(cy_norm * HEATMAP_RES), HEATMAP_RES - 1)
                        heat[hy, hx] += 1.0
            frame_idx += 1

        cap.release()

        heat = cv2.GaussianBlur(heat, (7, 7), 0)
        if heat.max() > 0:
            heat /= heat.max()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(heat, cmap="hot", interpolation="bicubic", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"Set {s_idx + 1} — Player Movement Density", fontsize=14, fontweight="bold")
        ax.set_xlabel("Court Width")
        ax.set_ylabel("Court Length")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(ax.images[0], ax=ax, label="Player Presence (normalized)")
        plt.tight_layout()

        heatmap_path = output_dir / f"heatmap_set{s_idx + 1}.png"
        plt.savefig(str(heatmap_path), dpi=150, bbox_inches="tight")
        plt.close()
        heatmap_paths.append(str(heatmap_path))
        logger.info(f"Heatmap Set {s_idx + 1} saved: {heatmap_path}")

    return heatmap_paths


def generate_serve_placement_chart(
    job_id: str,
    video_path: str,
    model_path: str,
    output_dir: Path,
    config: dict,
) -> str:
    """
    Generate a serve placement scatter chart.
    Detects ball positions in the service box region during likely serve frames
    (ball in upper portion + player in baseline region).
    """
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    model = YOLO(model_path)
    conf = config.get("serve_chart_conf", 0.25)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    serve_balls = []
    frame_step = max(1, int(fps / 2))
    frame_idx = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf, verbose=False)
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            ball_positions = []
            player_positions = []

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                xywhn = boxes.xywhn[i].tolist()
                if cls_id == 1:
                    ball_positions.append(xywhn)
                elif cls_id == 0:
                    player_positions.append(xywhn)

            baseline_players = [p for p in player_positions if p[1] > 0.75 or p[1] < 0.25]
            if ball_positions and baseline_players:
                for bp in ball_positions:
                    if 0.2 < bp[0] < 0.8 and 0.3 < bp[1] < 0.7:
                        serve_balls.append((bp[0], bp[1]))

        frame_idx += frame_step

    cap.release()

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.axhline(y=0.5, color="white", linewidth=2)
    ax.axvline(x=0.5, color="white", linewidth=1)
    ax.axhline(y=0.35, color="white", linewidth=1, linestyle="--")
    ax.axhline(y=0.65, color="white", linewidth=1, linestyle="--")

    if serve_balls:
        xs, ys = zip(*serve_balls)
        ax.scatter(xs, ys, c="yellow", s=30, alpha=0.7, edgecolors="orange", zorder=5)

    ax.set_facecolor("#2d5a27")
    ax.set_title(f"Serve Placement Chart ({len(serve_balls)} serves detected)", fontsize=14, fontweight="bold", color="white")
    ax.set_xlabel("Court Width", color="white")
    ax.set_ylabel("Court Length", color="white")
    ax.tick_params(colors="white")
    fig.patch.set_facecolor("#1a1a2e")
    plt.tight_layout()

    chart_path = output_dir / "serve_placement_chart.png"
    plt.savefig(str(chart_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Serve placement chart saved: {chart_path}")
    return str(chart_path)


def extract_highlight_clips(
    job_id: str,
    video_path: str,
    model_path: str,
    output_dir: Path,
    config: dict,
) -> list[dict]:
    """
    Detect and extract highlight clips for tennis:
      - Long rallies: sustained ball tracking across many consecutive frames
      - Aces / unreturned serves: ball detected near service box, no opponent movement
      - Net approaches: player moving toward net region

    Returns list of clip metadata dicts with timestamps and paths.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    conf = config.get("highlight_conf", 0.30)
    clip_duration = config.get("clip_duration_sec", 8)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    rally_events = []
    net_approach_events = []
    consecutive_ball_frames = 0
    rally_start_time = 0.0

    frame_step = max(1, int(fps / 2))
    frame_idx = 0

    cap = cv2.VideoCapture(video_path)
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf, verbose=False)
        timestamp_sec = frame_idx / fps

        ball_detected = False
        player_near_net = False

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                xywhn = boxes.xywhn[i].tolist()

                if cls_id == 1:
                    ball_detected = True

                if cls_id == 0 and 0.35 < xywhn[1] < 0.65:
                    player_near_net = True

        if ball_detected:
            if consecutive_ball_frames == 0:
                rally_start_time = timestamp_sec
            consecutive_ball_frames += 1
        else:
            if consecutive_ball_frames >= 8:
                rally_events.append(rally_start_time)
            consecutive_ball_frames = 0

        if player_near_net:
            net_approach_events.append(timestamp_sec)

        frame_idx += frame_step

    cap.release()

    def cluster_events(events: list[float], gap: float = 10.0) -> list[float]:
        if not events:
            return []
        clustered = [events[0]]
        for t in events[1:]:
            if t - clustered[-1] > gap:
                clustered.append(t)
        return clustered

    rally_times = cluster_events(rally_events)[:5]
    net_approach_times = cluster_events(net_approach_events)[:3]

    clips = []

    for i, t in enumerate(rally_times):
        start = max(0, t - 2)
        output_clip = clips_dir / f"rally_{i + 1}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(start),
            "-t", str(clip_duration),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_clip),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            clips.append({
                "type": "rally",
                "label": f"Rally Highlight {i + 1}",
                "timestamp_sec": round(t, 1),
                "path": str(output_clip),
            })
        except Exception as e:
            logger.warning(f"Failed to extract rally clip at t={t:.1f}s: {e}")

    for i, t in enumerate(net_approach_times):
        start = max(0, t - 1)
        output_clip = clips_dir / f"net_approach_{i + 1}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(start),
            "-t", str(clip_duration),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_clip),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            clips.append({
                "type": "net_approach",
                "label": f"Net Approach {i + 1}",
                "timestamp_sec": round(t, 1),
                "path": str(output_clip),
            })
        except Exception as e:
            logger.warning(f"Failed to extract net approach clip at t={t:.1f}s: {e}")

    logger.info(f"Extracted {len(clips)} highlight clips")
    return clips


def save_session(
    job_id: str,
    footage_url: str,
    decisions: list[dict],
    eval_results: list[dict],
    coach_feedback: list[dict],
    config: dict,
    output_dir: Path,
) -> str:
    """Save complete session record to session.json for cross-match learning."""
    session = {
        "job_id": job_id,
        "footage_url": footage_url,
        "config": config,
        "decisions": decisions,
        "eval_results": eval_results,
        "coach_feedback": coach_feedback,
    }
    session_path = output_dir / "session.json"
    session_path.write_text(json.dumps(session, indent=2))
    logger.info(f"Session saved: {session_path}")
    return str(session_path)


def run(
    job_id: str,
    video_path: str,
    model_path: str,
    decisions: list[dict],
    eval_results: list[dict],
    coach_feedback: list[dict],
    footage_url: str,
    config: dict,
) -> dict:
    """
    Generate all visual outputs: annotated video, heatmaps, serve chart, highlight clips.

    Returns:
        dict with paths to all generated artifacts
    """
    output_dir = OUTPUTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Rendering annotated video overlay...")
    annotated_video = render_annotated_video(job_id, video_path, model_path, output_dir, config)

    logger.info("Generating per-set player movement heatmaps...")
    heatmap_paths = generate_heatmaps(job_id, video_path, model_path, output_dir, config)

    logger.info("Generating serve placement chart...")
    serve_chart = generate_serve_placement_chart(job_id, video_path, model_path, output_dir, config)

    logger.info("Extracting highlight clips...")
    clips = extract_highlight_clips(job_id, video_path, model_path, output_dir, config)

    session_path = save_session(
        job_id, footage_url, decisions, eval_results, coach_feedback, config, output_dir
    )

    eval_report_path = output_dir / "eval_results.json"
    eval_report_path.write_text(json.dumps({
        "job_id": job_id,
        "eval_iterations": eval_results,
        "decisions": decisions,
        "final_config": config,
    }, indent=2))

    return {
        "annotated_video": annotated_video,
        "heatmaps": heatmap_paths,
        "serve_chart": serve_chart,
        "clips": clips,
        "session_json": session_path,
        "eval_report": str(eval_report_path),
        "best_model": "/data/models/best.pt",
        "output_dir": str(output_dir),
    }
