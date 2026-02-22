"""
Modal GPU function for the full TennisIQ inference pipeline.

Usage:
    python -m modal run tennisiq/modal_court.py --video-path <path> --start-sec 60 --end-sec 70
"""
import json
import time
import modal

app = modal.App("tennisiq-court")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "opencv-python-headless>=4.10.0",
        "numpy>=1.26.0",
        "scipy>=1.14.0",
        "ultralytics>=8.3.0",
        "lap>=0.4.0",
    )
    .add_local_dir("checkpoints", remote_path="/root/checkpoints")
    .add_local_dir("tennisiq", remote_path="/root/tennisiq")
)


@app.function(gpu="T4", timeout=600, image=image)
def run_court_and_ball(
    video_bytes: bytes,
    fps: float,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    court_batch_size: int = 32,
) -> dict:
    """
    Full TennisIQ inference + analytics pipeline on GPU.

    Phase 1:  ResNet50 court keypoint regression → 14 keypoints per frame
    Phase 2:  Homography computation + confidence scoring
    Phase 3:  TrackNetV3 ball detection per frame
    Phase 4:  Court-space projection + speed/acceleration
    Phase 5:  YOLOv8n + ByteTrack → player detection, court filter, A/B assignment
    Phase 6:  Shot detection via ball velocity reversal + ownership assignment
    Phase 6b: Pose-first shot type classification (YOLOv8-pose + fallback)
    Phase 6c: Event detection — bounces + hits with in/out classification
    Phase 6d: Point segmentation
    Phase 7:  Match analytics + coaching intelligence
    Phase 8:  Structured output files
    Phase 9:  Overlay video (FR-35) + per-point clips (FR-39)
    """
    import os
    import sys
    sys.path.insert(0, "/root")

    import tempfile
    import cv2
    import numpy as np
    import torch

    from tennisiq.cv.court.inference_resnet import CourtDetectorResNet
    from tennisiq.cv.ball.inference_tracknet import BallDetectorTrackNet
    from tennisiq.cv.ball.inference import compute_ball_physics, clean_ball_track
    from tennisiq.cv.players.inference import PlayerDetector
    from tennisiq.geometry.homography import compute_homographies
    from tennisiq.analytics.events import detect_events
    from tennisiq.analytics.shots import detect_shots, classify_shot_direction
    from tennisiq.analytics.shot_classifier import classify_shot_type
    from tennisiq.analytics.shot_classifier_pose import classify_shot_from_pose
    from tennisiq.analytics.points import segment_points

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    ckpt_root = os.environ.get("TENNISIQ_CHECKPOINT_ROOT", "/root/checkpoints")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        tmp_path = f.name

    # ── Phase 1: Court keypoint detection (ResNet50) ─────────────────────
    court_det = CourtDetectorResNet(
        os.path.join(ckpt_root, "court_resnet", "keypoints_model.pth"), device=device,
    )

    t0 = time.time()

    def court_progress(done, total):
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  [court-resnet] {done}/{total} frames ({rate:.0f} fps)")

    all_kps = court_det.predict_video(
        tmp_path, start_sec=start_sec, end_sec=end_sec,
        batch_size=court_batch_size, progress_callback=court_progress,
    )
    t_court = time.time() - t0
    print(f"  Court detection (ResNet50): {len(all_kps)} frames in {t_court:.1f}s")

    # ── Phase 2: Homography ──────────────────────────────────────────────
    t1 = time.time()
    homographies = compute_homographies(all_kps)
    t_hom = time.time() - t1
    print(f"  Homography: {t_hom:.2f}s")

    # ── Phase 3: Ball detection (TrackNetV3) ──────────────────────────────
    ball_det = BallDetectorTrackNet(
        os.path.join(ckpt_root, "ball_tracknet", "model_best.pt"), device=device,
    )

    t2 = time.time()

    def ball_progress(done, total):
        elapsed = time.time() - t2
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  [tracknetv3] {done}/{total} frames ({rate:.0f} fps)")

    ball_track_raw, ball_detections_raw = ball_det.detect_video(
        tmp_path, start_sec=start_sec, end_sec=end_sec,
        progress_callback=ball_progress,
    )
    t_ball = time.time() - t2
    raw_det = sum(1 for x, y in ball_track_raw if x is not None)
    print(f"  Ball detection (TrackNetV3): {len(ball_track_raw)} frames in {t_ball:.1f}s")
    print(f"    Raw detections: {raw_det}/{len(ball_track_raw)}")

    # ── Phase 3b: Ball track cleaning (outlier removal + interpolation) ──
    ball_track = clean_ball_track(ball_track_raw)
    clean_det = sum(1 for x, y in ball_track if x is not None)
    print(f"    After cleaning: {clean_det}/{len(ball_track)} ({clean_det - raw_det:+d} from interpolation)")

    # ── Phase 4: Court-space projection + physics ────────────────────────
    t3 = time.time()
    ball_physics = compute_ball_physics(
        ball_track, homographies=homographies, fps=fps,
    )
    t_phys = time.time() - t3

    # ── Phase 5: Player detection ────────────────────────────────────────
    player_det = PlayerDetector(device=device)

    t4 = time.time()

    def player_progress(done, total):
        elapsed = time.time() - t4
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  [players] {done}/{total} frames ({rate:.0f} fps)")

    player_results = player_det.detect_video(
        tmp_path, homographies=homographies,
        start_sec=start_sec, end_sec=end_sec,
        progress_callback=player_progress,
    )
    t_players = time.time() - t4
    print(f"  Player detection: {len(player_results)} frames in {t_players:.1f}s")

    # ── Phase 6: Shot detection via velocity reversal + ownership ─────────
    t_shots_start = time.time()
    shot_events = detect_shots(
        ball_physics=ball_physics,
        player_results=player_results,
        fps=fps,
        start_sec=start_sec,
    )
    t_shots = time.time() - t_shots_start
    n_a = sum(1 for s in shot_events if s.owner == "player_a")
    n_b = sum(1 for s in shot_events if s.owner == "player_b")
    print(f"  Shot detection: {len(shot_events)} shots (A={n_a}, B={n_b}) in {t_shots:.3f}s")

    # ── Phase 6b: Pose-first shot type classification ─────────────────────
    t_class_start = time.time()
    shot_directions = {}
    cap_cls = cv2.VideoCapture(tmp_path)
    frame_cache: dict[int, np.ndarray] = {}

    for i, shot in enumerate(shot_events):
        pose_result = None
        fi = int(shot.frame_idx)
        if 0 <= fi < len(player_results):
            pr = player_results[fi]
            player_det = pr.player_a if shot.owner == "player_a" else pr.player_b
            if player_det is not None:
                if fi in frame_cache:
                    frame = frame_cache[fi]
                else:
                    cap_cls.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ok, frame = cap_cls.read()
                    if ok and frame is not None:
                        frame_cache[fi] = frame
                    else:
                        frame = None
                if frame is not None:
                    pose_result = classify_shot_from_pose(frame, player_det.bbox)

        if pose_result is not None and pose_result.confidence >= 0.6 and pose_result.shot_type != "neutral":
            shot.shot_type = pose_result.shot_type
            shot.shot_type_confidence = pose_result.confidence
        else:
            # Fallback to trajectory classifier when pose is low-confidence.
            result = classify_shot_type(shot, i, shot_events, ball_physics)
            shot.shot_type = result.shot_type
            shot.shot_type_confidence = result.confidence

        shot_directions[shot.frame_idx] = classify_shot_direction(shot, shot.court_side or "near")

    cap_cls.release()

    t_class = time.time() - t_class_start
    type_counts = {}
    for s in shot_events:
        type_counts[s.shot_type] = type_counts.get(s.shot_type, 0) + 1
    print(f"  Shot classification: {type_counts} in {t_class:.3f}s")

    # ── Phase 6c: Event detection (bounces + hits) ────────────────────────
    t5_events = time.time()
    events = detect_events(
        ball_physics=ball_physics,
        player_results=player_results,
        fps=fps,
        start_sec=start_sec,
    )
    t_events = time.time() - t5_events
    n_bounces = sum(1 for e in events if e.event_type == "bounce")
    n_hits = sum(1 for e in events if e.event_type == "hit")
    print(f"  Event detection: {len(events)} events ({n_bounces} bounces, {n_hits} hits) in {t_events:.2f}s")

    # ── Phase 6d: Point segmentation ──────────────────────────────────────
    t6_points = time.time()
    points = segment_points(
        events=events,
        fps=fps,
        total_frames=len(all_kps),
        start_sec=start_sec,
        homographies=homographies,
    )
    t_points = time.time() - t6_points
    print(f"  Point segmentation: {len(points)} points in {t_points:.3f}s")
    for pt in points:
        print(f"    Point {pt.point_idx}: frames {pt.start_frame}-{pt.end_frame}, "
              f"{pt.rally_hit_count} hits, {pt.bounce_count} bounces, "
              f"end={pt.end_reason}, serve_zone={pt.serve_zone}")

    # ── Phase 7: Match analytics + coaching intelligence ──────────────────
    from tennisiq.analytics.match_analytics import compute_match_analytics
    from tennisiq.analytics.coaching_intelligence import generate_coaching_intelligence

    t_analytics_start = time.time()
    analytics = compute_match_analytics(
        shot_events=shot_events,
        shot_directions=shot_directions,
        points=points,
        events=events,
        ball_physics=ball_physics,
        player_results=player_results,
        fps=fps,
    )
    coaching = generate_coaching_intelligence(analytics, points, shot_events, shot_directions)
    t_analytics = time.time() - t_analytics_start
    print(f"  Match analytics + coaching: {t_analytics:.3f}s")

    total_elapsed = time.time() - t0
    n = len(all_kps)

    # ── Phase 8: Write structured outputs ─────────────────────────────────
    from tennisiq.io.output import write_outputs

    timing_data = {
        "court_sec": round(t_court, 2),
        "homography_sec": round(t_hom, 2),
        "ball_sec": round(t_ball, 2),
        "physics_sec": round(t_phys, 3),
        "player_sec": round(t_players, 2),
        "shots_sec": round(t_shots, 3),
        "classification_sec": round(t_class, 3),
        "events_sec": round(t_events, 3),
        "points_sec": round(t_points, 3),
        "analytics_sec": round(t_analytics, 3),
        "total": round(total_elapsed, 2),
    }

    out_dir = tempfile.mkdtemp()
    job_id = f"run_{int(start_sec)}s_{int(end_sec)}s"

    output_paths = write_outputs(
        job_id=job_id,
        output_dir=out_dir,
        video_path=tmp_path,
        fps=fps,
        start_sec=start_sec,
        end_sec=end_sec,
        ball_physics=ball_physics,
        homographies=homographies,
        player_results=player_results,
        court_keypoints=all_kps,
        timing=timing_data,
        events=events,
        points=points,
        shot_events=shot_events,
        shot_directions=shot_directions,
        analytics=analytics,
        coaching=coaching,
    )
    print(f"  Structured outputs written to {out_dir}/{job_id}/")

    # ── Phase 9: Render overlay video + extract point clips ─────────────
    from tennisiq.io.visualize import render_overlay_video, extract_point_clips

    t5 = time.time()
    overlay_path = os.path.join(out_dir, job_id, "visuals", "overlay.mp4")

    def overlay_progress(done, total):
        print(f"  [overlay] {done}/{total} frames")

    render_overlay_video(
        video_path=tmp_path,
        output_path=overlay_path,
        homographies=homographies,
        court_keypoints=all_kps,
        ball_positions=ball_physics,
        player_results=player_results,
        fps=fps,
        start_sec=start_sec,
        end_sec=end_sec,
        ball_detections_yolo=ball_detections_raw,
        events=events,
        draw_pose=False,
        progress_callback=overlay_progress,
    )
    t_overlay = time.time() - t5
    timing_data["overlay_sec"] = round(t_overlay, 2)
    print(f"  Overlay video: {t_overlay:.1f}s")

    # FR-39: Extract per-point video clips
    if points:
        t_clips = time.time()
        clips_dir = os.path.join(out_dir, job_id, "clips")
        clip_paths = extract_point_clips(
            video_path=tmp_path,
            output_dir=clips_dir,
            points=points,
            fps=fps,
            start_sec=start_sec,
        )
        timing_data["clips_sec"] = round(time.time() - t_clips, 2)
        print(f"  Point clips: {len(clip_paths)} clips in {timing_data['clips_sec']:.1f}s")

    timing_data["total"] = round(time.time() - t0, 2)

    base_dir = os.path.join(out_dir, job_id)
    output_files = {}
    video_files = {}
    for dirpath, _, filenames in os.walk(base_dir):
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, base_dir)
            if rel.endswith(".mp4"):
                with open(full, "rb") as fh:
                    video_files[rel] = fh.read()
            else:
                with open(full) as fh:
                    output_files[rel] = fh.read()

    return {
        "job_id": job_id,
        "frames_processed": n,
        "timing": timing_data,
        "output_files": output_files,
        "video_files": video_files,
    }


@app.local_entrypoint()
def main(
    video_path: str = r"C:\Users\badri\Downloads\tennistest.mp4",
    start_sec: float = 60.0,
    end_sec: float = 70.0,
):
    """Local entrypoint — reads video, sends to GPU, writes structured outputs locally."""
    import cv2
    from pathlib import Path

    print(f"Reading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"  FPS: {fps}, Total frames: {total}, Duration: {total/fps:.1f}s")
    print(f"  Processing segment: {start_sec}s - {end_sec}s")

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    print(f"  Video size: {len(video_bytes) / 1024 / 1024:.1f} MB")
    print(f"  Uploading to Modal GPU...\n")

    result = run_court_and_ball.remote(
        video_bytes=video_bytes,
        fps=fps,
        start_sec=start_sec,
        end_sec=end_sec,
    )

    job_id = result["job_id"]
    timing = result["timing"]
    n = result["frames_processed"]

    local_out = Path("outputs") / job_id
    local_out.mkdir(parents=True, exist_ok=True)

    for rel_path, content in result["output_files"].items():
        fp = local_out / rel_path
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")

    video_files = result.get("video_files", {})
    for rel_path, vbytes in video_files.items():
        fp = local_out / rel_path
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(vbytes)

    total_video_mb = sum(len(v) for v in video_files.values()) / 1024 / 1024
    if video_files:
        print(f"  Video files: {len(video_files)} ({total_video_mb:.1f} MB total)")

    print(f"\n{'='*60}")
    print(f"  TENNISIQ PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Job ID:          {job_id}")
    print(f"  Frames:          {n}")
    print(f"  Total time:      {timing['total']}s on GPU")
    print(f"{'─'*60}")
    print(f"  OUTPUT FILES ({local_out}):")
    for rel_path in sorted(result["output_files"].keys()):
        size = len(result["output_files"][rel_path])
        print(f"    {rel_path:<35} {size:>8,} bytes")
    for rel_path in sorted(video_files.keys()):
        size = len(video_files[rel_path])
        print(f"    {rel_path:<35} {size:>8,} bytes (video)")
    print(f"{'─'*60}")

    stats_content = result["output_files"].get("stats.json")
    if stats_content:
        stats = json.loads(stats_content)
        if "insights" in stats:
            print(f"  INSIGHTS:")
            for i, insight in enumerate(stats["insights"], 1):
                print(f"    {i}. {insight}")
    print(f"{'='*60}")
