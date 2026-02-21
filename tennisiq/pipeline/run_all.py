from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from tennisiq.io.export import export_json, export_jsonl
from tennisiq.io.video import normalize_video_fps_moviepy, read_video, write_video
from tennisiq.pipeline.step_01_court import run_step_01_court
from tennisiq.pipeline.step_02_ball import run_step_02_ball
from tennisiq.pipeline.step_03_players import run_step_03_players
from tennisiq.pipeline.step_04_join_frames import run_step_04_join_frames
from tennisiq.pipeline.step_05_map_and_points import run_step_05_map_and_points
from tennisiq.pipeline.step_06_export_clips import run_step_06_export_clips


TAXONOMY_VERSION = "v1_out_balllost_net_doublebounce"
COURT_LINE_PAIRS = [
    (0, 1),   # top baseline
    (2, 3),   # bottom baseline
    (0, 2),   # left doubles sideline
    (1, 3),   # right doubles sideline
    (4, 5),   # left singles sideline
    (6, 7),   # right singles sideline
    (8, 9),   # top service line
    (10, 11), # bottom service line
    (12, 13), # center service line
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--court-model", type=str, required=True)
    parser.add_argument("--ball-model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--normalize-fps", type=int, default=0, help="If > 0, preprocess input video with MoviePy to this FPS.")

    parser.add_argument("--player-model", type=str, default="yolov8n.pt")
    parser.add_argument("--player-backend", type=str, default="auto", choices=["artlabss", "yolo", "hog", "auto"])
    parser.add_argument("--player-conf", type=float, default=0.2)
    parser.add_argument("--player-iou", type=float, default=0.5)
    parser.add_argument("--player-tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--no-player-fallback", action="store_true")
    parser.add_argument("--allow-player-model-download", action="store_true")
    parser.add_argument("--court-use-refine-kps", action="store_true")
    parser.add_argument("--no-court-stabilization", action="store_true")
    parser.add_argument("--court-homography-min-confidence", type=float, default=0.24)
    parser.add_argument("--court-homography-carry-frames", type=int, default=5)

    parser.add_argument("--event-model-path", type=str, default=None)
    parser.add_argument("--event-threshold", type=float, default=0.5)
    parser.add_argument("--line-margin-px", type=float, default=12.0)
    parser.add_argument("--serve-speed-thresh", type=float, default=600.0)
    parser.add_argument("--inactivity-frames", type=int, default=24)
    parser.add_argument("--ball-lost-frames", type=int, default=12)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def _valid_point(point: Tuple[float | None, float | None]) -> bool:
    return point[0] is not None and point[1] is not None


def _smooth_ball_track(ball_track: List[Tuple[float | None, float | None]]) -> List[Tuple[float | None, float | None]]:
    # Median filter for jitter + EMA for smooth visual motion.
    if not ball_track:
        return []

    median_track: List[Tuple[float | None, float | None]] = []
    half_window = 2
    for idx in range(len(ball_track)):
        lo = max(0, idx - half_window)
        hi = min(len(ball_track), idx + half_window + 1)
        xs = [ball_track[j][0] for j in range(lo, hi) if _valid_point(ball_track[j])]
        ys = [ball_track[j][1] for j in range(lo, hi) if _valid_point(ball_track[j])]
        if xs and ys:
            median_track.append((float(np.median(xs)), float(np.median(ys))))
        else:
            median_track.append((None, None))

    smoothed: List[Tuple[float | None, float | None]] = []
    prev: Tuple[float | None, float | None] = (None, None)
    alpha = 0.60
    max_jump_px = 180.0

    for point in median_track:
        if not _valid_point(point):
            smoothed.append((None, None))
            prev = (None, None)
            continue

        x, y = float(point[0]), float(point[1])
        if _valid_point(prev):
            px, py = float(prev[0]), float(prev[1])
            if float(np.hypot(x - px, y - py)) > max_jump_px:
                sx, sy = x, y
            else:
                sx = alpha * x + (1.0 - alpha) * px
                sy = alpha * y + (1.0 - alpha) * py
        else:
            sx, sy = x, y

        smoothed.append((float(sx), float(sy)))
        prev = smoothed[-1]

    return smoothed


def _draw_ball_trail(frame, trail: List[Tuple[float, float]], color=(0, 255, 255)):
    if len(trail) < 2:
        return frame

    out = frame
    segments = len(trail) - 1
    for idx in range(segments):
        x1, y1 = trail[idx]
        x2, y2 = trail[idx + 1]
        if float(np.hypot(x2 - x1, y2 - y1)) > 220.0:
            continue
        t = (idx + 1) / max(1, segments)
        thickness = max(1, int(1 + 4 * t))
        seg_color = (
            int(0.6 * color[0]),
            int(90 + 0.65 * color[1] * t),
            int(70 + 0.75 * color[2] * t),
        )
        out = cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), seg_color, thickness, cv2.LINE_AA)
    return out


def _draw_overlay(frame, row, ball_point=None, ball_trail=None):
    out = frame.copy()

    court_keypoints = row.get("court_keypoints", [])
    for i, j in COURT_LINE_PAIRS:
        if i >= len(court_keypoints) or j >= len(court_keypoints):
            continue
        p1 = court_keypoints[i]
        p2 = court_keypoints[j]
        if not _valid_point(p1) or not _valid_point(p2):
            continue
        out = cv2.line(
            out,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    for pt in court_keypoints:
        if not _valid_point(pt):
            continue
        out = cv2.circle(out, (int(pt[0]), int(pt[1])), radius=2, color=(0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    if ball_point is None:
        ball_point = row.get("ball_xy", (None, None))

    if ball_trail:
        out = _draw_ball_trail(out, ball_trail, color=(0, 255, 255))

    bx, by = ball_point
    if bx is not None and by is not None:
        inout = row.get("ball_inout", "unknown")
        if inout == "in":
            color = (0, 255, 0)
        elif inout == "line":
            color = (0, 255, 255)
        elif inout == "out":
            color = (0, 0, 255)
        else:
            color = (200, 200, 200)

        glow = out.copy()
        glow = cv2.circle(glow, (int(bx), int(by)), radius=7, color=color, thickness=-1, lineType=cv2.LINE_AA)
        out = cv2.addWeighted(glow, 0.30, out, 0.70, 0.0)
        out = cv2.circle(out, (int(bx), int(by)), radius=4, color=color, thickness=-1, lineType=cv2.LINE_AA)
        out = cv2.circle(out, (int(bx), int(by)), radius=8, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    for name, color in [("playerA_bbox", (255, 0, 0)), ("playerB_bbox", (0, 165, 255))]:
        bbox = row.get(name)
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        out = cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    return out


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = args.video
    if args.normalize_fps > 0:
        normalized_input = output_dir / f"input_{args.normalize_fps}fps.mp4"
        video_path = normalize_video_fps_moviepy(args.video, str(normalized_input), target_fps=args.normalize_fps)

    frames, fps = read_video(video_path)

    court_points = run_step_01_court(
        frames,
        args.court_model,
        use_refine_kps=args.court_use_refine_kps,
        stabilize_homography=not args.no_court_stabilization,
        homography_min_confidence=args.court_homography_min_confidence,
        homography_carry_frames=args.court_homography_carry_frames,
        device=args.device,
    )
    ball_track = run_step_02_ball(frames, args.ball_model, extrapolation=True, device=args.device)
    players = run_step_03_players(
        frames,
        court_points=court_points,
        model_path=args.player_model,
        conf=args.player_conf,
        iou=args.player_iou,
        tracker=args.player_tracker,
        fallback_hog=not args.no_player_fallback,
        allow_model_download=args.allow_player_model_download,
        backend=args.player_backend,
        device=args.device,
    )

    records = run_step_04_join_frames(court_points, ball_track, players, fps=fps)
    mapped = run_step_05_map_and_points(
        records,
        fps=fps,
        event_model_path=args.event_model_path,
        event_threshold=args.event_threshold,
        line_margin_px=args.line_margin_px,
        serve_speed_thresh=args.serve_speed_thresh,
        inactivity_frames=args.inactivity_frames,
        ball_lost_frames=args.ball_lost_frames,
    )

    mapped_frames = mapped["frames"]
    ball_track_smoothed = _smooth_ball_track([tuple(row.get("ball_xy", (None, None))) for row in mapped_frames])

    overlay_frames = []
    trail = deque(maxlen=14)
    for idx, frame in enumerate(tqdm(frames[: len(mapped_frames)], desc="Overlay rendering", unit="frame")):
        ball_point = ball_track_smoothed[idx]
        if _valid_point(ball_point):
            trail.append((float(ball_point[0]), float(ball_point[1])))
        else:
            trail.clear()
        overlay_frames.append(_draw_overlay(frame, mapped_frames[idx], ball_point=ball_point, ball_trail=list(trail)))

    clips_dir, clip_meta = run_step_06_export_clips(str(output_dir), overlay_frames, mapped["points"], fps=fps)

    export_jsonl(str(output_dir / "frames.jsonl"), mapped_frames)
    export_json(str(output_dir / "tracks.json"), mapped["tracks"])
    export_json(
        str(output_dir / "points.json"),
        {
            "taxonomy_version": TAXONOMY_VERSION,
            "points": mapped["points"],
            "clips": clip_meta,
        },
    )
    export_json(
        str(output_dir / "insights.json"),
        {
            "stats": mapped["stats"],
            "insights": mapped["insights"],
            "visuals": mapped["visuals"],
            "point_cards": mapped.get("point_cards", []),
        },
    )
    export_json(
        str(output_dir / "meta.json"),
        {
            "fps": fps,
            "num_frames": len(mapped_frames),
            "clips_dir": clips_dir,
            "taxonomy_version": TAXONOMY_VERSION,
        },
    )

    write_video(overlay_frames, fps, str(output_dir / "overlay.mp4"), codec="mp4v")


if __name__ == "__main__":
    main()
