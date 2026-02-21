from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from tennisiq.io.export import export_json, export_jsonl
from tennisiq.io.video import normalize_video_fps_moviepy, read_video, write_video
from tennisiq.pipeline.step_01_court import run_step_01_court
from tennisiq.pipeline.step_02_ball import run_step_02_ball
from tennisiq.pipeline.step_03_players import run_step_03_players
from tennisiq.pipeline.step_04_join_frames import run_step_04_join_frames
from tennisiq.pipeline.step_05_map_and_points import run_step_05_map_and_points
from tennisiq.pipeline.step_06_export_clips import run_step_06_export_clips


TAXONOMY_VERSION = "v1_out_balllost_net_doublebounce"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--court-model", type=str, required=True)
    parser.add_argument("--ball-model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--normalize-fps", type=int, default=0, help="If > 0, preprocess input video with MoviePy to this FPS.")

    parser.add_argument("--player-model", type=str, default="yolov8n.pt")
    parser.add_argument("--player-conf", type=float, default=0.2)
    parser.add_argument("--player-iou", type=float, default=0.5)
    parser.add_argument("--player-tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--no-player-fallback", action="store_true")
    parser.add_argument("--allow-player-model-download", action="store_true")

    parser.add_argument("--event-model-path", type=str, default=None)
    parser.add_argument("--event-threshold", type=float, default=0.5)
    parser.add_argument("--line-margin-px", type=float, default=12.0)
    parser.add_argument("--serve-speed-thresh", type=float, default=600.0)
    parser.add_argument("--inactivity-frames", type=int, default=24)
    parser.add_argument("--ball-lost-frames", type=int, default=12)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def _draw_overlay(frame, row):
    out = frame.copy()

    for pt in row.get("court_keypoints", []):
        if pt[0] is None or pt[1] is None:
            continue
        out = cv2.circle(out, (int(pt[0]), int(pt[1])), radius=0, color=(255, 255, 0), thickness=4)

    bx, by = row.get("ball_xy", (None, None))
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
        out = cv2.circle(out, (int(bx), int(by)), radius=0, color=color, thickness=8)

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

    court_points = run_step_01_court(frames, args.court_model, device=args.device)
    ball_track = run_step_02_ball(frames, args.ball_model, extrapolation=True, device=args.device)
    players = run_step_03_players(
        frames,
        model_path=args.player_model,
        conf=args.player_conf,
        iou=args.player_iou,
        tracker=args.player_tracker,
        fallback_hog=not args.no_player_fallback,
        allow_model_download=args.allow_player_model_download,
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
    overlay_frames = [_draw_overlay(frame, mapped_frames[idx]) for idx, frame in enumerate(frames[: len(mapped_frames)])]

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
