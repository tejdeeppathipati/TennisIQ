from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from tennisiq.analytics.insights import summarize_insights
from tennisiq.analytics.stats import basic_match_stats
from tennisiq.events.bounces import detect_bounces
from tennisiq.events.hits import detect_hits
from tennisiq.io.export import export_json, export_jsonl
from tennisiq.io.video import read_video, write_video
from tennisiq.pipeline.step_01_court import run_step_01_court
from tennisiq.pipeline.step_02_ball import run_step_02_ball
from tennisiq.pipeline.step_03_players import run_step_03_players
from tennisiq.pipeline.step_04_join_frames import run_step_04_join_frames
from tennisiq.pipeline.step_05_map_and_points import run_step_05_map_and_points
from tennisiq.pipeline.step_06_export_clips import run_step_06_export_clips


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--court-model", type=str, required=True)
    parser.add_argument("--ball-model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, fps = read_video(args.video)

    court_points = run_step_01_court(frames, args.court_model, device=args.device)
    ball_track = run_step_02_ball(frames, args.ball_model, extrapolation=True, device=args.device)
    _players = run_step_03_players(frames)

    records = run_step_04_join_frames(court_points, ball_track)
    record_dicts = [r.to_dict() for r in records]
    enriched = run_step_05_map_and_points(record_dicts)

    clips_dir = run_step_06_export_clips(str(output_dir))

    # Overlay ball track for quick visual inspection.
    overlay_frames = []
    for idx, frame in enumerate(frames):
        out = frame.copy()
        if idx < len(ball_track):
            x, y = ball_track[idx]
            if x is not None and y is not None:
                out = cv2.circle(out, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=8)
        overlay_frames.append(out)

    export_jsonl(str(output_dir / "frames.jsonl"), enriched)
    export_json(str(output_dir / "tracks.json"), {"ball_track": ball_track})
    export_json(str(output_dir / "points.json"), {"court_points": court_points})
    export_json(str(output_dir / "meta.json"), {"clips_dir": clips_dir, "fps": fps})
    write_video(overlay_frames, fps, str(output_dir / "overlay.mp4"))

    bounces = detect_bounces(ball_track)
    hits = detect_hits(ball_track)
    stats = basic_match_stats(len(bounces), len(hits), len(frames))
    export_json(str(output_dir / "insights.json"), {"stats": stats, "summary": summarize_insights(stats)})


if __name__ == "__main__":
    main()
