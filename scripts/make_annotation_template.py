#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    points_path = run_dir / "points.json"
    if not points_path.exists():
        raise FileNotFoundError(f"Missing points.json in {run_dir}")

    payload = json.loads(points_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        points = payload
    else:
        points = payload.get("points", [])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in points:
            row = {
                "point_id": int(p.get("point_id", -1)),
                "pred_start_frame": p.get("start_frame"),
                "pred_end_frame": p.get("end_frame"),
                "pred_end_reason": p.get("end_reason"),
                "pred_bounce_frames": p.get("bounces", []),
                "pred_serve_zone": p.get("serve_zone"),
                "true_start_frame": None,
                "true_end_frame": None,
                "true_end_reason": None,
                "true_bounce_frames": [],
                "true_bounce_inout": [],
                "true_serve_in": None,
                "notes": "",
            }
            f.write(json.dumps(row) + "\n")

    print(f"Wrote template: {out_path}")


if __name__ == "__main__":
    main()
