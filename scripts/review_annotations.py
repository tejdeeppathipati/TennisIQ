#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_KEYS = {
    "point_id",
    "true_start_frame",
    "true_end_frame",
    "true_end_reason",
    "true_bounce_frames",
    "true_bounce_inout",
    "true_serve_in",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=str, required=True)
    p.add_argument("--num-frames", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.annotations)
    if not path.exists():
        raise FileNotFoundError(path)

    errors = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        row = json.loads(line)

        missing = REQUIRED_KEYS - row.keys()
        if missing:
            errors.append(f"line {i}: missing keys {sorted(missing)}")
            continue

        s = row.get("true_start_frame")
        e = row.get("true_end_frame")
        if s is not None and e is not None and s > e:
            errors.append(f"line {i}: true_start_frame > true_end_frame")

        if args.num_frames is not None:
            if s is not None and (s < 0 or s >= args.num_frames):
                errors.append(f"line {i}: true_start_frame out of range")
            if e is not None and (e < 0 or e >= args.num_frames):
                errors.append(f"line {i}: true_end_frame out of range")

        for b in row.get("true_bounce_frames", []):
            if args.num_frames is not None and (b < 0 or b >= args.num_frames):
                errors.append(f"line {i}: bounce frame {b} out of range")

    if errors:
        print("Annotation review failed:")
        for e in errors:
            print("-", e)
        raise SystemExit(2)

    print(f"Annotation review passed: {path}")


if __name__ == "__main__":
    main()
