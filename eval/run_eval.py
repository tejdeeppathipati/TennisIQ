#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _match_events_with_tolerance(pred: List[int], truth: List[int], tol: int = 3) -> Tuple[int, int, int]:
    pred = sorted(pred)
    truth = sorted(truth)
    used = [False] * len(truth)
    tp = 0
    for p in pred:
        found = False
        for i, t in enumerate(truth):
            if used[i]:
                continue
            if abs(p - t) <= tol:
                used[i] = True
                tp += 1
                found = True
                break
        if not found:
            pass
    fp = len(pred) - tp
    fn = len(truth) - tp
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int):
    eps = 1e-12
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    return p, r, f1


def evaluate(run_dir: Path, annotations_path: Path):
    points_payload = _read_json(run_dir / "points.json")
    frames = _read_jsonl(run_dir / "frames.jsonl")
    if isinstance(points_payload, list):
        points = points_payload
    else:
        points = points_payload.get("points", [])

    ann_rows = _read_jsonl(annotations_path)
    ann_by_id = {int(r["point_id"]): r for r in ann_rows}
    pred_by_id = {int(p["point_id"]): p for p in points}

    # Point boundary and end reason.
    start_abs_err = []
    end_abs_err = []
    end_reason_correct = 0
    end_reason_total = 0

    pred_bounces_all: List[int] = []
    true_bounces_all: List[int] = []

    serve_correct = 0
    serve_total = 0

    bounce_inout_correct = 0
    bounce_inout_total = 0

    for pid, ann in ann_by_id.items():
        pred = pred_by_id.get(pid)
        if pred is None:
            continue

        ts = ann.get("true_start_frame")
        te = ann.get("true_end_frame")
        tr = ann.get("true_end_reason")
        if ts is not None:
            start_abs_err.append(abs(int(pred.get("start_frame", 0)) - int(ts)))
        if te is not None:
            end_abs_err.append(abs(int(pred.get("end_frame", 0)) - int(te)))
        if tr is not None:
            end_reason_total += 1
            if str(pred.get("end_reason")) == str(tr):
                end_reason_correct += 1

        tb = [int(x) for x in ann.get("true_bounce_frames", [])]
        pb = [int(x) for x in pred.get("bounces", [])]
        true_bounces_all.extend(tb)
        pred_bounces_all.extend(pb)

        tsi = ann.get("true_serve_in")
        if tsi is not None:
            serve_total += 1
            pred_serve_in = pred.get("serve_zone") is not None
            if bool(pred_serve_in) == bool(tsi):
                serve_correct += 1

        for bi in ann.get("true_bounce_inout", []):
            frame_idx = int(bi.get("frame_idx", -1))
            label = str(bi.get("label", "")).lower()
            if 0 <= frame_idx < len(frames) and label:
                bounce_inout_total += 1
                pred_label = str(frames[frame_idx].get("ball_inout", "")).lower()
                if pred_label == label:
                    bounce_inout_correct += 1

    tp, fp, fn = _match_events_with_tolerance(pred_bounces_all, true_bounces_all, tol=3)
    b_prec, b_rec, b_f1 = _prf(tp, fp, fn)

    metrics = {
        "bounce": {
            "precision": b_prec,
            "recall": b_rec,
            "f1": b_f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "point_segmentation": {
            "start_mae_frames": (sum(start_abs_err) / len(start_abs_err)) if start_abs_err else None,
            "end_mae_frames": (sum(end_abs_err) / len(end_abs_err)) if end_abs_err else None,
            "end_reason_accuracy": (end_reason_correct / end_reason_total) if end_reason_total else None,
        },
        "inout": {
            "bounce_inout_accuracy": (bounce_inout_correct / bounce_inout_total) if bounce_inout_total else None,
        },
        "serve": {
            "serve_in_accuracy": (serve_correct / serve_total) if serve_total else None,
        },
    }
    return metrics


def _check_regression(metrics: Dict, baseline: Dict) -> List[str]:
    errs = []

    m_f1 = metrics["bounce"]["f1"]
    b_f1 = baseline.get("bounce", {}).get("f1")
    if b_f1 is not None and m_f1 < (b_f1 - 0.03):
        errs.append(f"Bounce F1 drop too large: {m_f1:.4f} vs baseline {b_f1:.4f}")

    m_end = metrics["point_segmentation"].get("end_reason_accuracy")
    b_end = baseline.get("point_segmentation", {}).get("end_reason_accuracy")
    if b_end is not None and m_end is not None and m_end < (b_end - 0.03):
        errs.append(f"End-reason accuracy drop too large: {m_end:.4f} vs baseline {b_end:.4f}")

    m_inout = metrics["inout"].get("bounce_inout_accuracy")
    b_inout = baseline.get("inout", {}).get("bounce_inout_accuracy")
    if b_inout is not None and m_inout is not None and m_inout < (b_inout - 0.02):
        errs.append(f"In/out accuracy drop too large: {m_inout:.4f} vs baseline {b_inout:.4f}")

    return errs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--annotations", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--baseline", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    metrics = evaluate(run_dir, Path(args.annotations))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    if args.baseline:
        baseline = _read_json(Path(args.baseline))
        errs = _check_regression(metrics, baseline)
        if errs:
            print("Regression gate failed:")
            for e in errs:
                print("-", e)
            raise SystemExit(3)


if __name__ == "__main__":
    main()
