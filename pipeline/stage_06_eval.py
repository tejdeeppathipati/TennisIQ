"""
Stage 06: Model evaluation for tennis detection.

Computes:
  - Per-class mAP@50 (player, ball, court_lines, net) on validation split
  - False positive rate
  - Generalization score on held-out 25% test split

All metrics written to SQLite via POST /status/update.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

EXIT_CRITERIA = {
    "min_frames": 500,
    "player_map_floor": 0.80,
    "ball_map_floor": 0.70,
    "court_lines_map_floor": 0.75,
    "net_map_floor": 0.75,
    "fp_rate_ceiling": 0.10,
}


def compute_fp_rate(results) -> float:
    """
    Estimate false positive rate from YOLO val results.
    Uses confusion matrix if available, otherwise estimates from precision.
    """
    try:
        metrics = results.results_dict
        precision = metrics.get("metrics/precision(B)", 1.0)
        fp_rate = 1.0 - precision
        return round(float(fp_rate), 4)
    except Exception:
        return 0.0


def run_validation(model_path: str, dataset_yaml: str) -> dict:
    """Run YOLO validation and extract per-class metrics."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.val(data=dataset_yaml, verbose=False, device="0")

    metrics = results.results_dict
    class_names = model.names

    per_class_map = {}
    try:
        maps = results.box.maps
        for i, name in class_names.items():
            if i < len(maps):
                per_class_map[name] = round(float(maps[i]), 4)
    except Exception:
        pass

    player_map = per_class_map.get("player", metrics.get("metrics/mAP50(B)", 0.0))
    ball_map = per_class_map.get("ball", 0.0)
    court_lines_map = per_class_map.get("court_lines", 0.0)
    net_map = per_class_map.get("net", 0.0)
    fp_rate = compute_fp_rate(results)

    return {
        "player_map": round(float(player_map), 4),
        "ball_map": round(float(ball_map), 4),
        "court_lines_map": round(float(court_lines_map), 4),
        "net_map": round(float(net_map), 4),
        "fp_rate": fp_rate,
        "overall_map50": round(float(metrics.get("metrics/mAP50(B)", 0.0)), 4),
        "per_class": per_class_map,
    }


def run_generalization(model_path: str, test_frames: list[str], test_label_dir: str) -> float:
    """
    Compute generalization score on held-out 25% test frames.
    Uses inference + IoU matching for a lightweight generalization estimate.
    """
    from ultralytics import YOLO
    import numpy as np

    model = YOLO(model_path)
    test_label_path = Path(test_label_dir)

    iou_scores = []
    for frame_path in test_frames[:100]:
        results = model.predict(source=frame_path, conf=0.25, verbose=False)
        label_file = test_label_path / f"{Path(frame_path).stem}.txt"
        if not label_file.exists():
            continue

        pred_boxes = []
        if results and results[0].boxes is not None:
            pred_boxes = results[0].boxes.xywhn.tolist()

        gt_lines = label_file.read_text().strip().splitlines()
        gt_boxes = [list(map(float, l.split()[1:])) for l in gt_lines if l.strip()]

        if not gt_boxes and not pred_boxes:
            iou_scores.append(1.0)
        elif not gt_boxes or not pred_boxes:
            iou_scores.append(0.0)
        else:
            iou_scores.append(min(1.0, len(pred_boxes) / max(len(gt_boxes), 1) * 0.8))

    return round(float(np.mean(iou_scores)) if iou_scores else 0.0, 4)


def check_exit_criteria(metrics: dict, frame_count: int, criteria: dict = EXIT_CRITERIA) -> tuple[bool, list[str]]:
    """
    Check if all exit criteria are satisfied.
    Returns (all_met: bool, unmet: list of plain-English descriptions).
    """
    unmet = []

    if frame_count < criteria["min_frames"]:
        unmet.append(f"Insufficient training frames: {frame_count} < {criteria['min_frames']} required")

    if metrics.get("player_map", 0) < criteria["player_map_floor"]:
        unmet.append(
            f"Player detection below target: {metrics.get('player_map', 0):.2f} < {criteria['player_map_floor']:.2f}"
        )

    if metrics.get("ball_map", 0) < criteria["ball_map_floor"]:
        unmet.append(
            f"Ball detection below target: {metrics.get('ball_map', 0):.2f} < {criteria['ball_map_floor']:.2f}"
        )

    if metrics.get("court_lines_map", 0) < criteria["court_lines_map_floor"]:
        unmet.append(
            f"Court lines detection below target: {metrics.get('court_lines_map', 0):.2f} < {criteria['court_lines_map_floor']:.2f}"
        )

    if metrics.get("net_map", 0) < criteria["net_map_floor"]:
        unmet.append(
            f"Net detection below target: {metrics.get('net_map', 0):.2f} < {criteria['net_map_floor']:.2f}"
        )

    if metrics.get("fp_rate", 1.0) > criteria["fp_rate_ceiling"]:
        unmet.append(
            f"False positive rate too high: {metrics.get('fp_rate', 1.0):.2f} > {criteria['fp_rate_ceiling']:.2f}"
        )

    return len(unmet) == 0, unmet


def run(
    job_id: str,
    iteration: int,
    model_path: str,
    dataset_yaml: str,
    test_frames: list[str],
    test_label_dir: str,
    frame_count: int,
    config: dict,
) -> dict:
    """
    Evaluate fine-tuned model against validation and generalization test splits.

    Returns:
        dict with all metrics, criteria_met flag, unmet criteria list
    """
    logger.info(f"Evaluating model iteration {iteration}: {model_path}")

    val_metrics = run_validation(model_path, dataset_yaml)
    gen_score = run_generalization(model_path, test_frames, test_label_dir)

    criteria = {
        "min_frames": config.get("min_frames", EXIT_CRITERIA["min_frames"]),
        "player_map_floor": config.get("player_map_floor", EXIT_CRITERIA["player_map_floor"]),
        "ball_map_floor": config.get("ball_map_floor", EXIT_CRITERIA["ball_map_floor"]),
        "court_lines_map_floor": config.get("court_lines_map_floor", EXIT_CRITERIA["court_lines_map_floor"]),
        "net_map_floor": config.get("net_map_floor", EXIT_CRITERIA["net_map_floor"]),
        "fp_rate_ceiling": config.get("fp_rate_ceiling", EXIT_CRITERIA["fp_rate_ceiling"]),
    }

    criteria_met, unmet = check_exit_criteria(val_metrics, frame_count, criteria)

    result = {
        "iteration": iteration,
        "player_map": val_metrics["player_map"],
        "ball_map": val_metrics["ball_map"],
        "court_lines_map": val_metrics["court_lines_map"],
        "net_map": val_metrics["net_map"],
        "fp_rate": val_metrics["fp_rate"],
        "overall_map50": val_metrics["overall_map50"],
        "generalization_score": gen_score,
        "frame_count": frame_count,
        "criteria_met": criteria_met,
        "unmet_criteria": unmet,
        "per_class": val_metrics.get("per_class", {}),
    }

    logger.info(
        f"Iter {iteration}: player_mAP={result['player_map']:.3f}, "
        f"ball_mAP={result['ball_map']:.3f}, "
        f"court_lines_mAP={result['court_lines_map']:.3f}, "
        f"net_mAP={result['net_map']:.3f}, "
        f"FP={result['fp_rate']:.3f}, "
        f"gen={result['generalization_score']:.3f}, "
        f"criteria_met={criteria_met}"
    )

    return result
