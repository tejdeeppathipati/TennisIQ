"""
Stage 03: Pseudo-label generation using pretrained YOLOv8n model.

Runs inference on deduplicated frames to generate first-pass YOLO-format
label files (.txt) for all four tennis classes: player, ball, court_lines, net.
These labels are refined by Codex subagents in Stage 04.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PSEUDO_LABELS_DIR = Path("/data/pseudo_labels")

CLASS_MAP = {
    0: "player",
    1: "ball",
    2: "court_lines",
    3: "net",
}


def write_yolo_label(label_path: Path, detections: list[dict]) -> None:
    """Write YOLO format label file: class cx cy w h (normalized)."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for det in detections:
        cls = det["class_id"]
        cx = det["cx"]
        cy = det["cy"]
        w = det["w"]
        h = det["h"]
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines))


def run(job_id: str, frame_paths: list[str], model_path: str, config: dict) -> dict:
    """
    Run pretrained YOLOv8n inference on frames to generate pseudo-labels.

    YOLO class mapping for tennis:
        0 = player
        1 = ball
        2 = court_lines
        3 = net

    Returns:
        dict with 'label_dir', 'labeled_frames', 'label_paths'
    """
    from ultralytics import YOLO

    conf_threshold = config.get("pseudo_label_conf", 0.20)
    label_dir = PSEUDO_LABELS_DIR / job_id
    label_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    labeled_frames = []
    label_paths = []

    for frame_path in frame_paths:
        results = model.predict(
            source=frame_path,
            conf=conf_threshold,
            verbose=False,
            save=False,
        )

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xywhn = boxes.xywhn[i].tolist()
                detections.append({
                    "class_id": cls_id,
                    "cx": xywhn[0],
                    "cy": xywhn[1],
                    "w": xywhn[2],
                    "h": xywhn[3],
                    "confidence": conf,
                })

        frame_stem = Path(frame_path).stem
        label_path = label_dir / f"{frame_stem}.txt"
        write_yolo_label(label_path, detections)

        labeled_frames.append(frame_path)
        label_paths.append(str(label_path))

    logger.info(f"Pseudo-labeled {len(labeled_frames)} frames -> {label_dir}")

    return {
        "label_dir": str(label_dir),
        "labeled_frames": labeled_frames,
        "label_paths": label_paths,
    }
