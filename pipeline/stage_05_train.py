"""
Stage 05: YOLOv8n fine-tuning on Modal A100.

Fine-tunes the pretrained YOLOv8n model on coach-approved, policy-refined labels
for the 4 tennis classes: player, ball, court_lines, net.
Generates a YOLO dataset.yaml, trains, and saves best.pt to Modal Volume.
"""
import logging
import random
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

TRAINING_DIR = Path("/data/training")
MODELS_DIR = Path("/data/models")


def build_dataset_yaml(
    job_id: str,
    train_frames: list[str],
    train_labels: list[str],
    val_frames: list[str],
    val_labels: list[str],
) -> Path:
    """
    Create YOLO dataset.yaml and copy frames + labels into train/val split dirs.
    """
    dataset_dir = TRAINING_DIR / job_id
    for split in ("train/images", "train/labels", "val/images", "val/labels"):
        (dataset_dir / split).mkdir(parents=True, exist_ok=True)

    for fp, lp in zip(train_frames, train_labels):
        shutil.copy2(fp, dataset_dir / "train" / "images" / Path(fp).name)
        shutil.copy2(lp, dataset_dir / "train" / "labels" / Path(lp).name)

    for fp, lp in zip(val_frames, val_labels):
        shutil.copy2(fp, dataset_dir / "val" / "images" / Path(fp).name)
        shutil.copy2(lp, dataset_dir / "val" / "labels" / Path(lp).name)

    yaml_content = f"""path: {dataset_dir}
train: train/images
val: val/images

nc: 4
names:
  0: player
  1: ball
  2: court_lines
  3: net
"""
    yaml_path = dataset_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    logger.info(f"Dataset YAML written: {yaml_path} ({len(train_frames)} train, {len(val_frames)} val)")
    return yaml_path


def split_train_val(
    frame_paths: list[str],
    label_paths: list[str],
    val_ratio: float = 0.20,
) -> tuple[list, list, list, list]:
    """Split frame/label pairs into train and val sets."""
    paired = list(zip(frame_paths, label_paths))
    random.shuffle(paired)
    split_idx = int(len(paired) * (1 - val_ratio))
    train = paired[:split_idx]
    val = paired[split_idx:]
    train_f, train_l = zip(*train) if train else ([], [])
    val_f, val_l = zip(*val) if val else ([], [])
    return list(train_f), list(train_l), list(val_f), list(val_l)


def run(
    job_id: str,
    iteration: int,
    frame_paths: list[str],
    label_paths: list[str],
    model_path: str,
    config: dict,
) -> dict:
    """
    Fine-tune YOLOv8n on labeled frames.

    Returns:
        dict with 'best_model_path', 'final_model_path'
    """
    from ultralytics import YOLO

    epochs = config.get("epochs", 50)
    batch_size = config.get("batch_size", 16)
    img_size = config.get("img_size", 640)
    val_ratio = 0.20

    valid_pairs = [
        (fp, lp) for fp, lp in zip(frame_paths, label_paths)
        if Path(fp).exists() and Path(lp).exists() and Path(lp).read_text().strip()
    ]

    if len(valid_pairs) < 10:
        raise ValueError(f"Insufficient training data: only {len(valid_pairs)} valid frame/label pairs")

    frame_paths_valid, label_paths_valid = zip(*valid_pairs)
    train_f, train_l, val_f, val_l = split_train_val(
        list(frame_paths_valid), list(label_paths_valid), val_ratio=val_ratio
    )

    logger.info(f"Training iteration {iteration}: {len(train_f)} train / {len(val_f)} val frames")

    yaml_path = build_dataset_yaml(job_id, train_f, train_l, val_f, val_l)

    model = YOLO(model_path)
    run_name = f"{job_id}_iter{iteration}"
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        name=run_name,
        project=str(TRAINING_DIR / "runs"),
        exist_ok=True,
        verbose=False,
        device="0",
    )

    run_dir = TRAINING_DIR / "runs" / run_name
    best_pt = run_dir / "weights" / "best.pt"

    saved_best = MODELS_DIR / f"{job_id}_iter{iteration}_best.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(best_pt), str(saved_best))

    latest_best = MODELS_DIR / "best.pt"
    shutil.copy2(str(best_pt), str(latest_best))

    logger.info(f"Fine-tuning complete. best.pt saved to {saved_best}")

    return {
        "best_model_path": str(saved_best),
        "latest_model_path": str(latest_best),
        "run_dir": str(run_dir),
        "train_count": len(train_f),
        "val_count": len(val_f),
    }
