"""
Stage 00: Load pretrained YOLOv8n baseline model for tennis.

Model is pretrained on combined tennis datasets (TrackNet for ball,
Roboflow Tennis Dataset for player/court/net) for a strong 4-class prior.
Loaded from Modal Volume at /data/models/tennis_pretrained.pt if available,
otherwise downloads the standard YOLOv8n checkpoint as fallback.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR = Path("/data/models")
TENNIS_MODEL_PATH = MODELS_DIR / "tennis_pretrained.pt"
FALLBACK_MODEL = "yolov8n.pt"


def run(config: dict) -> dict:
    """
    Load the pretrained YOLOv8n model.

    Returns:
        dict with 'model_path' pointing to the loaded checkpoint.
    """
    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if TENNIS_MODEL_PATH.exists():
        model_path = str(TENNIS_MODEL_PATH)
        logger.info(f"Loaded tennis pretrained model from {model_path}")
    else:
        logger.warning(
            f"Tennis pretrained model not found at {TENNIS_MODEL_PATH}. "
            f"Falling back to {FALLBACK_MODEL}. "
            "For best results, provide a checkpoint pretrained on TrackNet + Roboflow Tennis."
        )
        model = YOLO(FALLBACK_MODEL)
        model_path = FALLBACK_MODEL

    model = YOLO(model_path)
    logger.info(f"Model loaded: {model_path} | Classes: {model.names}")

    return {"model_path": model_path}
