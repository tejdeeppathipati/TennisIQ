from __future__ import annotations

from typing import List

from tennisiq.cv.players.infer import detect_players


def run_step_03_players(
    frames,
    model_path: str = "yolov8n.pt",
    conf: float = 0.2,
    iou: float = 0.5,
    tracker: str = "bytetrack.yaml",
    fallback_hog: bool = True,
    allow_model_download: bool = False,
) -> List[dict]:
    return detect_players(
        frames,
        model_path=model_path,
        conf=conf,
        iou=iou,
        tracker=tracker,
        fallback_hog=fallback_hog,
        allow_model_download=allow_model_download,
    )
