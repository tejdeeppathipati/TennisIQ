from __future__ import annotations

from typing import List

from tennisiq.cv.players.infer import detect_players


def run_step_03_players(
    frames,
    court_points=None,
    model_path: str = "yolov8n.pt",
    conf: float = 0.2,
    iou: float = 0.5,
    tracker: str = "bytetrack.yaml",
    fallback_hog: bool = True,
    allow_model_download: bool = False,
    backend: str = "auto",
    device: str = "auto",
) -> List[dict]:
    return detect_players(
        frames,
        court_points=court_points,
        model_path=model_path,
        conf=conf,
        iou=iou,
        tracker=tracker,
        fallback_hog=fallback_hog,
        allow_model_download=allow_model_download,
        backend=backend,
        device=device,
    )
