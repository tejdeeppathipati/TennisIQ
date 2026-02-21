from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


BBox = Tuple[float, float, float, float]


def _bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _canonical_two_players(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        return {"playerA_bbox": None, "playerB_bbox": None, "playerA_id": None, "playerB_id": None, "tracks": []}

    candidates = sorted(candidates, key=lambda x: _bbox_area(x["bbox"]), reverse=True)[:2]
    # Player A = near side (larger foot y), Player B = far side.
    candidates = sorted(candidates, key=lambda x: x["bbox"][3], reverse=True)

    player_a = candidates[0] if len(candidates) > 0 else None
    player_b = candidates[1] if len(candidates) > 1 else None

    return {
        "playerA_bbox": player_a["bbox"] if player_a else None,
        "playerB_bbox": player_b["bbox"] if player_b else None,
        "playerA_id": player_a.get("id") if player_a else None,
        "playerB_id": player_b.get("id") if player_b else None,
        "tracks": candidates,
    }


def _detect_players_with_ultralytics(
    frames: List[np.ndarray],
    model_path: str = "yolov8n.pt",
    conf: float = 0.2,
    iou: float = 0.5,
    tracker: str = "bytetrack.yaml",
    allow_model_download: bool = False,
) -> List[Dict[str, Any]]:
    try:
        from ultralytics import YOLO
    except Exception:
        return []

    model_candidate = Path(model_path)
    is_local_file = model_candidate.exists()
    is_remote_uri = model_path.startswith(("http://", "https://"))
    if not is_local_file and not is_remote_uri and not allow_model_download:
        return []

    model = YOLO(model_path)
    outputs: List[Dict[str, Any]] = []

    for frame in frames:
        results = model.track(
            source=frame,
            persist=True,
            classes=[0],
            conf=conf,
            iou=iou,
            tracker=tracker,
            verbose=False,
        )
        result = results[0]

        candidates: List[Dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones((xyxy.shape[0],), dtype=float)
            ids = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else np.arange(xyxy.shape[0], dtype=int)
            clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((xyxy.shape[0],), dtype=int)

            for i in range(xyxy.shape[0]):
                if clss[i] != 0:
                    continue
                x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                candidates.append({"id": int(ids[i]), "bbox": (x1, y1, x2, y2), "conf": float(confs[i])})

        outputs.append(_canonical_two_players(candidates))

    return outputs


def _detect_players_with_hog(frames: List[np.ndarray]) -> List[Dict[str, Any]]:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    outputs: List[Dict[str, Any]] = []
    for frame in frames:
        rects, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        candidates = []
        for i, (x, y, w, h) in enumerate(rects):
            x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
            conf = float(weights[i]) if i < len(weights) else 0.5
            candidates.append({"id": i, "bbox": (x1, y1, x2, y2), "conf": conf})
        outputs.append(_canonical_two_players(candidates))
    return outputs


def detect_players(
    frames: List[np.ndarray],
    model_path: str = "yolov8n.pt",
    conf: float = 0.2,
    iou: float = 0.5,
    tracker: str = "bytetrack.yaml",
    fallback_hog: bool = True,
    allow_model_download: bool = False,
) -> List[Dict[str, Any]]:
    outputs = _detect_players_with_ultralytics(
        frames,
        model_path=model_path,
        conf=conf,
        iou=iou,
        tracker=tracker,
        allow_model_download=allow_model_download,
    )
    if outputs:
        return outputs

    if fallback_hog:
        return _detect_players_with_hog(frames)

    return [{"playerA_bbox": None, "playerB_bbox": None, "playerA_id": None, "playerB_id": None, "tracks": []} for _ in frames]
