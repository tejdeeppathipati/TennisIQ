from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from tennisiq.common import get_device


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


def _bbox_center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))


def _bbox_footpoint(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) / 2.0), float(y2))


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _build_court_hull(court_kps: Sequence[tuple[float | None, float | None]] | None) -> np.ndarray | None:
    if not court_kps:
        return None
    pts = []
    for p in court_kps:
        if p is None:
            continue
        x, y = p
        if x is None or y is None:
            continue
        pts.append([float(x), float(y)])
    if len(pts) < 4:
        return None
    arr = np.asarray(pts, dtype=np.float32)
    hull = cv2.convexHull(arr.reshape((-1, 1, 2)))
    return hull


def _court_x_limits(court_kps: Sequence[tuple[float | None, float | None]] | None) -> tuple[float, float] | None:
    if not court_kps or len(court_kps) < 8:
        return None
    left_vals = []
    right_vals = []
    for i in (4, 5):
        x, y = court_kps[i]
        if x is not None and y is not None:
            left_vals.append(float(x))
    for i in (6, 7):
        x, y = court_kps[i]
        if x is not None and y is not None:
            right_vals.append(float(x))
    if not left_vals or not right_vals:
        return None
    return (float(np.median(left_vals)), float(np.median(right_vals)))


def _is_footpoint_in_hull(
    box: BBox,
    hull: np.ndarray | None,
    x_limits: tuple[float, float] | None,
    margin_px: float = 260.0,
    x_margin_px: float = 180.0,
) -> bool:
    if hull is None:
        if x_limits is None:
            return True
        fx, _ = _bbox_footpoint(box)
        return (x_limits[0] - x_margin_px) <= fx <= (x_limits[1] + x_margin_px)
    foot = _bbox_footpoint(box)
    if x_limits is not None:
        if not ((x_limits[0] - x_margin_px) <= foot[0] <= (x_limits[1] + x_margin_px)):
            return False
    # pointPolygonTest > 0 inside, =0 on edge, <0 outside
    dist = cv2.pointPolygonTest(hull, foot, True)
    return dist >= -margin_px


def _letterbox_for_detector(frame: np.ndarray, max_side: int = 1280) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return frame, 1.0
    scale = float(max_side / side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h)), scale


def _pick_initial_players(boxes: List[dict], frame_h: int) -> tuple[dict | None, dict | None]:
    if not boxes:
        return None, None
    near_candidates = [b for b in boxes if b["bbox"][3] >= 0.62 * frame_h]
    far_candidates = [b for b in boxes if b["bbox"][3] < 0.70 * frame_h]

    player_a = None
    if near_candidates:
        player_a = sorted(near_candidates, key=lambda x: (x["score"], x["bbox"][3]), reverse=True)[0]

    if player_a is not None:
        far_candidates = [b for b in far_candidates if _iou(b["bbox"], player_a["bbox"]) < 0.9]

    player_b = None
    if far_candidates:
        player_b = sorted(far_candidates, key=lambda x: (x["score"], -x["bbox"][3]), reverse=True)[0]
    return player_a, player_b


def _select_closest(prev_bbox: BBox | None, candidates: List[dict], max_dist: float) -> dict | None:
    if not candidates:
        return None
    if prev_bbox is None:
        ranked = sorted(candidates, key=lambda c: (c["score"], _bbox_area(c["bbox"])), reverse=True)
        return ranked[0] if ranked else None

    prev_pt = _bbox_footpoint(prev_bbox)
    best = None
    best_dist = float("inf")
    for cand in candidates:
        d = _dist(prev_pt, _bbox_footpoint(cand["bbox"]))
        if d < best_dist:
            best = cand
            best_dist = d
    if best is None or best_dist > max_dist:
        return None
    return best


def _detect_players_with_artlabss_frcnn(
    frames: List[np.ndarray],
    court_points: Sequence[Sequence[tuple[float | None, float | None]]] | None = None,
    device: str = "auto",
    min_conf: float = 0.45,
    max_track_dist_px: float = 220.0,
    max_detector_side: int = 1280,
) -> List[Dict[str, Any]]:
    # Inspired by ArtLabss repository player detection approach:
    # FasterRCNN person detection + temporal association.
    try:
        import torchvision
    except Exception:
        return []

    runtime_device = get_device(device)
    if runtime_device == "cuda":
        torch_device = torch.device("cuda")
    elif runtime_device == "mps":
        # torchvision detection ops on MPS currently fall back often and can be much slower than CPU.
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device("cpu")

    try:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    except Exception:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model = model.to(torch_device)
    model.eval()

    outputs: List[Dict[str, Any]] = []
    prev_a: BBox | None = None
    prev_b: BBox | None = None
    miss_a = 0
    miss_b = 0
    max_misses = 20

    for frame_idx, frame in enumerate(tqdm(frames, desc="Step 3/6 Players (ArtLabss)", unit="frame")):
        frame_h, frame_w = frame.shape[:2]
        frame_kps = court_points[frame_idx] if court_points is not None and frame_idx < len(court_points) else None
        hull = _build_court_hull(frame_kps)
        x_limits = _court_x_limits(frame_kps)
        if x_limits is not None:
            x_margin = float(max(300.0, min(900.0, 0.50 * (x_limits[1] - x_limits[0]))))
        else:
            x_margin = 300.0
        det_frame, scale = _letterbox_for_detector(frame, max_side=max_detector_side)
        rgb = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
        inp = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        inp = inp.to(torch_device)

        with torch.no_grad():
            pred = model([inp])[0]

        boxes: List[dict] = []
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            if int(label.item()) != 1:
                continue
            s = float(score.item())
            if s < min_conf:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.detach().cpu().tolist()]
            if scale != 1.0:
                x1 /= scale
                y1 /= scale
                x2 /= scale
                y2 /= scale
            area = _bbox_area((x1, y1, x2, y2))
            if area < 1800.0:
                continue
            if y2 < 0.12 * frame_h:
                continue
            bbox = (x1, y1, x2, y2)
            if not _is_footpoint_in_hull(bbox, hull, x_limits=x_limits, margin_px=260.0, x_margin_px=x_margin):
                continue
            boxes.append({"bbox": bbox, "score": s})

        if prev_a is None and prev_b is None:
            cand_a, cand_b = _pick_initial_players(boxes, frame_h)
        else:
            # Reserve distinct candidates for A/B using nearest-footpoint association.
            available = list(boxes)
            near_candidates = [c for c in available if c["bbox"][3] >= 0.62 * frame_h]
            far_candidates = [c for c in available if c["bbox"][3] < 0.70 * frame_h]

            cand_a_pool = near_candidates if (near_candidates or prev_a is not None) else []
            cand_a = _select_closest(prev_a, cand_a_pool, max_track_dist_px)
            if cand_a is not None:
                available = [c for c in available if _iou(c["bbox"], cand_a["bbox"]) < 0.9]
                far_candidates = [c for c in far_candidates if _iou(c["bbox"], cand_a["bbox"]) < 0.9]
            cand_b_pool = far_candidates if (far_candidates or prev_b is not None) else []
            cand_b = _select_closest(prev_b, cand_b_pool, max_track_dist_px)
            if cand_a is None and cand_b is None:
                # Reinitialize if both tracks are lost.
                cand_a, cand_b = _pick_initial_players(boxes, frame_h)

        if cand_a is not None:
            prev_a = cand_a["bbox"]
            miss_a = 0
        else:
            miss_a += 1
            if miss_a > max_misses:
                prev_a = None

        if cand_b is not None:
            prev_b = cand_b["bbox"]
            miss_b = 0
        else:
            miss_b += 1
            if miss_b > max_misses:
                prev_b = None

        # Keep semantic consistency: playerA is near side, playerB is far side.
        if prev_a is not None and prev_b is not None and prev_a[3] < prev_b[3]:
            prev_a, prev_b = prev_b, prev_a

        tracks = []
        if prev_a is not None:
            tracks.append({"id": 1, "bbox": prev_a, "conf": float(cand_a["score"]) if cand_a else 0.0})
        if prev_b is not None:
            tracks.append({"id": 2, "bbox": prev_b, "conf": float(cand_b["score"]) if cand_b else 0.0})

        outputs.append(
            {
                "playerA_bbox": prev_a,
                "playerB_bbox": prev_b,
                "playerA_id": 1 if prev_a is not None else None,
                "playerB_id": 2 if prev_b is not None else None,
                "tracks": tracks,
            }
        )

    return outputs


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

    for frame in tqdm(frames, desc="Step 3/6 Players (YOLO)", unit="frame"):
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
    for frame in tqdm(frames, desc="Step 3/6 Players (HOG)", unit="frame"):
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
    court_points: Sequence[Sequence[tuple[float | None, float | None]]] | None = None,
    model_path: str = "yolov8n.pt",
    conf: float = 0.2,
    iou: float = 0.5,
    tracker: str = "bytetrack.yaml",
    fallback_hog: bool = True,
    allow_model_download: bool = False,
    backend: str = "auto",
    device: str = "auto",
) -> List[Dict[str, Any]]:
    back = backend.lower()
    if back not in {"artlabss", "yolo", "hog", "auto"}:
        back = "auto"

    if back in {"yolo", "auto"}:
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

    if back in {"artlabss", "auto"}:
        outputs = _detect_players_with_artlabss_frcnn(frames, court_points=court_points, device=device)
        if outputs:
            return outputs

    if back in {"hog", "auto"} and fallback_hog:
        return _detect_players_with_hog(frames)

    return [{"playerA_bbox": None, "playerB_bbox": None, "playerA_id": None, "playerB_id": None, "tracks": []} for _ in frames]
