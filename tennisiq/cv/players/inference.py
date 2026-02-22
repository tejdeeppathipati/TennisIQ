"""
Player detection and tracking — FR-16 through FR-20.

FR-16: YOLOv8n + ByteTrack person detection/tracking per frame
FR-17: Court boundary filter (discard detections outside court + margin)
FR-18: Assign Player A (near-side, largest) and Player B (far-side, largest)
FR-19: HOG fallback if YOLO is unavailable
FR-20: Project player foot positions to court-space via homography
"""
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0
PERSON_CONF_THRESHOLD = 0.5
COURT_MARGIN_UNITS = 200


@dataclass
class PlayerDetection:
    """Single detected person in a frame."""
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    track_id: int | None
    foot_pixel: tuple[float, float]          # bottom-center of bbox
    foot_court: tuple[float | None, float | None]
    inside_court: bool


@dataclass
class FramePlayers:
    """Per-frame player assignment result."""
    frame_idx: int
    all_detections: list[PlayerDetection]
    player_a: PlayerDetection | None  # near-side (bottom of image)
    player_b: PlayerDetection | None  # far-side (top of image)


def _foot_position(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Bottom-center of a bounding box — best proxy for where the player stands."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def _is_inside_court(
    foot_court: tuple[float | None, float | None],
    court_ref,
    margin: float = COURT_MARGIN_UNITS,
) -> bool:
    """Check if a court-space foot position is within the court boundaries + margin."""
    cx, cy = foot_court
    if cx is None or cy is None:
        return False

    x_min = court_ref.baseline_top[0][0] - margin
    x_max = court_ref.baseline_top[1][0] + margin
    y_min = court_ref.baseline_top[0][1] - margin
    y_max = court_ref.baseline_bottom[0][1] + margin

    return x_min <= cx <= x_max and y_min <= cy <= y_max


def _assign_players(
    valid_detections: list[PlayerDetection],
    frame_height: int,
) -> tuple[PlayerDetection | None, PlayerDetection | None]:
    """
    FR-18: Player A = near-side (high foot y in pixel space, i.e. bottom of image),
    Player B = far-side (low foot y, i.e. top of image).
    Tie-break by bbox area (largest wins).
    """
    if not valid_detections:
        return None, None

    mid_y = frame_height / 2.0

    near_side = [d for d in valid_detections if d.foot_pixel[1] >= mid_y]
    far_side = [d for d in valid_detections if d.foot_pixel[1] < mid_y]

    def _bbox_area(d: PlayerDetection) -> float:
        x1, y1, x2, y2 = d.bbox
        return (x2 - x1) * (y2 - y1)

    player_a = max(near_side, key=_bbox_area) if near_side else None
    player_b = max(far_side, key=_bbox_area) if far_side else None

    return player_a, player_b


# ─── FR-19: HOG fallback ─────────────────────────────────────────────────────

def _hog_detect(frame_bgr: np.ndarray, conf_threshold: float = 0.3) -> list[dict]:
    """Fallback person detector using OpenCV's HOG + SVM."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(frame_bgr, winStride=(8, 8), padding=(4, 4), scale=1.05)
    results = []
    for (x, y, w, h), weight in zip(boxes, weights):
        if weight >= conf_threshold:
            results.append({
                "bbox": (float(x), float(y), float(x + w), float(y + h)),
                "confidence": float(weight),
                "track_id": None,
            })
    return results


# ─── FR-16: Main detector class ──────────────────────────────────────────────

class PlayerDetector:
    """Detects and tracks players using YOLOv8n + ByteTrack, with HOG fallback."""

    def __init__(self, device: str | None = None):
        self.yolo = None
        self.use_hog = False

        try:
            from ultralytics import YOLO
            self.yolo = YOLO("yolov8n.pt")
            if device:
                self.yolo.to(device)
            logger.info(f"YOLOv8n loaded for player detection (device={device})")
        except Exception as e:
            logger.warning(f"YOLOv8n unavailable ({e}), falling back to HOG detector")
            self.use_hog = True

    def detect_frame(
        self,
        frame_bgr: np.ndarray,
        homography=None,
        court_ref=None,
        conf_threshold: float = PERSON_CONF_THRESHOLD,
    ) -> list[PlayerDetection]:
        """Detect persons in a single frame and filter by court boundary."""
        raw_detections: list[dict] = []

        if self.yolo is not None:
            results = self.yolo.track(
                frame_bgr, persist=True, tracker="bytetrack.yaml",
                classes=[PERSON_CLASS_ID], conf=conf_threshold, verbose=False,
            )
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu())
                    tid = int(box.id[0].cpu()) if box.id is not None else None
                    raw_detections.append({
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "confidence": conf,
                        "track_id": tid,
                    })
        else:
            raw_detections = _hog_detect(frame_bgr, conf_threshold)

        player_dets = []
        for det in raw_detections:
            foot_px = _foot_position(det["bbox"])
            foot_ct: tuple[float | None, float | None] = (None, None)
            inside = True

            if homography is not None and homography.reliable:
                from tennisiq.geometry.homography import pixel_to_court
                foot_ct = pixel_to_court(foot_px[0], foot_px[1], homography)

                if court_ref is not None and foot_ct[0] is not None:
                    inside = _is_inside_court(foot_ct, court_ref)

            player_dets.append(PlayerDetection(
                bbox=det["bbox"],
                confidence=det["confidence"],
                track_id=det["track_id"],
                foot_pixel=foot_px,
                foot_court=foot_ct,
                inside_court=inside,
            ))

        return player_dets

    def detect_video(
        self,
        video_path: str,
        homographies: list | None = None,
        start_sec: float | None = None,
        end_sec: float | None = None,
        progress_callback=None,
    ) -> list[FramePlayers]:
        """
        Run player detection + assignment on a video segment.

        Returns a list of FramePlayers, one per frame.
        """
        from tennisiq.geometry.court_reference import CourtReference
        court_ref = CourtReference()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(start_sec * fps) if start_sec is not None else 0
        end_frame = int(end_sec * fps) if end_sec is not None else total_video_frames
        start_frame = max(0, min(start_frame, total_video_frames))
        end_frame = max(start_frame, min(end_frame, total_video_frames))
        total_to_process = end_frame - start_frame

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        logger.info(f"Player detection: frames {start_frame}-{end_frame} ({total_to_process} frames)")

        all_results: list[FramePlayers] = []
        frames_read = 0
        last_valid_a: PlayerDetection | None = None
        last_valid_b: PlayerDetection | None = None
        carry_forward_count = 0

        while frames_read < total_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            hom = homographies[frames_read] if homographies and frames_read < len(homographies) else None

            detections = self.detect_frame(frame, homography=hom, court_ref=court_ref)
            valid = [d for d in detections if d.inside_court]
            player_a, player_b = _assign_players(valid, frame_h)

            # NFR-R05: carry forward last valid positions when detection yields null
            if player_a is not None:
                last_valid_a = player_a
            elif last_valid_a is not None:
                player_a = last_valid_a
                carry_forward_count += 1

            if player_b is not None:
                last_valid_b = player_b
            elif last_valid_b is not None:
                player_b = last_valid_b
                carry_forward_count += 1

            all_results.append(FramePlayers(
                frame_idx=frames_read,
                all_detections=detections,
                player_a=player_a,
                player_b=player_b,
            ))

            frames_read += 1

            if progress_callback and frames_read % 50 == 0:
                progress_callback(frames_read, total_to_process)

        cap.release()

        total_a = sum(1 for r in all_results if r.player_a is not None)
        total_b = sum(1 for r in all_results if r.player_b is not None)
        if frames_read > 0:
            logger.info(
                f"Player detection complete: {frames_read} frames, "
                f"Player A in {total_a} ({total_a/frames_read*100:.0f}%), "
                f"Player B in {total_b} ({total_b/frames_read*100:.0f}%), "
                f"carry-forward fills: {carry_forward_count}"
            )
        else:
            logger.warning("Player detection: 0 frames read")

        return all_results
