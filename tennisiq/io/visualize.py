"""
Annotated frame renderer for TennisIQ — FR-35.

Overlay style:
  - Court lines projected via homography (red)
  - Player A bounding box in blue, Player B in orange
  - Ball in yellow (default), green at in-bounce frames, red at out-bounce frames
  - Ball trajectory trail (recent N frames) in yellow
  - Court keypoint dots (cyan)
  - Frame info HUD
"""
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from tennisiq.geometry.court_reference import CourtReference

logger = logging.getLogger(__name__)

_court_ref = CourtReference()

COURT_LINES = [
    (_court_ref.baseline_top[0], _court_ref.baseline_top[1]),
    (_court_ref.baseline_bottom[0], _court_ref.baseline_bottom[1]),
    (_court_ref.left_court_line[0], _court_ref.left_court_line[1]),
    (_court_ref.right_court_line[0], _court_ref.right_court_line[1]),
    (_court_ref.left_inner_line[0], _court_ref.left_inner_line[1]),
    (_court_ref.right_inner_line[0], _court_ref.right_inner_line[1]),
    (_court_ref.top_inner_line[0], _court_ref.top_inner_line[1]),
    (_court_ref.bottom_inner_line[0], _court_ref.bottom_inner_line[1]),
    (_court_ref.middle_line[0], _court_ref.middle_line[1]),
    (_court_ref.net[0], _court_ref.net[1]),
]

RED = (0, 0, 255)
LIGHT_RED = (100, 100, 255)
BLUE = (255, 140, 0)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
ORANGE = (0, 165, 255)


def _draw_label(frame, text, x, y, color, bg_color=None):
    """Draw a YOLO-style label with background above a bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    if bg_color is None:
        bg_color = color

    top_left = (x, y - th - 6)
    bot_right = (x + tw + 4, y)
    cv2.rectangle(frame, top_left, bot_right, bg_color, -1)
    cv2.putText(frame, text, (x + 2, y - 4), font, scale, WHITE, thickness, cv2.LINE_AA)


def _project_court_line_to_pixel(p1_court, p2_court, matrix):
    pts = np.array([[list(p1_court), list(p2_court)]], dtype=np.float32)
    projected = cv2.perspectiveTransform(pts, matrix)
    a = tuple(projected[0][0].astype(int))
    b = tuple(projected[0][1].astype(int))
    return a, b


def _in_bounds(pt, w, h, margin=500):
    return -margin < pt[0] < w + margin and -margin < pt[1] < h + margin


def _draw_keypoint_lines(frame, keypoints):
    kp_lines = [
        (0, 1), (2, 3), (0, 2), (1, 3),
        (4, 5), (6, 7), (8, 9), (10, 11), (12, 13),
    ]
    for i, j in kp_lines:
        if i < len(keypoints) and j < len(keypoints):
            kx1, ky1 = keypoints[i]
            kx2, ky2 = keypoints[j]
            if kx1 is not None and kx2 is not None:
                cv2.line(frame, (int(kx1), int(ky1)), (int(kx2), int(ky2)), RED, 2, cv2.LINE_AA)


BALL_TRAIL_LENGTH = 10

def draw_overlay(
    frame: np.ndarray,
    homography_matrix: np.ndarray | None = None,
    keypoints: list[tuple] | None = None,
    ball_xy: tuple | None = None,
    ball_bbox: tuple | None = None,
    ball_conf: float | None = None,
    player_a=None,
    player_b=None,
    ball_in_out: str | None = None,
    ball_trail: list[tuple] | None = None,
    frame_idx: int = 0,
    confidence: float = 0.0,
) -> np.ndarray:
    """
    Draw detection overlays on a single frame — FR-35 style.

    Args:
        frame: BGR image (H, W, 3)
        homography_matrix: 3x3 court->pixel matrix
        keypoints: 14 detected (x,y) pairs
        ball_xy: (x, y) pixel center of ball
        ball_bbox: (x1, y1, x2, y2) ball bounding box from YOLO
        ball_conf: ball detection confidence
        player_a: PlayerDetection for Player A (blue)
        player_b: PlayerDetection for Player B (orange)
        ball_in_out: "in", "out", or None — colors ball green/red at bounce frames
        ball_trail: recent ball positions for trajectory trace
        frame_idx: for HUD
        confidence: homography confidence for HUD
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Court lines via homography
    if homography_matrix is not None:
        for p1, p2 in COURT_LINES:
            try:
                a, b = _project_court_line_to_pixel(p1, p2, homography_matrix)
                if _in_bounds(a, w, h) or _in_bounds(b, w, h):
                    cv2.line(out, a, b, RED, 2, cv2.LINE_AA)
            except Exception:
                continue
    elif keypoints is not None:
        _draw_keypoint_lines(out, keypoints)

    # Keypoint dots
    if keypoints is not None:
        for kx, ky in keypoints:
            if kx is not None:
                cv2.circle(out, (int(kx), int(ky)), 4, CYAN, -1, cv2.LINE_AA)

    # Player A (blue) and Player B (orange) — FR-35
    for player, color, label in [(player_a, BLUE, "Player A"), (player_b, ORANGE, "Player B")]:
        if player is not None:
            x1, y1, x2, y2 = [int(c) for c in player.bbox]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            _draw_label(out, f"{label} {player.confidence:.2f}", x1, y1, color)

    # Ball trajectory trail in yellow — FR-35
    if ball_trail:
        valid = [(int(x), int(y)) for x, y in ball_trail if x is not None]
        for i in range(1, len(valid)):
            alpha = int(80 + 175 * (i / len(valid)))
            cv2.line(out, valid[i - 1], valid[i], (0, alpha, alpha), 2, cv2.LINE_AA)

    # Ball: green for in, red for out, yellow otherwise — FR-35
    if ball_in_out == "in":
        ball_color = GREEN
    elif ball_in_out == "out":
        ball_color = RED
    else:
        ball_color = YELLOW

    if ball_bbox is not None:
        x1, y1, x2, y2 = [int(c) for c in ball_bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), ball_color, 2, cv2.LINE_AA)
        label = f"ball {ball_conf:.2f}" if ball_conf is not None else "ball"
        suffix = f" [{ball_in_out}]" if ball_in_out else ""
        _draw_label(out, label + suffix, x1, y1, ball_color)
    elif ball_xy is not None and ball_xy[0] is not None:
        bx, by = int(ball_xy[0]), int(ball_xy[1])
        cv2.circle(out, (bx, by), 7, ball_color, -1, cv2.LINE_AA)
        cv2.circle(out, (bx, by), 7, (0, 0, 0), 2, cv2.LINE_AA)

    # HUD
    _draw_hud(out, frame_idx, confidence)

    return out


def _draw_hud(frame, frame_idx, confidence):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (220, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Frame {frame_idx}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    conf_color = GREEN if confidence >= 0.7 else YELLOW if confidence >= 0.4 else RED
    cv2.putText(frame, f"Court conf: {confidence:.2f}", (10, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)


def render_overlay_video(
    video_path: str,
    output_path: str,
    homographies: list,
    court_keypoints: list,
    ball_positions: list,
    player_results: list,
    fps: float,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    ball_detections_yolo: list | None = None,
    events: list | None = None,
    progress_callback=None,
) -> str:
    """
    Render an annotated overlay video — FR-35.

    Player A in blue, Player B in orange. Ball trail in yellow.
    Ball colored green at in-bounce frames, red at out-bounce frames.
    """
    # Pre-build bounce lookup: frame_idx → in_out
    bounce_at_frame: dict[int, str] = {}
    if events:
        for evt in events:
            if evt.event_type == "bounce" and evt.in_out:
                bounce_at_frame[evt.frame_idx] = evt.in_out

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * src_fps) if start_sec else 0
    end_frame = int(end_sec * src_fps) if end_sec is not None else total_frames
    n_frames = end_frame - start_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_output, fourcc, fps, (w, h))

    trail: list[tuple] = []

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        hom_matrix = None
        conf = 0.0
        if i < len(homographies) and homographies[i].matrix is not None:
            hom_matrix = homographies[i].matrix
            conf = homographies[i].confidence

        kps = court_keypoints[i] if i < len(court_keypoints) else None

        ball_xy = None
        ball_bbox = None
        ball_conf = None

        if ball_detections_yolo is not None and i < len(ball_detections_yolo):
            det = ball_detections_yolo[i]
            if det is not None:
                ball_xy = det.center_xy
                ball_bbox = det.bbox
                ball_conf = det.confidence
        elif i < len(ball_positions):
            bp = ball_positions[i]
            if hasattr(bp, "pixel_xy"):
                ball_xy = bp.pixel_xy
            elif isinstance(bp, (list, tuple)) and len(bp) == 2:
                ball_xy = bp

        # Maintain ball trail — clear on tracking gaps to avoid cross-frame lines
        if ball_xy and ball_xy[0] is not None:
            trail.append(ball_xy)
            if len(trail) > BALL_TRAIL_LENGTH:
                trail = trail[-BALL_TRAIL_LENGTH:]
        else:
            trail = []

        # Player A / B from frame results
        pa = player_results[i].player_a if i < len(player_results) else None
        pb = player_results[i].player_b if i < len(player_results) else None

        annotated = draw_overlay(
            frame,
            homography_matrix=hom_matrix,
            keypoints=kps,
            ball_xy=ball_xy,
            ball_bbox=ball_bbox,
            ball_conf=ball_conf,
            player_a=pa,
            player_b=pb,
            ball_in_out=bounce_at_frame.get(i),
            ball_trail=list(trail),
            frame_idx=i,
            confidence=conf,
        )
        writer.write(annotated)

        if progress_callback and (i + 1) % 50 == 0:
            progress_callback(i + 1, n_frames)

    cap.release()
    writer.release()

    # Re-encode to H.264 for universal playback (Windows, macOS, browsers)
    import subprocess
    import shutil
    if shutil.which("ffmpeg"):
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_output, "-c:v", "libx264",
                 "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
                 output_path],
                check=True, capture_output=True,
            )
            os.remove(tmp_output)
            logger.info(f"Overlay video (H.264): {output_path} ({n_frames} frames)")
        except (subprocess.CalledProcessError, FileNotFoundError):
            os.rename(tmp_output, output_path)
            logger.warning(f"ffmpeg re-encode failed, using mp4v: {output_path}")
    else:
        os.rename(tmp_output, output_path)
        logger.info(f"Overlay video (mp4v): {output_path} ({n_frames} frames)")

    return output_path


PRE_PAD_SEC = 1.0
POST_PAD_SEC = 0.2


def extract_point_clips(
    video_path: str,
    output_dir: str,
    points: list,
    fps: float,
    start_sec: float = 0.0,
) -> list[str]:
    """FR-39: Extract a video clip for every detected point.

    Each clip has 1s of pre-padding and 0.2s of post-padding.
    Uses ffmpeg for fast seeking + H.264 output.
    """
    import subprocess
    import shutil

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = []

    has_ffmpeg = shutil.which("ffmpeg") is not None

    for pt in points:
        clip_start = max(0, pt.start_sec - PRE_PAD_SEC)
        clip_end = pt.end_sec + POST_PAD_SEC
        duration = clip_end - clip_start

        out_path = os.path.join(output_dir, f"point_{pt.point_idx}.mp4")

        if has_ffmpeg:
            try:
                subprocess.run(
                    ["ffmpeg", "-y",
                     "-ss", f"{clip_start:.3f}",
                     "-i", video_path,
                     "-t", f"{duration:.3f}",
                     "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                     "-pix_fmt", "yuv420p",
                     "-an", out_path],
                    check=True, capture_output=True,
                )
                paths.append(out_path)
                logger.info(f"Point {pt.point_idx} clip: {clip_start:.1f}s-{clip_end:.1f}s → {out_path}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"ffmpeg clip extraction failed for point {pt.point_idx}: {e}")
        else:
            cap = cv2.VideoCapture(video_path)
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            s_frame = int(clip_start * src_fps)
            e_frame = int(clip_end * src_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, src_fps, (w, h))

            for _ in range(e_frame - s_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)

            writer.release()
            cap.release()
            paths.append(out_path)
            logger.info(f"Point {pt.point_idx} clip (cv2): {clip_start:.1f}s-{clip_end:.1f}s → {out_path}")

    return paths
