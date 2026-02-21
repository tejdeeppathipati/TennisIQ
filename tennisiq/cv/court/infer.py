from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from tennisiq.common import get_device
from tennisiq.cv.court.model import CourtKeypointNet
from tennisiq.cv.court.postprocess import postprocess, refine_kps
from tennisiq.geometry.homography import get_trans_matrix, refer_kps
from tennisiq.io.video import normalize_video_fps_moviepy


Point = Tuple[float | None, float | None]


def _project_reference_points(h_ref_to_img: np.ndarray | None) -> List[Point]:
    if h_ref_to_img is None:
        return [(None, None)] * int(refer_kps.shape[0])
    projected = cv2.perspectiveTransform(refer_kps, h_ref_to_img)
    return [tuple(map(float, p[0])) for p in projected]


def _homography_confidence(
    pred_points: List[Point],
    h_ref_to_img: np.ndarray | None,
    prev_h_ref_to_img: np.ndarray | None,
) -> float:
    if h_ref_to_img is None:
        return 0.0

    projected = cv2.perspectiveTransform(refer_kps, h_ref_to_img)
    errors = []
    for i in range(min(len(pred_points), projected.shape[0])):
        px, py = pred_points[i]
        if px is None or py is None:
            continue
        rx, ry = projected[i][0]
        errors.append(float(np.hypot(float(px) - rx, float(py) - ry)))

    if errors:
        err_mean = float(np.mean(errors))
        conf_err = float(np.exp(-err_mean / 45.0))
    else:
        conf_err = 0.12

    if prev_h_ref_to_img is not None:
        denom = max(1e-6, float(np.linalg.norm(prev_h_ref_to_img)))
        diff = float(np.linalg.norm(h_ref_to_img - prev_h_ref_to_img) / denom)
        conf_temp = float(np.exp(-2.0 * diff))
    else:
        conf_temp = 1.0

    return float(max(0.0, min(1.0, 0.75 * conf_err + 0.25 * conf_temp)))


def _letterbox(frame: np.ndarray, dst_w: int, dst_h: int):
    src_h, src_w = frame.shape[:2]
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((dst_h, dst_w, 3), dtype=frame.dtype)
    pad_x = (dst_w - new_w) // 2
    pad_y = (dst_h - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


@torch.no_grad()
def infer_points(model, image: np.ndarray, device: str, use_refine_kps: bool = False, use_homography: bool = False) -> List[Point]:
    output_w, output_h = 640, 360
    src_h, src_w = image.shape[:2]
    img, scale, pad_x, pad_y = _letterbox(image, output_w, output_h)
    inp = np.rollaxis((img.astype(np.float32) / 255.0), 2, 0)
    inp = torch.tensor(inp).unsqueeze(0).float().to(device)

    out = model(inp)[0]
    pred = torch.sigmoid(out).detach().cpu().numpy()

    points: List[Point] = []
    for kps_num in range(14):
        heatmap = (pred[kps_num] * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, scale=1, low_thresh=170, max_radius=25)
        if x_pred is None or y_pred is None:
            points.append((None, None))
            continue

        x_src = (x_pred - pad_x) / scale
        y_src = (y_pred - pad_y) / scale
        if not (0.0 <= x_src < src_w and 0.0 <= y_src < src_h):
            points.append((None, None))
            continue

        if use_refine_kps and kps_num not in [8, 12, 9]:
            x_src, y_src = refine_kps(image, int(y_src), int(x_src))
        points.append((float(x_src), float(y_src)))

    if use_homography:
        matrix_trans = get_trans_matrix(points)
        if matrix_trans is not None:
            points = [tuple(np.squeeze(x)) for x in cv2.perspectiveTransform(refer_kps, matrix_trans)]

    return points


def draw_points(image: np.ndarray, points: List[Point]) -> np.ndarray:
    out = image.copy()

    # Draw court layout lines when both endpoints are available.
    line_pairs = [
        (0, 1),   # top baseline
        (2, 3),   # bottom baseline
        (0, 2),   # left doubles sideline
        (1, 3),   # right doubles sideline
        (4, 5),   # left singles sideline
        (6, 7),   # right singles sideline
        (8, 9),   # top service line
        (10, 11), # bottom service line
        (12, 13), # center service line
    ]
    for i, j in line_pairs:
        if i >= len(points) or j >= len(points):
            continue
        x1, y1 = points[i]
        x2, y2 = points[j]
        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue
        out = cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    for x, y in points:
        if x is None or y is None:
            continue
        out = cv2.circle(out, (int(x), int(y)), radius=0, color=(0, 255, 255), thickness=6)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--normalize-fps", type=int, default=0, help="If > 0, preprocess input video with MoviePy to this FPS.")
    parser.add_argument("--use-refine-kps", action="store_true")
    parser.add_argument("--use-homography", action="store_true")
    parser.add_argument("--no-homography-stabilization", action="store_true", help="Disable temporal homography stabilization for video inference.")
    parser.add_argument("--homography-min-confidence", type=float, default=0.24)
    parser.add_argument("--homography-carry-frames", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def _infer_video_stream(
    model,
    input_path: str,
    output_path: str,
    device: str,
    use_refine_kps: bool,
    use_homography: bool,
    stabilize_homography: bool,
    homography_min_confidence: float,
    homography_carry_frames: int,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot read input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output video writer: {output_path}")

    progress = tqdm(total=total_frames, desc="Court layout", unit="frame")
    last_reliable_h_ref_to_img: np.ndarray | None = None
    last_reliable_conf = 0.0
    last_reliable_idx = -999999
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_points = infer_points(model, frame, device, use_refine_kps, use_homography=False)
            points = raw_points

            if stabilize_homography:
                h_ref_to_img = get_trans_matrix(raw_points)
                conf = _homography_confidence(raw_points, h_ref_to_img, last_reliable_h_ref_to_img)
                carried = False

                if h_ref_to_img is not None and conf >= homography_min_confidence:
                    points = _project_reference_points(h_ref_to_img)
                    last_reliable_h_ref_to_img = h_ref_to_img
                    last_reliable_conf = conf
                    last_reliable_idx = frame_idx
                elif last_reliable_h_ref_to_img is not None and (frame_idx - last_reliable_idx) <= homography_carry_frames:
                    points = _project_reference_points(last_reliable_h_ref_to_img)
                    conf = max(0.0, last_reliable_conf * (0.85 ** (frame_idx - last_reliable_idx)))
                    carried = True
                else:
                    # Prefer no overlay over clearly wrong overlay.
                    points = [(None, None)] * len(raw_points)

                if frame_idx % 20 == 0:
                    progress.set_postfix({"hom_conf": f"{conf:.2f}", "carried": int(carried)})
            elif use_homography:
                h_ref_to_img = get_trans_matrix(raw_points)
                points = _project_reference_points(h_ref_to_img)

            out.write(draw_points(frame, points))
            progress.update(1)
            frame_idx += 1
    finally:
        progress.close()
        cap.release()
        out.release()


def main():
    args = parse_args()
    device = get_device(args.device)

    model = CourtKeypointNet(out_channels=15).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    if args.input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        input_path = args.input_path
        if args.normalize_fps > 0:
            out_path = Path(args.output_path)
            normalized_input = out_path.parent / f"{out_path.stem}_input_{args.normalize_fps}fps.mp4"
            input_path = normalize_video_fps_moviepy(args.input_path, str(normalized_input), target_fps=args.normalize_fps)

        _infer_video_stream(
            model=model,
            input_path=input_path,
            output_path=args.output_path,
            device=device,
            use_refine_kps=args.use_refine_kps,
            use_homography=args.use_homography,
            stabilize_homography=not args.no_homography_stabilization,
            homography_min_confidence=args.homography_min_confidence,
            homography_carry_frames=args.homography_carry_frames,
        )
        return

    image = cv2.imread(args.input_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read input image: {args.input_path}")
    points = infer_points(model, image, device, args.use_refine_kps, args.use_homography)
    cv2.imwrite(args.output_path, draw_points(image, points))


if __name__ == "__main__":
    main()
