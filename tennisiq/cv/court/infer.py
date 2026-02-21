from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from tennisiq.common import get_device
from tennisiq.cv.court.model import CourtKeypointNet
from tennisiq.cv.court.postprocess import postprocess, refine_kps
from tennisiq.geometry.homography import get_trans_matrix, refer_kps
from tennisiq.io.video import normalize_video_fps_moviepy, read_video, write_video


Point = Tuple[float | None, float | None]


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
        out = cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

    for x, y in points:
        if x is None or y is None:
            continue
        out = cv2.circle(out, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=7)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--normalize-fps", type=int, default=0, help="If > 0, preprocess input video with MoviePy to this FPS.")
    parser.add_argument("--use-refine-kps", action="store_true")
    parser.add_argument("--use-homography", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


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

        frames, fps = read_video(input_path)
        out_frames = []
        for frame in frames:
            points = infer_points(model, frame, device, args.use_refine_kps, args.use_homography)
            out_frames.append(draw_points(frame, points))
        write_video(out_frames, fps, args.output_path)
        return

    image = cv2.imread(args.input_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read input image: {args.input_path}")
    points = infer_points(model, image, device, args.use_refine_kps, args.use_homography)
    cv2.imwrite(args.output_path, draw_points(image, points))


if __name__ == "__main__":
    main()
