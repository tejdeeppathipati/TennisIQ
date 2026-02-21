from __future__ import annotations

import argparse
from typing import List, Tuple

import cv2
import numpy as np
import torch

from tennisiq.common import get_device
from tennisiq.cv.court.model import CourtKeypointNet
from tennisiq.cv.court.postprocess import postprocess, refine_kps
from tennisiq.geometry.homography import get_trans_matrix, refer_kps
from tennisiq.io.video import read_video, write_video


Point = Tuple[float | None, float | None]


@torch.no_grad()
def infer_points(model, image: np.ndarray, device: str, use_refine_kps: bool = False, use_homography: bool = False) -> List[Point]:
    output_w, output_h = 640, 360
    img = cv2.resize(image, (output_w, output_h))
    inp = np.rollaxis((img.astype(np.float32) / 255.0), 2, 0)
    inp = torch.tensor(inp).unsqueeze(0).float().to(device)

    out = model(inp)[0]
    pred = torch.sigmoid(out).detach().cpu().numpy()

    points: List[Point] = []
    for kps_num in range(14):
        heatmap = (pred[kps_num] * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
        if use_refine_kps and kps_num not in [8, 12, 9] and x_pred is not None and y_pred is not None:
            x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
        points.append((x_pred, y_pred))

    if use_homography:
        matrix_trans = get_trans_matrix(points)
        if matrix_trans is not None:
            points = [tuple(np.squeeze(x)) for x in cv2.perspectiveTransform(refer_kps, matrix_trans)]

    return points


def draw_points(image: np.ndarray, points: List[Point]) -> np.ndarray:
    out = image.copy()
    for x, y in points:
        if x is None or y is None:
            continue
        out = cv2.circle(out, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=10)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
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
        frames, fps = read_video(args.input_path)
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
