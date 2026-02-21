from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def gaussian_kernel(size: int, variance: float):
    x, y = np.mgrid[-size : size + 1, -size : size + 1]
    return np.exp(-(x**2 + y**2) / float(2 * variance))


def create_gaussian(size: int, variance: float):
    g = gaussian_kernel(size, variance)
    g = g * 255 / g[int(len(g) / 2)][int(len(g) / 2)]
    return g.astype(int)


def create_gt_images(path_input: str, path_output: str, size: int, variance: float, width: int, height: int) -> None:
    gaussian_kernel_array = create_gaussian(size, variance)
    for game_id in range(1, 11):
        game = f"game{game_id}"
        game_dir = Path(path_input) / game
        if not game_dir.exists():
            continue
        clips = os.listdir(game_dir)
        for clip in clips:
            print(f"game={game} clip={clip}")
            out_clip = Path(path_output) / "gts" / game / clip
            out_clip.mkdir(parents=True, exist_ok=True)

            labels = pd.read_csv(game_dir / clip / "Label.csv")
            for idx in range(labels.shape[0]):
                file_name, vis, x, y, _ = labels.loc[idx, :]
                heatmap = np.zeros((height, width, 3), dtype=np.uint8)
                if vis != 0 and not (math.isnan(x) or math.isnan(y)):
                    x = int(x)
                    y = int(y)
                    for i in range(-size, size + 1):
                        for j in range(-size, size + 1):
                            if 0 <= x + i < width and 0 <= y + j < height:
                                temp = gaussian_kernel_array[i + size][j + size]
                                if temp > 0:
                                    heatmap[y + j, x + i] = (temp, temp, temp)
                cv2.imwrite(str(out_clip / file_name), heatmap)


def create_gt_labels(path_input: str, path_output: str, train_rate: float = 0.7) -> None:
    frames = []
    for game_id in range(1, 11):
        game = f"game{game_id}"
        game_dir = Path(path_input) / game
        if not game_dir.exists():
            continue
        clips = os.listdir(game_dir)
        for clip in clips:
            labels = pd.read_csv(game_dir / clip / "Label.csv")
            labels = labels.copy()
            labels["gt_path"] = "gts/" + game + "/" + clip + "/" + labels["file name"]
            labels["path1"] = "images/" + game + "/" + clip + "/" + labels["file name"]
            labels_target = labels.iloc[2:].copy()
            labels_target.loc[:, "path2"] = list(labels["path1"].iloc[1:-1])
            labels_target.loc[:, "path3"] = list(labels["path1"].iloc[:-2])
            frames.append(labels_target)

    if not frames:
        raise RuntimeError("No labels found to create train/val CSVs")

    df = pd.concat(frames, ignore_index=True)
    df = df[["path1", "path2", "path3", "gt_path", "x-coordinate", "y-coordinate", "status", "visibility"]]
    df = df.sample(frac=1.0, random_state=5).reset_index(drop=True)

    num_train = int(df.shape[0] * train_rate)
    df.iloc[:num_train].to_csv(Path(path_output) / "labels_train.csv", index=False)
    df.iloc[num_train:].to_csv(Path(path_output) / "labels_val.csv", index=False)


def prepare_tracknet_dataset(raw_data_dir: str, output_dir: str, size: int = 20, variance: float = 10, width: int = 1280, height: int = 720) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # link/copy raw frames layout into images/ if missing
    images_dir = output / "images"
    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        for game_id in range(1, 11):
            game = f"game{game_id}"
            src = Path(raw_data_dir) / game
            dst = images_dir / game
            if src.exists() and not dst.exists():
                os.symlink(src.resolve(), dst)

    create_gt_images(raw_data_dir, output_dir, size, variance, width, height)
    create_gt_labels(raw_data_dir, output_dir)


class TrackNetDataset(Dataset):
    def __init__(self, mode: str, dataset_root: str = "data/datasets/balltracking", input_height: int = 360, input_width: int = 640):
        assert mode in ["train", "val"], "mode must be train or val"
        self.path_dataset = Path(dataset_root)
        self.height = input_height
        self.width = input_width

        labels_path = self.path_dataset / f"labels_{mode}.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")
        self.data = pd.read_csv(labels_path)
        print(f"mode={mode}, samples={self.data.shape[0]}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        path, path_prev, path_preprev, path_gt, x, y, status, vis = self.data.loc[idx, :]

        path = self.path_dataset / path
        path_prev = self.path_dataset / path_prev
        path_preprev = self.path_dataset / path_preprev
        path_gt = self.path_dataset / path_gt
        if isinstance(x, float) and math.isnan(x):
            x, y = -1, -1

        inputs = self.get_input(str(path), str(path_prev), str(path_preprev))
        outputs = self.get_output(str(path_gt))

        return inputs, outputs, x, y, int(vis)

    def get_output(self, path_gt: str):
        img = cv2.imread(path_gt)
        if img is None:
            raise FileNotFoundError(f"Missing GT image: {path_gt}")
        img = cv2.resize(img, (self.width, self.height))
        img = img[:, :, 0]
        img = np.reshape(img, (self.width * self.height))
        return img

    def get_input(self, path: str, path_prev: str, path_preprev: str):
        img = cv2.imread(path)
        img_prev = cv2.imread(path_prev)
        img_preprev = cv2.imread(path_preprev)
        if img is None or img_prev is None or img_preprev is None:
            raise FileNotFoundError(f"Missing frame among: {path}, {path_prev}, {path_preprev}")

        img = cv2.resize(img, (self.width, self.height))
        img_prev = cv2.resize(img_prev, (self.width, self.height))
        img_preprev = cv2.resize(img_preprev, (self.width, self.height))

        imgs = np.concatenate((img, img_prev, img_preprev), axis=2).astype(np.float32) / 255.0
        return np.rollaxis(imgs, 2, 0)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/datasets/balltracking")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare_tracknet_dataset(args.raw_data, args.output)
