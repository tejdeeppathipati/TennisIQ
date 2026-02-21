from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
from torch.utils.data import Dataset

from tennisiq.cv.court.utils import draw_umich_gaussian, line_intersection


class CourtDataset(Dataset):
    def __init__(
        self,
        mode: str,
        data_root: str = "data/datasets/court_identification",
        input_height: int = 720,
        input_width: int = 1280,
        scale: int = 2,
        hp_radius: int = 55,
    ) -> None:
        assert mode in ["train", "val"], "mode must be train or val"
        self.mode = mode
        self.data_root = Path(data_root)
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = input_height // scale
        self.output_width = input_width // scale
        self.num_joints = 14
        self.hp_radius = hp_radius
        self.scale = scale

        self.path_images = self._resolve_images_dir()
        self.data = self._load_annotations()
        print(f"mode={mode}, samples={len(self.data)}")

    def _resolve_images_dir(self) -> Path:
        candidates = [self.data_root / "images", self.data_root / "imgs"]
        for c in candidates:
            if c.exists():
                return c
        return self.data_root / "images"

    def _load_annotations(self):
        candidates = [
            self.data_root / f"data_{self.mode}.json",
            self.data_root / f"{self.mode}.json",
            self.data_root / f"annotations_{self.mode}.json",
        ]
        for p in candidates:
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    return json.load(f)
        raise FileNotFoundError(f"Missing annotation json for mode={self.mode} under {self.data_root}")

    def _resolve_image_path(self, image_id: str) -> Path:
        if "." in image_id:
            p = self.path_images / image_id
            if p.exists():
                return p
        for ext in [".png", ".jpg", ".jpeg"]:
            p = self.path_images / f"{image_id}{ext}"
            if p.exists():
                return p
        return self.path_images / f"{image_id}.png"

    @staticmethod
    def _to_xy_list(raw_kps: Sequence) -> List[List[float]]:
        out = []
        for kp in raw_kps:
            if kp is None or len(kp) < 2:
                out.append([-1, -1])
                continue
            x, y = kp[0], kp[1]
            x = -1 if x is None else float(x)
            y = -1 if y is None else float(y)
            out.append([x, y])
        return out

    def __getitem__(self, index: int):
        sample = self.data[index]
        image_id = sample["id"]
        kps = self._to_xy_list(sample["kps"])

        img_path = self._resolve_image_path(image_id)
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        img = cv2.resize(img, (self.output_width, self.output_height))
        inp = np.rollaxis(img.astype(np.float32) / 255.0, 2, 0)

        hm_hp = np.zeros((self.num_joints + 1, self.output_height, self.output_width), dtype=np.float32)
        for i, (x, y) in enumerate(kps[: self.num_joints]):
            if 0 <= x <= self.input_width and 0 <= y <= self.input_height:
                draw_umich_gaussian(hm_hp[i], (int(x / self.scale), int(y / self.scale)), self.hp_radius)

        # Add center point if outer-corner lines can be intersected.
        if len(kps) >= 4:
            ct = line_intersection((kps[0][0], kps[0][1], kps[3][0], kps[3][1]), (kps[1][0], kps[1][1], kps[2][0], kps[2][1]))
            if ct is not None:
                draw_umich_gaussian(hm_hp[self.num_joints], (int(ct[0] / self.scale), int(ct[1] / self.scale)), self.hp_radius)

        return inp, hm_hp, np.array(kps, dtype=np.int32), str(image_id)

    def __len__(self):
        return len(self.data)
