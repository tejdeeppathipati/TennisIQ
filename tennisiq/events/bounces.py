from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm


Point = Tuple[Optional[float], Optional[float]]


def detect_bounces(ball_track: List[Point], min_window: int = 2) -> List[int]:
    bounces = []
    ys = [p[1] for p in ball_track]
    for i in range(min_window, len(ys) - min_window):
        if ys[i] is None:
            continue
        prev_vals = [v for v in ys[i - min_window : i] if v is not None]
        next_vals = [v for v in ys[i + 1 : i + 1 + min_window] if v is not None]
        if not prev_vals or not next_vals:
            continue
        if ys[i] > max(prev_vals) and ys[i] > max(next_vals):
            bounces.append(i)
    return bounces


def create_features(path_dataset: str, num_frames: int = 3) -> pd.DataFrame:
    import pandas as pd

    dataset_dir = Path(path_dataset)
    games = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])

    frames = []
    for game in tqdm(games):
        clips = [p for p in (dataset_dir / game).iterdir() if p.is_dir()]
        for clip in clips:
            labels_path = clip / "Label.csv"
            if not labels_path.exists():
                continue
            labels = pd.read_csv(labels_path).copy()

            eps = 1e-15
            for i in range(1, num_frames):
                labels[f"x_lag_{i}"] = labels["x-coordinate"].shift(i)
                labels[f"x_lag_inv_{i}"] = labels["x-coordinate"].shift(-i)
                labels[f"y_lag_{i}"] = labels["y-coordinate"].shift(i)
                labels[f"y_lag_inv_{i}"] = labels["y-coordinate"].shift(-i)
                labels[f"x_diff_{i}"] = abs(labels[f"x_lag_{i}"] - labels["x-coordinate"])
                labels[f"y_diff_{i}"] = labels[f"y_lag_{i}"] - labels["y-coordinate"]
                labels[f"x_diff_inv_{i}"] = abs(labels[f"x_lag_inv_{i}"] - labels["x-coordinate"])
                labels[f"y_diff_inv_{i}"] = labels[f"y_lag_inv_{i}"] - labels["y-coordinate"]
                labels[f"x_div_{i}"] = abs(labels[f"x_diff_{i}"] / (labels[f"x_diff_inv_{i}"] + eps))
                labels[f"y_div_{i}"] = labels[f"y_diff_{i}"] / (labels[f"y_diff_inv_{i}"] + eps)

            labels["target"] = (labels["status"] == 2).astype(int)
            for i in range(1, num_frames):
                labels = labels[labels[f"x_lag_{i}"].notna()]
                labels = labels[labels[f"x_lag_inv_{i}"].notna()]
            labels = labels[labels["x-coordinate"].notna()]
            labels["status"] = labels["status"].astype(int)
            frames.append(labels)

    if not frames:
        raise RuntimeError(f"No valid Label.csv files found in {path_dataset}")
    return pd.concat(frames, ignore_index=True)


def create_train_test(df: pd.DataFrame, num_frames: int = 3):
    from sklearn.model_selection import train_test_split

    colnames_x = [f"x_diff_{i}" for i in range(1, num_frames)] + [f"x_diff_inv_{i}" for i in range(1, num_frames)] + [f"x_div_{i}" for i in range(1, num_frames)]
    colnames_y = [f"y_diff_{i}" for i in range(1, num_frames)] + [f"y_diff_inv_{i}" for i in range(1, num_frames)] + [f"y_div_{i}" for i in range(1, num_frames)]
    colnames = colnames_x + colnames_y

    df_train, df_test = train_test_split(df, test_size=0.25, random_state=5)
    X_train = df_train[colnames]
    X_test = df_test[colnames]
    y_train = df_train["target"]
    y_test = df_test["target"]
    return X_train, y_train, X_test, y_test


def train_bounce_model(path_dataset: str, path_save_model: str, num_feature_frames: int = 3):
    import catboost as ctb
    from sklearn.metrics import accuracy_score, confusion_matrix

    df_features = create_features(path_dataset, num_feature_frames)
    X_train, y_train, X_test, y_test = create_train_test(df_features, num_feature_frames)

    train_dataset = ctb.Pool(X_train, y_train)
    model = ctb.CatBoostRegressor(loss_function="RMSE", verbose=False)
    grid = {
        "iterations": [150, 200],
        "learning_rate": [0.03, 0.1],
        "depth": [2, 4, 6],
        "l2_leaf_reg": [0.2, 0.5, 1, 3],
    }
    model.grid_search(grid, train_dataset, verbose=False)

    pred = model.predict(X_test)
    y_pred_bin = (pred > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_bin).ravel()
    acc = accuracy_score(y_test, y_pred_bin)
    print(f"tn={tn}, fp={fp}, fn={fn}, tp={tp}")
    print(f"accuracy={acc:.5f}")

    model.save_model(path_save_model)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-dataset", type=str, required=True, help="path to TrackNet raw dataset")
    parser.add_argument("--path-save-model", type=str, required=True, help="path to output .cbm model")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_bounce_model(args.path_dataset, args.path_save_model)
