from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from tennisiq.events.features import build_candidate_feature_row, compute_track_kinematics
from tennisiq.events.model_inference import EventModelScorer, score_candidates


Point = Tuple[Optional[float], Optional[float]]


def detect_bounce_candidates(
    ball_track: Sequence[Point],
    ball_court_track: Sequence[Point] | None = None,
    net_y: float | None = None,
    fps: int = 30,
    min_speed_drop_ratio: float = 0.15,
) -> List[Dict]:
    kin = compute_track_kinematics(ball_track, fps=fps)
    n = len(ball_track)
    candidates: List[Dict] = []

    for i in range(2, n - 2):
        prev = kin[i - 1]
        curr = kin[i]
        nxt = kin[i + 1]

        if not bool(prev.get("visible")) or not bool(curr.get("visible")) or not bool(nxt.get("visible")):
            continue

        prev_vy = float(prev.get("vy", np.nan))
        next_vy = float(nxt.get("vy", np.nan))
        prev_speed = float(prev.get("speed", np.nan))
        curr_speed = float(curr.get("speed", np.nan))
        next_speed = float(nxt.get("speed", np.nan))

        if not np.isfinite(prev_vy) or not np.isfinite(next_vy):
            continue
        if not np.isfinite(prev_speed) or not np.isfinite(curr_speed) or not np.isfinite(next_speed):
            continue

        reversal = prev_vy > 0 and next_vy < 0
        ref_speed = max((prev_speed + next_speed) / 2.0, 1e-6)
        speed_drop = curr_speed < (1.0 - min_speed_drop_ratio) * ref_speed

        near_net = False
        if ball_court_track is not None and net_y is not None and i < len(ball_court_track):
            by = ball_court_track[i][1]
            if by is not None:
                near_net = abs(float(by) - float(net_y)) < 180.0

        # Candidate generation: require reversal with either speed drop or court-gated cue.
        if reversal and (speed_drop or near_net):
            rule_score = 0.0
            rule_score += 0.55 if reversal else 0.0
            rule_score += 0.35 if speed_drop else 0.0
            rule_score += 0.10 if near_net else 0.0
            candidates.append(
                {
                    "frame_idx": i,
                    "reversal": reversal,
                    "speed_drop": speed_drop,
                    "near_net": near_net,
                    "rule_score": float(min(rule_score, 1.0)),
                    "reasons": [
                        x
                        for x, ok in [
                            ("reversal", reversal),
                            ("speed_drop", speed_drop),
                            ("near_net", near_net),
                        ]
                        if ok
                    ],
                }
            )

    return candidates


def score_bounce_candidates(
    candidates: List[Dict],
    ball_track: Sequence[Point],
    mapped_frames: Sequence[Dict],
    net_y: float,
    fps: int,
    model_path: str | None = None,
    threshold: float = 0.5,
    nms_window: int = 3,
) -> List[Dict]:
    if not candidates:
        return []

    kinematics = compute_track_kinematics(ball_track, fps=fps)
    with_features: List[Dict] = []
    for c in candidates:
        idx = c["frame_idx"]
        feat = build_candidate_feature_row(
            frame_idx=idx,
            candidate=c,
            kinematics=kinematics,
            ball_track=ball_track,
            mapped_frames=mapped_frames,
            net_y=net_y,
        )
        payload = dict(c)
        payload["features"] = feat
        with_features.append(payload)

    scorer = EventModelScorer(model_path=model_path)
    return score_candidates(with_features, scorer=scorer, threshold=threshold, nms_window=nms_window)


def detect_bounces(
    ball_track: Sequence[Point],
    fps: int = 30,
    threshold: float = 0.5,
) -> List[int]:
    # Backward-compatible wrapper used in some parts of the code.
    candidates = detect_bounce_candidates(ball_track, fps=fps)
    dummy_mapped = [{"ball_court_xy": (None, None), "playerA_court_xy": (None, None), "playerB_court_xy": (None, None), "homography_confidence": 0.0} for _ in ball_track]
    scored = score_bounce_candidates(
        candidates=candidates,
        ball_track=ball_track,
        mapped_frames=dummy_mapped,
        net_y=0.0,
        fps=fps,
        model_path=None,
        threshold=threshold,
        nms_window=3,
    )
    return [int(x["frame_idx"]) for x in scored]


def create_features(path_dataset: str, num_frames: int = 3):
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


def create_train_test(df, num_frames: int = 3):
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
    model = ctb.CatBoostClassifier(loss_function="Logloss", verbose=False)
    grid = {
        "iterations": [150, 200],
        "learning_rate": [0.03, 0.1],
        "depth": [2, 4, 6],
        "l2_leaf_reg": [0.2, 0.5, 1, 3],
    }
    model.grid_search(grid, train_dataset, verbose=False)

    pred = model.predict_proba(X_test)[:, 1]
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
