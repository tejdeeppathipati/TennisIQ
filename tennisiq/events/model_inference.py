from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from tennisiq.events.features import FEATURE_NAMES, feature_matrix


class EventModelScorer:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.model = None
        self._load_error = None
        self._try_load_model(model_path)

    def _try_load_model(self, model_path: str | None) -> None:
        if not model_path:
            return
        p = Path(model_path)
        if not p.exists():
            return
        try:
            import catboost as ctb

            model = ctb.CatBoostClassifier()
            model.load_model(str(p))
            self.model = model
            self._load_error = None
        except Exception as e:  # pragma: no cover - best effort
            self.model = None
            self._load_error = str(e)

    def model_loaded(self) -> bool:
        return self.model is not None

    def heuristic_score(self, row: Dict[str, float]) -> float:
        # Deterministic fallback scoring in [0, 1].
        score = 0.0
        score += min(max(row.get("rule_score", 0.0), 0.0), 1.0) * 0.35
        score += float(row.get("reversal", 0.0)) * 0.25
        score += float(row.get("speed_drop", 0.0)) * 0.2

        speed = abs(float(row.get("speed", 0.0)))
        accel = abs(float(row.get("accel", 0.0)))
        hom = float(row.get("homography_confidence", 0.0))

        score += min(speed / 1200.0, 1.0) * 0.08
        score += min(accel / 25000.0, 1.0) * 0.07
        score += min(max(hom, 0.0), 1.0) * 0.05
        return float(min(max(score, 0.0), 1.0))

    def score(self, feature_rows: Iterable[Dict[str, float]]) -> List[float]:
        rows = list(feature_rows)
        if not rows:
            return []

        if self.model is None:
            return [self.heuristic_score(r) for r in rows]

        x = feature_matrix(rows)
        try:
            if hasattr(self.model, "predict_proba"):
                p = self.model.predict_proba(x)
                p = np.asarray(p)
                if p.ndim == 2 and p.shape[1] > 1:
                    return [float(v) for v in p[:, 1]]
                return [float(v) for v in p.ravel()]
            pred = self.model.predict(x)
            return [float(v) for v in np.asarray(pred).ravel()]
        except Exception:
            return [self.heuristic_score(r) for r in rows]


def temporal_nms(index_score_pairs: List[Tuple[int, float]], window: int = 3) -> List[Tuple[int, float]]:
    if not index_score_pairs:
        return []

    index_score_pairs = sorted(index_score_pairs, key=lambda x: x[1], reverse=True)
    chosen: List[Tuple[int, float]] = []
    blocked: set[int] = set()

    for idx, score in index_score_pairs:
        if idx in blocked:
            continue
        chosen.append((idx, score))
        for k in range(idx - window, idx + window + 1):
            blocked.add(k)

    chosen.sort(key=lambda x: x[0])
    return chosen


def score_candidates(
    candidates: List[Dict],
    scorer: EventModelScorer,
    threshold: float,
    nms_window: int,
) -> List[Dict]:
    if not candidates:
        return []

    rows = [c["features"] for c in candidates]
    scores = scorer.score(rows)
    for c, s in zip(candidates, scores):
        c["score"] = float(s)

    selected = [(c["frame_idx"], c["score"]) for c in candidates if c["score"] >= threshold]
    kept = temporal_nms(selected, window=nms_window)
    kept_idx = {idx for idx, _ in kept}

    out = []
    for c in candidates:
        if c["frame_idx"] in kept_idx and c["score"] >= threshold:
            out.append(c)
    out.sort(key=lambda x: x["frame_idx"])
    return out
