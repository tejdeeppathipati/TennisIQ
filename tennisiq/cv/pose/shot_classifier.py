"""
Shot type classifier using a GRU-based RNN on MoveNet pose sequences.

Architecture (matching tennis_rnn.h5):
    Input:  (batch, 30, 26) — 30 frames × 13 keypoints × (y, x)
    GRU:    24 hidden units, reset_after=True (Keras GRUv2 convention)
    Dense:  24 → 8 (ReLU)
    Output: 8 → 4 (softmax) → [backhand, forehand, neutral, serve]

Weights are loaded from the Keras .h5 file via h5py, then mapped into
PyTorch tensors. This eliminates the TensorFlow runtime dependency.

Falls back to a geometric heuristic when no model is available.
"""
import logging
import os
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tennisiq.cv.pose.inference_movenet import (
    NUM_POSE_FEATURES,
    PlayerPose,
    extract_pose_features,
    extract_pose_sequence,
)

logger = logging.getLogger(__name__)

SHOT_CLASSES = ["backhand", "forehand", "neutral", "serve"]
SHOT_CLASS_INDEX = {name: i for i, name in enumerate(SHOT_CLASSES)}


@dataclass
class ShotClassification:
    """Result of shot type classification."""
    shot_type: str               # one of SHOT_CLASSES
    confidence: float            # 0-1
    probabilities: dict[str, float]


# ── PyTorch model matching the Keras GRU architecture ────────────────────────

class _ShotRNN(nn.Module):
    """GRU(26→24) → Dense(24→8, relu) → Dense(8→4, softmax)."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=NUM_POSE_FEATURES, hidden_size=24, batch_first=True)
        self.fc1 = nn.Linear(24, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)            # h: (1, batch, 24)
        h = h.squeeze(0)              # (batch, 24)
        h = F.relu(self.fc1(h))       # (batch, 8)
        return F.softmax(self.fc2(h), dim=-1)  # (batch, 4)


def _load_keras_gru_weights(model: _ShotRNN, h5_path: str) -> None:
    """
    Transfer weights from a Keras GRU .h5 file into a PyTorch _ShotRNN.

    Keras GRU (reset_after=True) stores:
      kernel:           (input_size, 3*units) — [z, r, h] gates
      recurrent_kernel: (units, 3*units)
      bias:             (2, 3*units)  — bias[0] = input bias, bias[1] = recurrent bias

    PyTorch GRU stores:
      weight_ih_l0: (3*units, input_size) — [r, z, n] gates (different order!)
      weight_hh_l0: (3*units, units)
      bias_ih_l0:   (3*units,)
      bias_hh_l0:   (3*units,)
    """
    with h5py.File(h5_path, "r") as f:
        kernel = f["model_weights/gru/gru/gru_cell/kernel:0"][:]         # (26, 72)
        rec_kernel = f["model_weights/gru/gru/gru_cell/recurrent_kernel:0"][:]  # (24, 72)
        bias = f["model_weights/gru/gru/gru_cell/bias:0"][:]            # (2, 72)

        dense1_w = f["model_weights/dense/dense/kernel:0"][:]            # (24, 8)
        dense1_b = f["model_weights/dense/dense/bias:0"][:]              # (8,)
        dense2_w = f["model_weights/dense_1/dense_1/kernel:0"][:]        # (8, 4)
        dense2_b = f["model_weights/dense_1/dense_1/bias:0"][:]          # (4,)

    units = 24

    # Keras gate order: [z, r, h] → PyTorch gate order: [r, z, n]
    def _reorder_gates(w, axis):
        """Reorder from Keras [z, r, h] to PyTorch [r, z, n] along given axis."""
        z, r, h = np.split(w, 3, axis=axis)
        return np.concatenate([r, z, h], axis=axis)

    # Input weights: Keras (input, 3*units) → PyTorch (3*units, input), reorder gates
    kernel_reordered = _reorder_gates(kernel, axis=1)
    model.gru.weight_ih_l0.data = torch.from_numpy(kernel_reordered.T.copy())

    # Recurrent weights: Keras (units, 3*units) → PyTorch (3*units, units)
    rec_reordered = _reorder_gates(rec_kernel, axis=1)
    model.gru.weight_hh_l0.data = torch.from_numpy(rec_reordered.T.copy())

    # Bias: Keras stores (2, 3*units) — row 0 = input bias, row 1 = recurrent bias
    bias_ih = _reorder_gates(bias[0:1], axis=1).squeeze(0)
    bias_hh = _reorder_gates(bias[1:2], axis=1).squeeze(0)
    model.gru.bias_ih_l0.data = torch.from_numpy(bias_ih.copy())
    model.gru.bias_hh_l0.data = torch.from_numpy(bias_hh.copy())

    # Dense layers: Keras (in, out) → PyTorch (out, in)
    model.fc1.weight.data = torch.from_numpy(dense1_w.T.copy())
    model.fc1.bias.data = torch.from_numpy(dense1_b.copy())
    model.fc2.weight.data = torch.from_numpy(dense2_w.T.copy())
    model.fc2.bias.data = torch.from_numpy(dense2_b.copy())


# ── Public classifier API ────────────────────────────────────────────────────

class ShotClassifier:
    """Classifies shots using a GRU RNN on 30-frame pose sequences."""

    def __init__(self, model_path: str | None = None, model_type: str = "rnn"):
        self.model: _ShotRNN | None = None
        self.model_type = model_type
        self._available = False

        if model_path and os.path.isfile(model_path):
            try:
                self.model = _ShotRNN()
                _load_keras_gru_weights(self.model, model_path)
                self.model.eval()
                self._available = True
                logger.info(f"Shot classifier loaded (PyTorch GRU from {model_path})")
            except Exception as e:
                logger.warning(f"Failed to load shot classifier: {e}; using heuristic fallback")
                self.model = None
        else:
            logger.info("No shot classifier model provided; using heuristic fallback")

    @property
    def available(self) -> bool:
        return self._available

    def classify_shot_event(
        self,
        poses: dict[int, dict[str, PlayerPose]],
        owner: str,
        frame_idx: int,
    ) -> ShotClassification:
        """Classify a shot event using the RNN on a 30-frame window, or heuristic."""
        if self._available and self.model is not None:
            seq = extract_pose_sequence(poses, owner, frame_idx)
            if seq is not None and seq.shape == (30, NUM_POSE_FEATURES):
                # Guard: only run RNN if at least a few frames have real pose data
                # (non-zero rows). All-zero input means no MoveNet data available —
                # feeding it to the RNN produces a confident but wrong prediction.
                non_zero_frames = int(np.any(seq != 0, axis=1).sum())
                if non_zero_frames >= 3:
                    inp = torch.from_numpy(seq).unsqueeze(0)  # (1, 30, 26)
                    with torch.no_grad():
                        probs = self.model(inp)[0].numpy()     # (4,)
                    idx = int(np.argmax(probs))
                    return ShotClassification(
                        shot_type=SHOT_CLASSES[idx],
                        confidence=float(probs[idx]),
                        probabilities={name: float(probs[i]) for i, name in enumerate(SHOT_CLASSES)},
                    )

        # Heuristic fallback (also used when pose data is insufficient for RNN)
        if frame_idx in poses and owner in poses[frame_idx]:
            return _heuristic_classify(poses[frame_idx][owner])

        # No pose data at all — return neutral
        return ShotClassification(
            shot_type="neutral",
            confidence=0.0,
            probabilities={c: 0.25 for c in SHOT_CLASSES},
        )


def _heuristic_classify(pose: PlayerPose) -> ShotClassification:
    """
    Heuristic shot classification based on wrist position relative to shoulders.

    MoveNet keypoints are in **normalized [0,1] bbox coordinates** (y, x, conf).
    In a typical standing player crop the torso spans ~0.3 in Y, so thresholds
    are expressed relative to torso height.
    """
    kps = pose.keypoints  # (17, 3) — y, x, conf

    l_shoulder = kps[5]
    r_shoulder = kps[6]
    l_elbow = kps[7]
    r_elbow = kps[8]
    l_wrist = kps[9]
    r_wrist = kps[10]

    CONF_MIN = 0.1
    has_left = l_shoulder[2] > CONF_MIN and l_wrist[2] > CONF_MIN
    has_right = r_shoulder[2] > CONF_MIN and r_wrist[2] > CONF_MIN

    if not has_left and not has_right:
        return ShotClassification("neutral", 0.3, {c: 0.25 for c in SHOT_CLASSES})

    l_hip, r_hip = kps[11], kps[12]
    mid_shoulder_y = (l_shoulder[0] + r_shoulder[0]) / 2
    mid_hip_y = (l_hip[0] + r_hip[0]) / 2
    torso = max(abs(mid_hip_y - mid_shoulder_y), 0.05)

    l_elev = (l_shoulder[0] - l_wrist[0]) / torso if has_left else -99
    r_elev = (r_shoulder[0] - r_wrist[0]) / torso if has_right else -99
    max_elev = max(l_elev, r_elev)

    if max_elev > 0.6:
        conf = min(0.8, 0.4 + max_elev * 0.2)
        return ShotClassification("serve", conf,
            {"serve": conf, "forehand": (1 - conf) / 3, "backhand": (1 - conf) / 3, "neutral": (1 - conf) / 3})

    l_lateral = abs(l_wrist[1] - l_shoulder[1]) / torso if has_left else 0
    r_lateral = abs(r_wrist[1] - r_shoulder[1]) / torso if has_right else 0

    mid_x = (l_shoulder[1] + r_shoulder[1]) / 2

    if l_lateral > 0.3 or r_lateral > 0.3:
        if r_lateral > l_lateral:
            is_cross = r_wrist[1] < mid_x
            shot = "backhand" if is_cross else "forehand"
        else:
            is_cross = l_wrist[1] > mid_x
            shot = "forehand" if is_cross else "backhand"

        conf = min(0.7, 0.35 + max(l_lateral, r_lateral) * 0.15)
        probs = {c: (1 - conf) / 3 for c in SHOT_CLASSES}
        probs[shot] = conf
        return ShotClassification(shot, conf, probs)

    if l_lateral > 0.15 or r_lateral > 0.15:
        if r_lateral > l_lateral:
            is_cross = r_wrist[1] < mid_x
            shot = "backhand" if is_cross else "forehand"
        else:
            is_cross = l_wrist[1] > mid_x
            shot = "forehand" if is_cross else "backhand"
        return ShotClassification(shot, 0.35,
            {shot: 0.35, "neutral": 0.30, "forehand" if shot != "forehand" else "backhand": 0.20, "serve": 0.15})

    return ShotClassification("neutral", 0.4, {"neutral": 0.4, "forehand": 0.25, "backhand": 0.25, "serve": 0.1})
