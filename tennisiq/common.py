from __future__ import annotations

from pathlib import Path
from typing import Union

import torch


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Union[str, Path]) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p


def ensure_dir(path: Union[str, Path]) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device(preferred: str = "auto") -> str:
    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferred == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # auto priority: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
