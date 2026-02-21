from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from tennisiq.common import ensure_dir


def export_json(path: str, payload: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def export_jsonl(path: str, rows: Iterable[Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
