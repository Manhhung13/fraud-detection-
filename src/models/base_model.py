from __future__ import annotations

from pathlib import Path

import joblib


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy model file: {path}")

    return joblib.load(path)