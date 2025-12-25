# src/utils.py
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from scipy import sparse
import numpy as np


def plan_hash(plan_path: Path) -> str:
    return hashlib.sha1(plan_path.read_bytes()).hexdigest()

def load_dataset(ds_dir: Path, expected_plan_hash: str) -> dict:
    meta_path = ds_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing dataset meta: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("plan_hash") != expected_plan_hash:
        raise ValueError(
            f"Plan hash mismatch for dataset {ds_dir} (expected {expected_plan_hash}, got {meta.get('plan_hash')})"
        )

    def _load_x(name: str):
        return sparse.load_npz(ds_dir / f"{name}.npz")

    def _load_y(name: str):
        return np.load(ds_dir / f"{name}.npy", allow_pickle=False)

    return {
        "meta": meta,
        "X_train": _load_x("X_train"),
        "X_val": _load_x("X_val"),
        "X_test": _load_x("X_test"),
        "X_trainval": _load_x("X_trainval"),
        "y_train": _load_y("y_train"),
        "y_val": _load_y("y_val"),
        "y_test": _load_y("y_test"),
        "y_trainval": _load_y("y_trainval"),
    }