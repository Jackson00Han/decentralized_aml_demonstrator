from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from src.data_splits import split_fixed_windows
from src.fl_protocol import GlobalPlan


def plan_fingerprint(plan_path: Path) -> str:
    return hashlib.sha1(plan_path.read_bytes()).hexdigest()


def cache_dir(client_out: Path, bank: str) -> Path:
    return client_out / bank / "cache"


def load_cache(cache_path: Path, expected_hash: str) -> dict[str, Any] | None:
    meta_path = cache_path / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("plan_hash") != expected_hash:
        return None

    def _load_x(name: str):
        p = cache_path / f"{name}.npz"
        if not p.exists():
            return None
        return sparse.load_npz(p)

    def _load_y(name: str):
        p = cache_path / f"{name}.npy"
        if not p.exists():
            return None
        return np.load(p, allow_pickle=False)

    X_train = _load_x("X_train")
    X_val = _load_x("X_val")
    X_test = _load_x("X_test")
    X_trainval = _load_x("X_trainval")
    y_train = _load_y("y_train")
    y_val = _load_y("y_val")
    y_test = _load_y("y_test")
    y_trainval = _load_y("y_trainval")

    if any(v is None for v in [X_train, X_val, X_test, X_trainval, y_train, y_val, y_test, y_trainval]):
        return None

    counts = meta.get("counts", {})
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "X_trainval": X_trainval,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_trainval": y_trainval,
        "counts": counts,
    }


def save_cache(
    cache_path: Path,
    plan_hash: str,
    plan: GlobalPlan,
    payload: dict[str, Any],
) -> None:
    cache_path.mkdir(parents=True, exist_ok=True)

    def _save_x(name: str, X):
        sparse.save_npz(cache_path / f"{name}.npz", sparse.csr_matrix(X))

    def _save_y(name: str, y: np.ndarray):
        np.save(cache_path / f"{name}.npy", y)

    _save_x("X_train", payload["X_train"])
    _save_x("X_val", payload["X_val"])
    _save_x("X_test", payload["X_test"])
    _save_x("X_trainval", payload["X_trainval"])
    _save_y("y_train", payload["y_train"])
    _save_y("y_val", payload["y_val"])
    _save_y("y_test", payload["y_test"])
    _save_y("y_trainval", payload["y_trainval"])

    meta = {
        "plan_hash": plan_hash,
        "schema_version": plan.schema_version,
        "counts": payload.get("counts", {}),
    }
    (cache_path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_cache(
    bank: str,
    data_processed: Path,
    plan: GlobalPlan,
    preprocess,
    ts_col: str = "tran_timestamp",
) -> dict[str, Any]:
    df = pd.read_parquet(data_processed / bank / f"{bank}_merged.parquet")
    tr, va, te, df_use = split_fixed_windows(df, ts_col=ts_col)

    feat_cols = plan.feature_schema.num_cols + plan.feature_schema.cat_cols
    X_train = preprocess.transform(tr[feat_cols])
    X_val = preprocess.transform(va[feat_cols])
    X_test = preprocess.transform(te[feat_cols])

    y_train = tr["y"].astype(int).to_numpy()
    y_val = va["y"].astype(int).to_numpy()
    y_test = te["y"].astype(int).to_numpy()

    X_trainval = sparse.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "X_trainval": X_trainval,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_trainval": y_trainval,
        "counts": {
            "data_used_n": int(len(df_use)),
            "train_n": int(len(tr)),
            "val_n": int(len(va)),
            "test_n": int(len(te)),
            "train_pos": int(tr["y"].sum()),
            "val_pos": int(va["y"].sum()),
            "test_pos": int(te["y"].sum()),
        },
    }


def load_or_build_cache(
    bank: str,
    data_processed: Path,
    client_out: Path,
    plan_path: Path,
    plan: GlobalPlan,
    preprocess,
    ts_col: str = "tran_timestamp",
) -> dict[str, Any]:
    cache_path = cache_dir(client_out, bank)
    plan_hash = plan_fingerprint(plan_path)
    cached = load_cache(cache_path, plan_hash)
    if cached is not None:
        return cached

    payload = build_cache(bank, data_processed, plan, preprocess, ts_col=ts_col)
    save_cache(cache_path, plan_hash, plan, payload)
    return payload
