from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import load_config
from src.data_splits import split_fixed_windows
from src.fl_protocol import GlobalPlan
from src.fl_preprocess import build_preprocessor


def plan_hash(plan_path: Path) -> str:
    return hashlib.sha1(plan_path.read_bytes()).hexdigest()


def main(bank: str, out_dir: str | None = None, overwrite: bool = False) -> None:
    cfg = load_config()
    server_out = cfg.paths.out_fl_server
    plan_path = server_out / "global_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing global plan: {plan_path} (run 04b_server_build_global_plan.py)")

    plan = GlobalPlan.load(plan_path)
    preprocess = build_preprocessor(plan)

    data_processed = cfg.paths.data_processed
    df = pd.read_parquet(data_processed / bank / f"{bank}_merged.parquet")
    tr, va, te, df_use = split_fixed_windows(df, ts_col="tran_timestamp")

    feat_cols = plan.feature_schema.num_cols + plan.feature_schema.cat_cols
    X_train = preprocess.transform(tr[feat_cols])
    X_val = preprocess.transform(va[feat_cols])
    X_test = preprocess.transform(te[feat_cols])

    y_train = tr["y"].astype(int).to_numpy()
    y_val = va["y"].astype(int).to_numpy()
    y_test = te["y"].astype(int).to_numpy()

    X_trainval = sparse.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    base_out = Path(out_dir) if out_dir else (cfg.paths.out_fl_clients / bank / "datasets")
    ds_dir = base_out / f"plan_{plan_hash(plan_path)[:8]}"
    if ds_dir.exists() and not overwrite:
        raise FileExistsError(f"Dataset already exists: {ds_dir} (use --overwrite)")
    ds_dir.mkdir(parents=True, exist_ok=True)

    sparse.save_npz(ds_dir / "X_train.npz", sparse.csr_matrix(X_train))
    sparse.save_npz(ds_dir / "X_val.npz", sparse.csr_matrix(X_val))
    sparse.save_npz(ds_dir / "X_test.npz", sparse.csr_matrix(X_test))
    sparse.save_npz(ds_dir / "X_trainval.npz", sparse.csr_matrix(X_trainval))
    np.save(ds_dir / "y_train.npy", y_train)
    np.save(ds_dir / "y_val.npy", y_val)
    np.save(ds_dir / "y_test.npy", y_test)
    np.save(ds_dir / "y_trainval.npy", y_trainval)

    meta = {
        "bank": bank,
        "plan_hash": plan_hash(plan_path),
        "schema_version": plan.schema_version,
        "feature_dim": int(X_train.shape[1]),
        "counts": {
            "data_used_n": int(len(df_use)),
            "train_n": int(len(tr)),
            "val_n": int(len(va)),
            "test_n": int(len(te)),
            "train_pos": int(tr["y"].sum()),
            "val_pos": int(va["y"].sum()),
            "test_pos": int(te["y"].sum()),
        }
    }
    (ds_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] wrote dataset for {bank} -> {ds_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default="bank_a")
    parser.add_argument("--out_dir", default=None, help="override output dir for dataset")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(args.client, out_dir=args.out_dir, overwrite=args.overwrite)
