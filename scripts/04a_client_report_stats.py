from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data_splits import split_fixed_windows

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
CLIENT_OUT = REPO_ROOT / "outputs" / "fl_clients"

CAT_COLS = ["orig_state", "bene_state"]
NUM_COLS = ["base_amt", "orig_initial_deposit", "bene_initial_deposit"]


def main(bank: str):
    p = DATA_PROCESSED / bank / f"{bank}_merged.parquet"
    df = pd.read_parquet(p)

    # Fixed-window split (UTC), train includes 2017-01-01..2018-04-30
    tr, va, te, df_use = split_fixed_windows(df, ts_col="tran_timestamp")

    # stats must NOT use test to avoid leakage
    tv = pd.concat([tr, va], ignore_index=True)

    # categorical vocab sets (train+val)
    cat_sets = {}
    for c in CAT_COLS:
        # normalize to string + dropna
        cat_sets[c] = sorted(tv[c].dropna().astype(str).unique().tolist())

    # numeric stats (train+val): n, sum, sumsq -> server can aggregate to mean/std
    num_stats = {}
    for c in NUM_COLS:
        x = pd.to_numeric(tv[c], errors="coerce").to_numpy()
        x = x[np.isfinite(x)]
        n = int(x.size)
        s = float(x.sum()) if n else 0.0
        ss = float((x * x).sum()) if n else 0.0
        num_stats[c] = {"n": n, "sum": s, "sumsq": ss}

    out = {
        "bank": bank,
        "cat_sets": cat_sets,
        "num_stats": num_stats,
        "counts": {
            "data_used_n": int(len(df_use)),
            "train_n": int(len(tr)),
            "val_n": int(len(va)),
            "test_n": int(len(te)),
            "train_pos": int(tr["y"].sum()),
            "val_pos": int(va["y"].sum()),
            "test_pos": int(te["y"].sum()),
        },
        "split_rule": {
            "train": "[2017-01-01, 2018-05-01)",
            "val":   "[2018-05-01, 2018-09-01)",
            "test":  "[2018-09-01, 2019-01-01)",
            "timezone": "UTC",
        }
    }

    out_dir = CLIENT_OUT / bank
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir/'stats.json'} | train={len(tr)} val={len(va)} test={len(te)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", default='bank_a') # default bank for a quick test
    args = ap.parse_args()
    main(args.client)
