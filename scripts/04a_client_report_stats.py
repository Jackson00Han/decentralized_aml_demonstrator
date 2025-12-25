# 04a_client_report_stats.py
from __future__ import annotations

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd



def main(bank: str):
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))
    from src.config import load_config
    from src.data_splits import split_fixed_windows

    cfg = load_config()

    DATA_PROCESSED = cfg.paths.data_processed
    CLIENT_OUT = cfg.paths.out_fl_clients

    CAT_COLS = cfg.schema.cat_cols
    NUM_COLS = cfg.schema.num_cols

    p = DATA_PROCESSED / bank / f"{bank}_merged.parquet"
    df = pd.read_parquet(p)

    # Fixed-window split (UTC)
    tr, va, te, df_use = split_fixed_windows(df)

    # stats must NOT use val/test to avoid leakage
    tr_only = tr

    # categorical vocab sets (train only)
    cat_sets = {}
    for c in CAT_COLS:
        # normalize to string + dropna
        cat_sets[c] = sorted(tr_only[c].dropna().astype(str).unique().tolist())

    # numeric stats (train only): n, sum, sumsq -> server can aggregate to mean/std
    num_stats = {}
    for c in NUM_COLS:
        x = pd.to_numeric(tr_only[c], errors="coerce").to_numpy()
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
        }
    }

    out_dir = CLIENT_OUT / bank; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir/'stats.json'} | train={len(tr)} val={len(va)} test={len(te)}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", default='bank_a') # default bank_a for a quick test
    args = ap.parse_args()
    main(args.client)
