from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_splits import split_fixed_windows
from src.fl_protocol import GlobalPlan, load_params_npz
from src.fl_preprocess import build_preprocessor
from src.fl_adapters import SkLogRegSGD
from src.metrics import ap, safe_roc_auc, topk

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
CLIENT_OUT = REPO_ROOT / "outputs" / "fl_clients"
SERVER_OUT = REPO_ROOT / "outputs" / "fl_server"

TOP_K = 500


def main(bank: str, use_best: bool = True):
    plan = GlobalPlan.load(SERVER_OUT / "global_plan.json")
    preprocess = build_preprocessor(plan)

    model_path = SERVER_OUT / ("best_model.npz" if use_best else "global_model_latest.npz")
    params = load_params_npz(model_path)

    df = pd.read_parquet(DATA_PROCESSED / bank / f"{bank}_merged.parquet")

    # fixed-window split
    tr, va, te, df_use = split_fixed_windows(df, ts_col="tran_timestamp")

    feat_cols = plan.feature_schema.num_cols + plan.feature_schema.cat_cols
    X_te = preprocess.transform(te[feat_cols])
    y_te = te["y"].astype(int).to_numpy()

    d = X_te.shape[1]
    model = SkLogRegSGD(d=d)
    model.set_params(params)
    scores = model.predict_scores(X_te)

    res = {
        "bank": bank,
        "which_model": "best_model" if use_best else "latest_model",
        "test_n": int(len(te)),
        "test_pos": int(y_te.sum()),
        "test_ap": ap(y_te, scores),
        "test_roc_auc": safe_roc_auc(y_te, scores),
        "topk": topk(y_te, scores, TOP_K),
        "split_rule": {
            "train": "[2017-01-01, 2018-05-01)",
            "val":   "[2018-05-01, 2018-09-01)",
            "test":  "[2018-09-01, 2019-01-01)",
            "timezone": "UTC",
        },
    }

    out_dir = CLIENT_OUT / bank
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_test.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir/'metrics_test.json'} | test_ap={res['test_ap']:.6f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--use_best", action="store_true", help="evaluate best_model.npz (default)")
    ap.add_argument("--use_latest", action="store_true", help="evaluate global_model_latest.npz")
    args = ap.parse_args()

    use_best = True
    if args.use_latest:
        use_best = False

    main(args.bank, use_best=use_best)
