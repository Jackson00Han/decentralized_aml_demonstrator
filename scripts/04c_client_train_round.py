from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data_splits import split_fixed_windows
from src.fl_protocol import GlobalPlan, load_params_npz, save_params_npz
from src.fl_preprocess import build_preprocessor
from src.fl_adapters import SkLogRegSGD
from src.metrics import ap

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
CLIENT_OUT = REPO_ROOT / "outputs" / "fl_clients"
SERVER_OUT = REPO_ROOT / "outputs" / "fl_server"


def main(bank: str, round_id: int, local_epochs: int = 2, seed: int = 42):
    # load plan + build preprocessor (fixed vocab + fixed mean/std)
    plan = GlobalPlan.load(SERVER_OUT / "global_plan.json")
    preprocess = build_preprocessor(plan)

    # load global model (server publishes latest)
    global_model_path = SERVER_OUT / "global_model_latest.npz"
    if global_model_path.exists():
        params = load_params_npz(global_model_path)
    else:
        # initialize from scratch (need feature dimension)
        dummy = pd.DataFrame(
            {
                **{c: [plan.global_categories[c][0] if len(plan.global_categories[c]) else "NA"]
                   for c in plan.feature_schema.cat_cols},
                **{c: [0.0] for c in plan.feature_schema.num_cols},
            }
        )
        d = preprocess.transform(dummy[plan.feature_schema.num_cols + plan.feature_schema.cat_cols]).shape[1]
        params = {"coef": np.zeros((1, d), dtype=float), "intercept": np.zeros((1,), dtype=float)}

    # load data
    df = pd.read_parquet(DATA_PROCESSED / bank / f"{bank}_merged.parquet")

    # fixed-window split
    tr, va, te, df_use = split_fixed_windows(df, ts_col="tran_timestamp")

    # build X/y
    feat_cols = plan.feature_schema.num_cols + plan.feature_schema.cat_cols
    X_tr = preprocess.transform(tr[feat_cols])
    y_tr = tr["y"].astype(int).to_numpy()

    X_va = preprocess.transform(va[feat_cols])
    y_va = va["y"].astype(int).to_numpy()

    # local train
    d = X_tr.shape[1]
    model = SkLogRegSGD(d=d, seed=seed + round_id)
    model.set_params(params)
    model.train_one_round(X_tr, y_tr, local_epochs=local_epochs, seed=round_id)

    # local val metric (scalar only)
    scores_va = model.predict_scores(X_va)
    val_ap = ap(y_va, scores_va)

    # write update
    out_dir = CLIENT_OUT / bank / "updates"
    out_dir.mkdir(parents=True, exist_ok=True)

    update_params = model.get_params()
    save_params_npz(
        out_dir / f"round_{round_id:03d}_update.npz",
        update_params,
        meta={
            "bank": bank,
            "round": int(round_id),
            "n_train": int(X_tr.shape[0]),
            "val_n": int(len(y_va)),
            "val_pos": int(y_va.sum()),
            "val_ap": float(val_ap),
            "schema_version": plan.schema_version,
            "split_rule": "fixed_windows_2017_to_2018",
        },
    )

    print(
        f"[OK] {bank} round {round_id:03d} | "
        f"train={len(tr)} val={len(va)} test={len(te)} | val_ap={val_ap:.6f}"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default="bank_a")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args.client, args.round, local_epochs=args.local_epochs, seed=args.seed)

