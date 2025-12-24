from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(
    bank: str,
    round_id: int,
    alpha_override: float | None = None,
    use_trainval: bool = False,
):
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))
    from src.fl_cache import load_or_build_cache
    from src.fl_protocol import GlobalPlan, load_params_npz, save_params_npz
    from src.fl_preprocess import build_preprocessor
    from src.fl_adapters import SkLogRegSGD
    from src.metrics import ap
    from src.config import load_config
    cfg = load_config()

    local_epochs = cfg.fl.local_epochs
    alpha = float(alpha_override) if alpha_override is not None else cfg.fl.alpha
    seed = cfg.project.seed

    DATA_PROCESSED = cfg.paths.data_processed
    CLIENT_OUT = cfg.paths.out_fl_clients
    SERVER_OUT = cfg.paths.out_fl_server
    # load plan + build preprocessor (fixed vocab + fixed mean/std)
    plan_path = SERVER_OUT / "global_plan.json"
    plan = GlobalPlan.load(plan_path)
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

    cache = load_or_build_cache(
        bank=bank,
        data_processed=DATA_PROCESSED,
        client_out=CLIENT_OUT,
        plan_path=plan_path,
        plan=plan,
        preprocess=preprocess,
        ts_col="tran_timestamp",
    )
    counts = cache.get("counts", {})
    train_n = int(counts.get("train_n", len(cache["y_train"])))
    val_n_all = int(counts.get("val_n", len(cache["y_val"])))
    test_n = int(counts.get("test_n", len(cache["y_test"])))

    # build X/y
    if use_trainval:
        X_tr = cache["X_trainval"]
        y_tr = cache["y_trainval"].astype(int)
        val_ap = None
        val_n = 0
        val_pos = 0
        n_train = int(y_tr.shape[0])
    else:
        X_tr = cache["X_train"]
        y_tr = cache["y_train"].astype(int)
        X_va = cache["X_val"]
        y_va = cache["y_val"].astype(int)
        val_n = int(len(y_va))
        val_pos = int(y_va.sum())
        n_train = int(len(y_tr))

    # local train
    d = X_tr.shape[1]
    model = SkLogRegSGD(d=d, alpha=alpha, seed=seed + round_id)
    model.set_params(params)
    model.train_one_round(X_tr, y_tr, local_epochs=local_epochs, seed=seed + round_id)

    # local val metric (scalar only)
    if not use_trainval:
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
            "n_train": int(n_train),
            "val_n": int(val_n),
            "val_pos": int(val_pos),
            "val_ap": float(val_ap) if val_ap is not None else 0.0,
            "schema_version": plan.schema_version,
            "split_rule": "fixed_windows_2017_to_2018",
        },
    )

    val_ap_disp = f"{val_ap:.6f}" if val_ap is not None else "NA"
    print(
        f"[OK] {bank} round {round_id:03d} | "
        f"train={train_n} val={val_n_all} test={test_n} | val_ap={val_ap_disp}"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default="bank_a")
    parser.add_argument("--round_id", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=None, help="override fl.alpha for this run")
    parser.add_argument("--use_trainval", action="store_true", help="train on train+val (skip val metrics)")
    args = parser.parse_args()

    main(args.client, args.round_id, alpha_override=args.alpha, use_trainval=args.use_trainval)
