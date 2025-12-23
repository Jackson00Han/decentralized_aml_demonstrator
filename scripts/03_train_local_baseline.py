from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
from src.config import load_config
from src.data_splits import split_fixed_windows
from src.metrics import ap, safe_roc_auc, topk_report
cfg = load_config()
DATA_PROCESSED = cfg.paths.data_processed

def make_model(cat_cols: list[str], num_cols: list[str], C: float) -> Pipeline:
    cat_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num_pipe = Pipeline([("scaler", StandardScaler())])
    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(C=float(C), max_iter=200, n_jobs=1, solver="lbfgs")
    return Pipeline([("pre", pre), ("clf", clf)])

def find_banks() -> list[str]:
    if not DATA_PROCESSED.exists():
        return []
    return sorted([p.name for p in DATA_PROCESSED.iterdir() if p.is_dir()]) # iterdir() returns the immediate children only

def fmt(x: float | None, nd: int = 6) -> str:
    if x is None:
        return "NA"
    return f"{float(x):.{nd}f}"

def main() -> None:

    OUT_ROOT = cfg.paths.out_local_baseline; OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TOP_K = cfg.baseline.top_k
    C_GRID = cfg.baseline.c_grid

    banks = find_banks()
    if not banks:
        raise RuntimeError(f"No banks found under {DATA_PROCESSED}")

    summary_rows: list[dict] = []

    for bank in banks:
        in_path = DATA_PROCESSED / bank / f"{bank}_merged.parquet"
        if not in_path.exists():
            raise FileNotFoundError(f"Missing processed file: {in_path}")

        df = pd.read_parquet(in_path)

        if "y" not in df.columns or "tran_timestamp" not in df.columns:
            raise KeyError(f"{bank}: missing required columns y / tran_timestamp")

        train_df, val_df, test_df, df_use = split_fixed_windows(df, ts_col="tran_timestamp")
        y_train = train_df["y"].astype(int).to_numpy()
        y_val = val_df["y"].astype(int).to_numpy()
        y_test = test_df["y"].astype(int).to_numpy()

        feature_cols = [c for c in df_use.columns if c not in {"y", "tran_timestamp"}]
        X_train = train_df[feature_cols]
        X_val = val_df[feature_cols]
        X_test = test_df[feature_cols]

        cat_cols = cfg.schema.cat_cols
        num_cols = cfg.schema.num_cols

        print("\n" + "=" * 68)
        print(f"Local baseline | {bank}")
        print("=" * 68)
        print(
            f"Split: train={len(train_df)} (pos={int(y_train.sum())}) | "
            f"val={len(val_df)} (pos={int(y_val.sum())}) | "
            f"test={len(test_df)} (pos={int(y_test.sum())})"
        )

        # Tune C by val AP
        cand = []
        for C in C_GRID:
            model = make_model(cat_cols, num_cols, C=C)
            model.fit(X_train, y_train)
            val_scores = model.predict_proba(X_val)[:, 1]
            val_ap = float(ap(y_val, val_scores)) if y_val.sum() > 0 else 0.0
            cand.append((float(C), val_ap))

        best_C, best_val_ap = max(cand, key=lambda x: x[1])
        tune_str = " | ".join([f"C={c:g}: {avp:.6f}" for c, avp in cand])
        print(f"Tune C (val AP): {tune_str}")
        print(f"Selected: C={best_C:g} (val_AP={best_val_ap:.6f})")

        # Train final on train+val
        trainval_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
        y_trainval = trainval_df["y"].astype(int).to_numpy()
        X_trainval = trainval_df[feature_cols]

        model = make_model(cat_cols, num_cols, C=best_C)
        model.fit(X_trainval, y_trainval)

        test_scores = model.predict_proba(X_test)[:, 1]
        test_ap = float(ap(y_test, test_scores)) if y_test.sum() > 0 else 0.0
        test_auc = safe_roc_auc(y_test, test_scores)
        rep = topk_report(y_test, test_scores, k=TOP_K)
        print(f"Test AP: {fmt(test_ap)} | Test ROC AUC: {fmt(test_auc)}")
        print("Top-k report:")
        print(json.dumps(rep, indent=2, sort_keys=True))

        bank_out = OUT_ROOT / bank; bank_out.mkdir(parents=True, exist_ok=True)

        metrics = {
            "bank": bank,
            "selected_C": best_C,
            "val_ap_best": best_val_ap,
            "test_ap": test_ap,
            "test_roc_auc": test_auc,
            "topk": rep,
        }

        out_json = bank_out / "val_test_report.json"
        out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved: {out_json}")

        summary_rows.append(
            {
                "bank": bank,
                "test_ap": test_ap,
                "roc_auc": test_auc if test_auc is not None else np.nan,
                "pos": rep["pos"],
                "n": rep["n"],
                f"hits@{rep['k']}": rep["hits"],
                f"expected_hits_random@{rep['k']}": rep["expected_hits_random"],
                f"P@{rep['k']}": rep["precision_at_k"],
                f"base_P@{rep['k']}": rep["baseline_precision_at_k"],
                f"R@{rep['k']}": rep["recall_at_k"],
                f"base_R@{rep['k']}": rep["baseline_recall_at_k"],
                f"liftP@{rep['k']}": rep["lift_precision_at_k"] if rep["lift_precision_at_k"] is not None else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("bank")
    out_csv = OUT_ROOT / "summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print("\n" + "=" * 68)
    print("Summary")
    print("=" * 68)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
