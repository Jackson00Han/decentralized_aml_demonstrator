from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
from src.config import load_config
from src.data_splits import split_fixed_windows
from src.fl_adapters import SkLogRegSGD
from src.metrics import ap, safe_roc_auc, topk_report
from src.fl_preprocess import build_local_preprocessor
cfg = load_config()
DATA_PROCESSED = cfg.paths.data_processed


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
    ALPHA_GRID = cfg.baseline.alpha_grid
    max_rounds = cfg.baseline.max_rounds
    patience = cfg.baseline.patience
    local_epochs = cfg.fl.local_epochs
    seed = cfg.project.seed
    cat_cols = cfg.schema.cat_cols
    num_cols = cfg.schema.num_cols
    feat_cols = num_cols + cat_cols

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

        preprocess = build_local_preprocessor(cat_cols, num_cols)
        preprocess.fit(train_df[feat_cols])

        X_train = preprocess.transform(train_df[feat_cols])
        X_val = preprocess.transform(val_df[feat_cols])
        X_test = preprocess.transform(test_df[feat_cols])

        print("\n" + "=" * 68)
        print(f"Local baseline | {bank}")
        print("=" * 68)
        print(
            f"Split: train={len(train_df)} (pos={int(y_train.sum())}) | "
            f"val={len(val_df)} (pos={int(y_val.sum())}) | "
            f"test={len(test_df)} (pos={int(y_test.sum())})"
        )

        # Tune alpha by val AP
        cand = []
        for alpha in ALPHA_GRID:
            model = SkLogRegSGD(d=X_train.shape[1], alpha=alpha, seed=seed)
            best_val_ap = -1.0
            best_params = None
            best_round = 0
            no_improve = 0
            for r in range(1, max_rounds + 1):
                model.train_one_round(X_train, y_train, local_epochs=local_epochs, seed=seed + r)
                val_scores = model.predict_scores(X_val)
                val_ap = float(ap(y_val, val_scores)) if y_val.sum() > 0 else 0.0
                if val_ap > best_val_ap + 1e-9:
                    best_val_ap = val_ap
                    best_params = model.get_params()
                    best_round = r
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break
            cand.append((float(alpha), best_val_ap, best_params, best_round))

        best_alpha, best_val_ap, best_params, best_round = max(cand, key=lambda x: x[1])
        tune_str = " | ".join([f"alpha={c:g}: {avp:.6f}@r{r}" for c, avp, _, r in cand])
        print(f"Tune alpha (val AP): {tune_str}")
        print(f"Selected: alpha={best_alpha:g} (val_AP={best_val_ap:.6f})")

        if best_params is None:
            raise RuntimeError(f"{bank}: no model params saved for alpha={best_alpha:g}")
        model = SkLogRegSGD(d=X_test.shape[1], alpha=best_alpha, seed=seed)
        model.set_params(best_params)
        test_scores = model.predict_scores(X_test)
        test_ap = float(ap(y_test, test_scores)) if y_test.sum() > 0 else 0.0
        test_auc = safe_roc_auc(y_test, test_scores)
        rep = topk_report(y_test, test_scores, k=TOP_K)
        print(f"Test AP: {fmt(test_ap)} | Test ROC AUC: {fmt(test_auc)}")
        print("Top-k report:")
        print(json.dumps(rep, indent=2, sort_keys=True))

        bank_out = OUT_ROOT / bank; bank_out.mkdir(parents=True, exist_ok=True)

        metrics = {
            "bank": bank,
            "selected_alpha": best_alpha,
            "selected_round": best_round,
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
                "selected_alpha": best_alpha,
                "selected_round": best_round,
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
