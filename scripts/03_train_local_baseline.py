from __future__ import annotations

from pathlib import Path

import pandas as pd

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
from src.config import load_config
from src.data import split_stratified
from src.fl_adapters import SkLogRegSGD
from src.metrics import ap, safe_roc_auc, precision_recall_f1_at_threshold, best_f1_threshold
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
    ALPHA_GRID = cfg.baseline.alpha_grid
    max_rounds = cfg.baseline.max_rounds
    patience = cfg.baseline.patience
    local_epochs = cfg.fl.local_epochs
    seed = cfg.project.seed

    cat_cols = cfg.schema.cat_cols
    num_cols = cfg.schema.num_cols
    feat_cols = num_cols + cat_cols
    label_col = "is_sar"

    banks = find_banks()
    if not banks:
        raise RuntimeError(f"No banks found under {DATA_PROCESSED}")

    summary_rows: list[dict] = []

    for bank in banks:
        in_path = DATA_PROCESSED / bank / "processed.parquet"
        if not in_path.exists():
            raise FileNotFoundError(f"Missing processed file: {in_path}")

        df = pd.read_parquet(in_path)

        if label_col not in df.columns:
            raise KeyError(f"{bank}: missing required columns {label_col}")

        train_df, val_df, test_df = split_stratified(
            df,
            label_col=label_col,
            train_frac=0.7,
            val_frac=0.15,
            test_frac=0.15,
            seed=seed,
        )
        y_train = train_df[label_col].astype(int).to_numpy()
        y_val = val_df[label_col].astype(int).to_numpy()
        y_test = test_df[label_col].astype(int).to_numpy()

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
        print(f"best alpha={best_alpha:g} | val_ap={best_val_ap:.6f} at round {best_round} ")

        if best_params is None:
            raise RuntimeError(f"{bank}: no model params saved for alpha={best_alpha:g}")
        model = SkLogRegSGD(d=X_test.shape[1], alpha=best_alpha, seed=seed)
        model.set_params(best_params)

        # 1) Select threshold on validation set
        val_scores = model.predict_scores(X_val)
        thr, p_val, r_val, f1_val = best_f1_threshold(y_val, val_scores)
        print(f"Selected threshold on val: thr={thr:.6f} | P={p_val:.4f} R={r_val:.4f} F1={f1_val:.4f}")


        # 2) Evaluate on test with that threshold
        test_scores = model.predict_scores(X_test)
        test_auc = safe_roc_auc(y_test, test_scores)
        p_test, r_test, f1_test = precision_recall_f1_at_threshold(y_test, test_scores, thr)
        test_ap = float(ap(y_test, test_scores)) if y_test.sum() > 0 else 0.0
        
        print(f"Test results: test_ap={test_ap:.6f} F1={f1_test:.4f} AUC={test_auc:.6f} | P={p_test:.4f} R={r_test:.4f}")





if __name__ == "__main__":
    main()
