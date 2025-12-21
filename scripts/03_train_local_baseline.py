from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
OUT_ROOT = REPO_ROOT / "outputs" / "local_baseline"

TOP_K = 500
C_GRID = [0.1, 1.0]


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def split_by_time(
    df: pd.DataFrame,
    ts_col: str = "tran_timestamp",
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test


def topk_report(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict:
    n = len(y_true)
    k = int(min(k, n))
    order = np.argsort(-scores)
    top = y_true[order[:k]]

    pos = int(y_true.sum())
    hits = int(top.sum())
    p_at_k = hits / k if k > 0 else 0.0
    r_at_k = hits / pos if pos > 0 else 0.0

    base_p = pos / n if n > 0 else 0.0
    base_r = k / n if n > 0 else 0.0

    lift_p = (p_at_k / base_p) if base_p > 0 else None
    lift_r = (r_at_k / base_r) if base_r > 0 else None

    return {
        "n": n,
        "pos": pos,
        "k": k,
        "hits": hits,
        "precision_at_k": p_at_k,
        "recall_at_k": r_at_k,
        "baseline_precision_at_k": base_p,
        "baseline_recall_at_k": base_r,
        "lift_precision_at_k": lift_p,
        "lift_recall_at_k": lift_r,
        "expected_hits_random": k * base_p,
    }


def make_model(cat_cols: list[str], num_cols: list[str], C: float) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(C=float(C), max_iter=200, n_jobs=1, solver="lbfgs")
    return Pipeline([("pre", pre), ("clf", clf)])


def find_banks() -> list[str]:
    if not DATA_PROCESSED.exists():
        return []
    return sorted([p.name for p in DATA_PROCESSED.iterdir() if p.is_dir()])


def fmt(x: float | None, nd: int = 6) -> str:
    if x is None:
        return "NA"
    return f"{float(x):.{nd}f}"


def main() -> None:
    banks = find_banks()
    if not banks:
        raise RuntimeError(f"No banks found under {DATA_PROCESSED}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for bank in banks:
        in_path = DATA_PROCESSED / bank / f"{bank}_merged.parquet"
        if not in_path.exists():
            raise FileNotFoundError(f"Missing processed file: {in_path}")

        df = pd.read_parquet(in_path)

        if "y" not in df.columns or "tran_timestamp" not in df.columns:
            raise KeyError(f"{bank}: missing required columns y / tran_timestamp")

        df["tran_timestamp"] = pd.to_datetime(df["tran_timestamp"], utc=True, errors="coerce")
        if df["tran_timestamp"].isna().any():
            raise ValueError(f"{bank}: tran_timestamp has NaT")

        train_df, val_df, test_df = split_by_time(df)
        y_train = train_df["y"].astype(int).to_numpy()
        y_val = val_df["y"].astype(int).to_numpy()
        y_test = test_df["y"].astype(int).to_numpy()

        feature_cols = [c for c in df.columns if c not in {"y", "tran_timestamp"}]
        X_train = train_df[feature_cols]
        X_val = val_df[feature_cols]
        X_test = test_df[feature_cols]

        cat_cols = ["orig_state", "bene_state"]
        num_cols = ["base_amt", "orig_initial_deposit", "bene_initial_deposit"]

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
            val_ap = float(average_precision_score(y_val, val_scores)) if y_val.sum() > 0 else 0.0
            cand.append((float(C), val_ap))

        best_C, best_val_ap = max(cand, key=lambda x: x[1])
        tune_str = " | ".join([f"C={c:g}: {ap:.6f}" for c, ap in cand])
        print(f"Tune C (val AP): {tune_str}")
        print(f"Selected: C={best_C:g} (val_AP={best_val_ap:.6f})")

        # Train final on train+val
        trainval_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
        y_trainval = trainval_df["y"].astype(int).to_numpy()
        X_trainval = trainval_df[feature_cols]

        model = make_model(cat_cols, num_cols, C=best_C)
        model.fit(X_trainval, y_trainval)

        test_scores = model.predict_proba(X_test)[:, 1]
        test_ap = float(average_precision_score(y_test, test_scores)) if y_test.sum() > 0 else 0.0
        test_auc = safe_roc_auc(y_test, test_scores)
        rep = topk_report(y_test, test_scores, k=TOP_K)

        lift_p_str = fmt(rep["lift_precision_at_k"], nd=3)
        lift_r_str = fmt(rep["lift_recall_at_k"], nd=3)

        print("\nTest metrics:")
        print(f"  AP={fmt(test_ap)} | ROC_AUC={fmt(test_auc)}")
        print(
            f"  Top-{rep['k']}: hits={rep['hits']}/{rep['pos']} | "
            f"P@{rep['k']}={fmt(rep['precision_at_k'])} (lift={lift_p_str}) | "
            f"R@{rep['k']}={fmt(rep['recall_at_k'])} (lift={lift_r_str})"
        )
        print(
            f"  Random expected hits in Top-{rep['k']}: {rep['expected_hits_random']:.3f} "
            f"(baseline P@{rep['k']}={fmt(rep['baseline_precision_at_k'])})"
        )

        bank_out = OUT_ROOT / bank
        bank_out.mkdir(parents=True, exist_ok=True)

        metrics = {
            "bank": bank,
            "selected_C": best_C,
            "val_ap_best": best_val_ap,
            "test_ap": test_ap,
            "test_roc_auc": test_auc,
            "topk": rep,
        }

        out_json = bank_out / "metrics_test.json"
        out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved: {out_json}")

        summary_rows.append(
            {
                "bank": bank,
                "C": best_C,
                "val_ap": best_val_ap,
                "test_ap": test_ap,
                "roc_auc": test_auc if test_auc is not None else np.nan,
                f"hits@{rep['k']}": rep["hits"],
                f"P@{rep['k']}": rep["precision_at_k"],
                f"R@{rep['k']}": rep["recall_at_k"],
                f"liftP@{rep['k']}": rep["lift_precision_at_k"] if rep["lift_precision_at_k"] is not None else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("bank")
    print("\n" + "=" * 68)
    print("Summary")
    print("=" * 68)
    print(summary_df.to_string(index=False))

    out_csv = OUT_ROOT / "summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv}")


if __name__ == "__main__":
    main()
