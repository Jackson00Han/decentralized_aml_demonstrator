# scripts/02_train_local_baseline.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]

BANKS = ["bank_small", "bank_medium", "bank_large"]

CAT_COLS = ["orig_state", "bene_state"]
NUM_COLS = ["base_amt", "orig_initial_deposit", "bene_initial_deposit"]
FEATURE_COLS = NUM_COLS + CAT_COLS
DROP_COLS = ["tran_id", "orig_acct", "bene_acct"]

SEED = 42
K = 1000
C_GRID = [0.1, 1.0]  # MVP grid; you can add 0.01 if you want

TRAIN_VAL_CUTOFF = pd.Timestamp("2017-09-01", tz="UTC")
VAL_TEST_CUTOFF = pd.Timestamp("2017-11-01", tz="UTC")
END_CUTOFF = pd.Timestamp("2018-01-01", tz="UTC")


def precision_recall_at_k(y_true: pd.Series, scores: np.ndarray, k: int):
    k = min(k, len(y_true))
    idx = np.argsort(-scores)[:k]
    hits = int(y_true.iloc[idx].sum())
    precision_at_k = hits / k
    recall_at_k = hits / max(1, int(y_true.sum()))
    return hits, precision_at_k, recall_at_k


def expected_random_baseline(y_true: pd.Series, k: int):
    n = len(y_true)
    base_rate = float(y_true.mean()) if n > 0 else 0.0
    random_precision = base_rate
    random_recall = k / n if n > 0 else 0.0
    return random_precision, random_recall


def build_preprocess():
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), NUM_COLS),
            ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
        ],
        remainder="drop",
    )


def safe_ap(y_true: pd.Series, scores: np.ndarray):
    # AP is meaningful only if there is at least 1 positive
    if int(y_true.sum()) == 0:
        return None
    return float(average_precision_score(y_true, scores))


def safe_roc_auc(y_true: pd.Series, scores: np.ndarray):
    # ROC-AUC requires both classes
    if len(np.unique(y_true.to_numpy())) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def main():
    out_root = REPO_ROOT / "outputs" / "local_baseline"
    out_root.mkdir(parents=True, exist_ok=True)

    for bank in BANKS:
        df = pd.read_parquet(REPO_ROOT / f"data/processed/{bank}/{bank}_merged.parquet")
        df = df.drop(columns=DROP_COLS, errors="ignore")
        df = df[df["tran_timestamp"] < END_CUTOFF].reset_index(drop=True)

        ts = df["tran_timestamp"]
        train_mask = ts < TRAIN_VAL_CUTOFF
        val_mask = (ts >= TRAIN_VAL_CUTOFF) & (ts < VAL_TEST_CUTOFF)
        test_mask = (ts >= VAL_TEST_CUTOFF) & (ts < END_CUTOFF)

        df_train = df[train_mask].reset_index(drop=True)
        df_val = df[val_mask].reset_index(drop=True)
        df_test = df[test_mask].reset_index(drop=True)

        print(f"\n=== Local baseline (tune on val) | {bank} ===")
        print(f"train pos={int(df_train['y'].sum())} val pos={int(df_val['y'].sum())} test pos={int(df_test['y'].sum())}")

        X_train = df_train[FEATURE_COLS]
        y_train = df_train["y"].astype(int)

        X_val = df_val[FEATURE_COLS]
        y_val = df_val["y"].astype(int)

        # -----------------------
        # 1) Tune C on validation
        # -----------------------
        candidates = []

        for C in C_GRID:
            preprocess = build_preprocess()
            X_train_p = preprocess.fit_transform(X_train)
            X_val_p = preprocess.transform(X_val)

            model = LogisticRegression(
                max_iter=1000,
                C=C,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            )
            model.fit(X_train_p, y_train)

            val_scores = model.predict_proba(X_val_p)[:, 1]
            val_ap = safe_ap(y_val, val_scores)

            # Use AP if possible; otherwise fallback to base rate (no signal)
            # (You could also fallback to log loss, but keep MVP simple.)
            score_for_selection = val_ap if val_ap is not None else -1.0

            candidates.append(
                {
                    "C": C,
                    "val_ap": val_ap,
                    "score_for_selection": score_for_selection,
                }
            )

            print(f"  C={C} -> val_ap={val_ap}")

        best = max(candidates, key=lambda d: d["score_for_selection"])
        best_C = best["C"]
        print(f"Selected C={best_C} (by val AP; if no positives in val, this may be arbitrary)")

        # -----------------------
        # 2) Retrain on train+val
        # -----------------------
        df_trainval = pd.concat([df_train, df_val], axis=0, ignore_index=True)
        X_trainval = df_trainval[FEATURE_COLS]
        y_trainval = df_trainval["y"].astype(int)

        preprocess = build_preprocess()
        X_trainval_p = preprocess.fit_transform(X_trainval)

        model = LogisticRegression(
            max_iter=1000,
            C=best_C,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(X_trainval_p, y_trainval)

        # -----------------------
        # 3) Final test evaluation
        # -----------------------
        X_test = df_test[FEATURE_COLS]
        y_test = df_test["y"].astype(int)

        X_test_p = preprocess.transform(X_test)
        test_scores = model.predict_proba(X_test_p)[:, 1]

        test_ap = safe_ap(y_test, test_scores)
        test_roc_auc = safe_roc_auc(y_test, test_scores)

        hits, p_at_k, r_at_k = precision_recall_at_k(y_test, test_scores, K)
        base_p, base_r = expected_random_baseline(y_test, K)

        report = {
            "bank": bank,
            "selected_C": best_C,
            "candidates": candidates,
            "n_train": int(len(df_train)),
            "n_val": int(len(df_val)),
            "n_test": int(len(df_test)),
            "train_pos": int(df_train["y"].sum()),
            "val_pos": int(df_val["y"].sum()),
            "test_pos": int(y_test.sum()),
            "k": K,
            "test_ap": test_ap,
            "test_roc_auc": test_roc_auc,
            "test_hits_topk": hits,
            "test_precision_at_k": float(p_at_k),
            "test_recall_at_k": float(r_at_k),
            "random_expected_precision_at_k": float(base_p),
            "random_expected_recall_at_k": float(base_r),
            "lift_precision_at_k": float(p_at_k / max(1e-12, base_p)) if base_p > 0 else None,
            "lift_recall_at_k": float(r_at_k / max(1e-12, base_r)) if base_r > 0 else None,
        }

        print(f"TEST AP={test_ap} ROC_AUC={test_roc_auc}")
        print(
            f"TEST Top-{K}: hits={hits}/{int(y_test.sum())} "
            f"P@{K}={p_at_k:.6f} (lift={report['lift_precision_at_k']}) "
            f"R@{K}={r_at_k:.6f} (lift={report['lift_recall_at_k']})"
        )

        expected_hits = K * base_p
        print("expected hits in random topK:", expected_hits)
        print(f"baseline precision@{K}={base_p} recall@{K}={base_r}")

        bank_out = out_root / bank
        bank_out.mkdir(parents=True, exist_ok=True)
        with open(bank_out / "metrics_test.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"Saved {bank_out / 'metrics_test.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
