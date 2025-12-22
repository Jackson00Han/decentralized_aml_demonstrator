import sys
from pathlib import Path

# make repo root importable when running "python scripts/..."
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.fl_protocol import FeatureSchema, GlobalPlan
from src.fl_preprocess import build_preprocessor


def main():
    out = Path("outputs/_debug_preprocess")
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1) Build a tiny GlobalPlan (toy vocab + fixed mean/std) ----
    plan = GlobalPlan(
        schema_version="v1",
        feature_schema=FeatureSchema(
            cat_cols=["orig_state", "bene_state"],
            num_cols=["base_amt", "orig_initial_deposit", "bene_initial_deposit"],
        ),
        global_categories={
            # pretend we only allow these states in vocab
            "orig_state": ["CA", "NY"],
            "bene_state": ["CA", "TX"],
        },
        global_numeric={
            # fixed scaling
            "base_amt": {"mean": 10.0, "std": 2.0},
            "orig_initial_deposit": {"mean": 100.0, "std": 50.0},
            "bene_initial_deposit": {"mean": 120.0, "std": 60.0},
        },
    )

    preprocess = build_preprocessor(plan)

    # ---- 2) Create toy data (includes unknown states) ----
    df = pd.DataFrame(
        {
            "orig_state": ["CA", "NY", "TX"],   # TX is unknown for orig_state vocab
            "bene_state": ["TX", "CA", "WA"],   # WA is unknown for bene_state vocab
            "base_amt": [10.0, 14.0, 6.0],
            "orig_initial_deposit": [100.0, 150.0, 50.0],
            "bene_initial_deposit": [120.0, 180.0, 60.0],
        }
    )

    # note: build_preprocessor expects columns in plan order (num + cat in my earlier scripts)
    X = preprocess.transform(df[plan.feature_schema.num_cols + plan.feature_schema.cat_cols])

    print("== Input df ==")
    print(df)
    print()

    print("== Transformed X ==")
    print("type:", type(X))
    print("shape:", X.shape)  # (n_rows, n_features)
    print()

    # ---- 3) Inspect feature names ----
    names = preprocess.get_feature_names_out()
    print("== Feature names (first 20) ==")
    for i, n in enumerate(names[:20]):
        print(i, n)
    print("total:", len(names))
    print()

    # ---- 4) Check numeric scaling for row 0 explicitly ----
    # We know the first 3 columns are numeric (num_cols)
    X_dense0 = X[0].toarray().ravel() if hasattr(X, "toarray") else np.asarray(X[0]).ravel()
    num_part = X_dense0[: len(plan.feature_schema.num_cols)]

    # manual scaling
    manual = np.array(
        [
            (10.0 - 10.0) / 2.0,     # base_amt
            (100.0 - 100.0) / 50.0,  # orig_initial_deposit
            (120.0 - 120.0) / 60.0,  # bene_initial_deposit
        ],
        dtype=float,
    )

    print("== Numeric scaling check (row 0) ==")
    print("from transformer:", num_part)
    print("manual expected:", manual)
    print("allclose:", np.allclose(num_part, manual))
    print()

    # ---- 5) Show non-zero indices per row (good for seeing OHE behavior) ----
    print("== Non-zero feature indices per row ==")

    # If sparse, use CSR; if dense, use numpy nonzero
    if hasattr(X, "tocsr"):
        X_csr = X.tocsr()
        for i in range(X_csr.shape[0]):
            idx = X_csr[i].indices
            vals = X_csr[i].data
            pairs = list(zip(idx.tolist(), vals.tolist()))
            print(f"row {i}: nnz={len(idx)} -> {pairs[:10]}")
    else:
        X_np = np.asarray(X)
        for i in range(X_np.shape[0]):
            idx = np.flatnonzero(X_np[i])
            vals = X_np[i, idx]
            pairs = list(zip(idx.tolist(), vals.tolist()))
            print(f"row {i}: nnz={len(idx)} -> {pairs[:10]}")
    print()


    print("Note:")
    print("- unknown orig_state='TX' (row 2) produces no orig_state one-hot column (all zeros for that group).")
    print("- unknown bene_state='WA' (row 2) produces no bene_state one-hot column (all zeros for that group).")


if __name__ == "__main__":
    main()
