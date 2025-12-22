import sys
from pathlib import Path

# Allow "python scripts/..." to import from repo root
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import sparse

from src.fl_adapters import SkLogRegSGD


def make_toy_data(n=200, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float64)

    # Create labels from a ground-truth logistic model
    w_true = rng.normal(size=(d,)).astype(np.float64)
    b_true = -0.2
    logits = X @ w_true + b_true
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < probs).astype(np.int32)
    return X, y


def main():
    d = 10
    X, y = make_toy_data(n=300, d=d, seed=42)

    # Optional: test sparse input too
    X_sparse = sparse.csr_matrix(X)

    # --- init model with zeros (like server global model) ---
    model = SkLogRegSGD(d=d, eta0=0.05, alpha=1e-6, seed=123)

    global_params = {
        "coef": np.zeros((1, d), dtype=np.float64),
        "intercept": np.zeros((1,), dtype=np.float64),
    }
    model.set_params(global_params)

    p0 = model.get_params()
    print("== Initial params ==")
    print("coef L2:", float(np.linalg.norm(p0["coef"])))
    print("intercept:", float(p0["intercept"][0]))
    print()

    # --- one local round (dense) ---
    model.train_one_round(X, y, local_epochs=3, seed=1)
    p1 = model.get_params()

    print("== After 1 local round (dense) ==")
    print("coef L2:", float(np.linalg.norm(p1["coef"])))
    print("intercept:", float(p1["intercept"][0]))
    print("coef changed:", not np.allclose(p0["coef"], p1["coef"]))
    print("intercept changed:", not np.allclose(p0["intercept"], p1["intercept"]))
    print()

    # --- predictions sanity check ---
    scores = model.predict_scores(X[:5])
    print("== Scores (first 5) ==")
    print(scores)
    print("in (0,1):", bool(np.all((scores > 0) & (scores < 1))))
    print()

    # --- train another round using sparse input (should also work) ---
    model2 = SkLogRegSGD(d=d, eta0=0.05, alpha=1e-6, seed=123)
    model2.set_params(global_params)
    model2.train_one_round(X_sparse, y, local_epochs=3, seed=1)
    p2 = model2.get_params()

    print("== After 1 local round (sparse) ==")
    print("coef L2:", float(np.linalg.norm(p2["coef"])))
    print("intercept:", float(p2["intercept"][0]))
    print("works on sparse:", True)
    print()

    # quick comparison (they won't be identical, but should be reasonable)
    diff = float(np.linalg.norm(p1["coef"] - p2["coef"]))
    print("Dense vs sparse coef diff L2:", diff)


if __name__ == "__main__":
    main()
