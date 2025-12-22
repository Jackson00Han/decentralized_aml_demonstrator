from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y_true, scores)) if y_true.sum() > 0 else 0.0


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def topk(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict:
    n = len(y_true)
    k = int(min(k, n))
    order = np.argsort(-scores)
    hits = int(y_true[order[:k]].sum())
    pos = int(y_true.sum())
    p = hits / k if k > 0 else 0.0
    r = hits / pos if pos > 0 else 0.0
    return {"k": k, "hits": hits, "p_at_k": p, "r_at_k": r, "pos": pos, "n": n}
