from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y_true, scores)) if y_true.sum() > 0 else 0.0


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


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

