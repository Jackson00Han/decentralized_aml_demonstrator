from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y_true, scores)) if y_true.sum() > 0 else 0.0


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def precision_recall_f1_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float):
    """Compute P/R/F1 for a given threshold."""
    y_hat = (scores >= thr).astype(int)

    tp = int(np.sum((y_hat == 1) & (y_true == 1)))
    fp = int(np.sum((y_hat == 1) & (y_true == 0)))
    fn = int(np.sum((y_hat == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray):
    """
    Pick threshold that maximizes F1 on the given set.
    Tie-break: higher precision, then higher recall.
    """
    # Important: if scores are not probabilities, do NOT add "1.1" / "0" sentinels.
    thresholds = np.unique(scores)
    thresholds.sort()
    thresholds = thresholds[::-1]  # high -> low

    best_thr = float(thresholds[0])
    best_p, best_r, best_f1 = 0.0, 0.0, -1.0

    for thr in thresholds:
        p, r, f1 = precision_recall_f1_at_threshold(y_true, scores, float(thr))
        if (f1 > best_f1 + 1e-12) or (abs(f1 - best_f1) <= 1e-12 and (p > best_p + 1e-12)) or \
           (abs(f1 - best_f1) <= 1e-12 and abs(p - best_p) <= 1e-12 and (r > best_r + 1e-12)):
            best_thr, best_p, best_r, best_f1 = float(thr), float(p), float(r), float(f1)

    return best_thr, best_p, best_r, best_f1


import numpy as np


def weighted_logloss_sums(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    w_pos: float,
    w_neg: float,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """
    Weighted binary cross-entropy (logloss) aggregation-friendly form.

    Returns:
      - sum_weighted_loss = Σ w_i * loss_i
      - sum_weights       = Σ w_i

    So global weighted mean logloss can be computed by secure-agg:
      global = Σ(sum_weighted_loss) / Σ(sum_weights)
    """
    y = y_true.astype(float)
    p = np.clip(p_pred.astype(float), eps, 1.0 - eps)

    # per-example unweighted logloss
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    # per-example weights
    w = np.where(y > 0.5, float(w_pos), float(w_neg)).astype(float)

    return float((w * loss).sum()), float(w.sum())


def class_balance_weights(n: int, pos: int) -> tuple[float, float]:
    """
    Balanced weights such that total weight of positives == total weight of negatives.

    w_pos = n / (2*pos)
    w_neg = n / (2*(n-pos))

    If pos==0 or pos==n, fall back to 1.0/1.0.
    """
    n = int(n)
    pos = int(pos)
    neg = n - pos
    if n <= 0 or pos <= 0 or neg <= 0:
        return 1.0, 1.0
    return n / (2.0 * pos), n / (2.0 * neg)
