# src/fl/metrics.py
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def ranking_metrics(y: pd.Series, scores: np.ndarray) -> Dict:
    ap = float(average_precision_score(y, scores))
    roc = float(roc_auc_score(y, scores)) if y.nunique() > 1 else float("nan")
    return {"ap": ap, "roc_auc": roc}


def topk_metrics(y: pd.Series, scores: np.ndarray, k: int) -> Dict:
    idx = np.argsort(-scores)[:k]
    hits = int(y.iloc[idx].sum())
    P = int(y.sum())
    N = int(len(y))

    precision_at_k = hits / k
    recall_at_k = hits / max(1, P)

    base_rate = float(y.mean())
    baseline_precision = base_rate     # E[Precision@K] if random sampling
    baseline_recall = k / N            # E[Recall@K] if random sampling

    return {
        "N": N,
        "P": P,
        "hits": hits,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "baseline_precision_at_k": baseline_precision,
        "baseline_recall_at_k": baseline_recall,
        "lift_precision": precision_at_k / max(1e-12, baseline_precision),
        "lift_recall": recall_at_k / max(1e-12, baseline_recall),
    }
