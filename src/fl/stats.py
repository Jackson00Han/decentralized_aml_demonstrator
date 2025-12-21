# src/fl/stats.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd


@dataclass
class NumericStats:
    n: int
    sum: np.ndarray
    sumsq: np.ndarray


def compute_local_numeric_stats(df_train: pd.DataFrame, num_cols: List[str]) -> NumericStats:
    """
    Client-side: compute n / sum / sumsq on TRAIN split only.
    No raw data leaves the client in a real deployment.
    """
    X = df_train[num_cols].to_numpy(dtype=float)
    n = X.shape[0]

    # Ignore NaN in sums by treating NaN as 0 but also adjust n? For simplicity:
    # - we impute NaN to 0 here, and GlobalStandardizer will impute to mean later.
    # If NaNs exist, a more careful approach is to track per-feature counts.
    X0 = np.nan_to_num(X, nan=0.0)

    s = X0.sum(axis=0)
    ss = (X0 ** 2).sum(axis=0)
    return NumericStats(n=n, sum=s, sumsq=ss)


def aggregate_global_mean_std(stats_list: List[NumericStats], eps: float = 1e-12) -> Dict[str, np.ndarray]:
    """
    Server-side: aggregate sums to compute global mean and std.
    """
    total_n = sum(s.n for s in stats_list)
    if total_n <= 0:
        raise ValueError("total_n is zero")

    total_sum = np.zeros_like(stats_list[0].sum, dtype=float)
    total_sumsq = np.zeros_like(stats_list[0].sumsq, dtype=float)

    for s in stats_list:
        total_sum += s.sum
        total_sumsq += s.sumsq

    mean = total_sum / total_n
    var = (total_sumsq / total_n) - (mean ** 2)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    # avoid division by zero
    std = np.where(std > eps, std, 1.0)

    return {"mean": mean, "std": std, "var": var}
