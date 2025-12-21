# src/fl/transforms.py
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class GlobalStandardizer(BaseEstimator, TransformerMixin):
    """
    Standardize numeric features using *global* mean/std (provided by server).
    - No fitting on client
    - Handles NaN by imputing with global mean
    """

    def __init__(self, mean: np.ndarray, scale: np.ndarray, eps: float = 1e-12):
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.eps = eps

    def fit(self, X, y=None):
        # No-op: parameters are provided
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        # Replace NaN with mean
        if np.isnan(X).any():
            X = X.copy()
            nan_mask = np.isnan(X)
            X[nan_mask] = np.take(self.mean, np.where(nan_mask)[1])

        scale = np.where(self.scale > self.eps, self.scale, 1.0)
        return (X - self.mean) / scale
