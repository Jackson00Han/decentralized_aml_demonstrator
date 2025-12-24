from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



from src.fl_protocol import GlobalPlan


class FixedStandardScaler(BaseEstimator, TransformerMixin):
    """
    Standardize with pre-computed mean/std (from global_plan).
    No fitting on client data -> consistent across banks.
    """
    def __init__(self, means: np.ndarray, stds: np.ndarray):
        self.means = np.asarray(means, dtype=float)
        self.stds = np.asarray(stds, dtype=float)
        self.feature_names_in_ = None  # will be set by builder

    def fit(self, X, y=None):
        # no-op, but mark as fitted
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        stds = np.where(self.stds == 0, 1.0, self.stds)
        return (X - self.means) / stds

    def get_feature_names_out(self, input_features=None):
        # sklearn calls this from ColumnTransformer -> Pipeline
        if input_features is None:
            if self.feature_names_in_ is None:
                # fallback
                return np.array([f"x{i}" for i in range(len(self.means))], dtype=object)
            return np.array(self.feature_names_in_, dtype=object)
        return np.array(input_features, dtype=object)

def build_local_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    cat_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num_pipe = Pipeline([("scaler", StandardScaler())])
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
    )


def build_preprocessor(plan: GlobalPlan) -> ColumnTransformer:
    cat_cols = plan.feature_schema.cat_cols
    num_cols = plan.feature_schema.num_cols

    # fixed categories for OHE => same feature space across banks
    categories = [plan.global_categories[c] for c in cat_cols]
    ohe = OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=True)

    means = np.array([plan.global_numeric[c]["mean"] for c in num_cols], dtype=float)
    stds = np.array([plan.global_numeric[c]["std"] for c in num_cols], dtype=float)
    scaler = FixedStandardScaler(means=means, stds=stds)
    scaler.feature_names_in_ = np.array(num_cols, dtype=object)

    numeric_pipe = Pipeline([("scaler", scaler)])
    categorical_pipe = Pipeline([("onehot", ohe)])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,  # always return sparse matrix
    )

    # IMPORTANT: sklearn transformers still need .fit() to finalize internal state.
    # Fit on a tiny dummy frame using plan vocab.
    dummy = pd.DataFrame(
        {
            **{cat_cols[i]: [categories[i][0] if len(categories[i]) else "NA"] for i in range(len(cat_cols))},
            **{c: [0.0] for c in num_cols},
        }
    )
    preprocess.fit(dummy)
    return preprocess
