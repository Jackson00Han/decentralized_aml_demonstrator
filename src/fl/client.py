# src/fl/client.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from .adapters import ModelAdapter
from .core import ClientUpdate, Params
from .metrics import ranking_metrics, topk_metrics
from .stats import compute_local_numeric_stats, NumericStats


@dataclass
class SplitConfig:
    train_cutoff: pd.Timestamp
    val_cutoff: pd.Timestamp
    end_cutoff: pd.Timestamp


class BankClient:
    def __init__(
        self,
        client_id: str,
        parquet_path: str,
        preprocess: ColumnTransformer,
        model_adapter: ModelAdapter,
        split_cfg: SplitConfig,
        feature_cols: List[str],
        num_cols: List[str],
        label_col: str = "y",
    ):
        self.client_id = client_id
        self.parquet_path = parquet_path
        self.preprocess = preprocess
        self.model_adapter = model_adapter
        self.split_cfg = split_cfg
        self.feature_cols = feature_cols
        self.num_cols = num_cols
        self.label_col = label_col

        self.df_train: pd.DataFrame | None = None
        self.df_val: pd.DataFrame | None = None
        self.df_test: pd.DataFrame | None = None

    def load_and_split(self) -> None:
        df = pd.read_parquet(self.parquet_path)
        df = df.drop(columns=["tran_id", "orig_acct", "bene_acct"], errors="ignore")
        df = df[df["tran_timestamp"] < self.split_cfg.end_cutoff].reset_index(drop=True)

        ts = df["tran_timestamp"]
        train_mask = ts < self.split_cfg.train_cutoff
        val_mask = (ts >= self.split_cfg.train_cutoff) & (ts < self.split_cfg.val_cutoff)
        test_mask = (ts >= self.split_cfg.val_cutoff) & (ts < self.split_cfg.end_cutoff)

        self.df_train = df[train_mask].reset_index(drop=True)
        self.df_val = df[val_mask].reset_index(drop=True)
        self.df_test = df[test_mask].reset_index(drop=True)

    def pos_counts(self) -> Dict[str, int]:
        assert self.df_train is not None and self.df_val is not None and self.df_test is not None
        return {
            "train_pos": int(self.df_train[self.label_col].sum()),
            "val_pos": int(self.df_val[self.label_col].sum()),
            "test_pos": int(self.df_test[self.label_col].sum()),
        }

    def local_numeric_stats(self) -> NumericStats:
        assert self.df_train is not None
        return compute_local_numeric_stats(self.df_train, self.num_cols)

    def _xy(self, split: str):
        df = {"train": self.df_train, "val": self.df_val, "test": self.df_test}[split]
        assert df is not None
        X = self.preprocess.transform(df[self.feature_cols])
        y = df[self.label_col].astype(int).to_numpy()
        return X, y

    def initialize_model(self) -> None:
        X, y = self._xy("train")
        self.model_adapter.initialize(X, y)

    def fit_round(self, global_params: Params, local_epochs: int = 1) -> ClientUpdate:
        X, y = self._xy("train")
        self.model_adapter.set_params(global_params)
        self.model_adapter.fit_local(X, y, local_epochs=local_epochs)
        new_params = self.model_adapter.get_params()
        return ClientUpdate(client_id=self.client_id, n_train=X.shape[0], params=new_params)

    def evaluate(self, global_params: Params, split: str, k_review: int) -> Dict:
        X, y_np = self._xy(split)
        self.model_adapter.set_params(global_params)
        scores = self.model_adapter.predict_proba(X)
        y = pd.Series(y_np)

        out = {"client_id": self.client_id, "split": split}
        out.update(ranking_metrics(y, scores))
        out.update(topk_metrics(y, scores, k=k_review))
        return out
