# src/data_splits.py
from __future__ import annotations

import pandas as pd


def split_fixed_windows(df: pd.DataFrame, ts_col: str = "tran_timestamp"):
    """
    Fixed time split (UTC):
      - Train: 2017-01-01 .. 2018-04-30 23:59:59  (ts < 2018-05-01)
      - Val:   2018-05-01 .. 2018-08-31 23:59:59  (2018-05-01 <= ts < 2018-09-01)
      - Test:  2018-09-01 .. 2018-12-31 23:59:59  (2018-09-01 <= ts < 2019-01-01)

    Returns: (train_df, val_df, test_df, df_2018_used)
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df[df[ts_col].notna()].copy()

    start_data = pd.Timestamp("2017-01-01", tz="UTC")
    start_may = pd.Timestamp("2018-05-01", tz="UTC")
    start_sep = pd.Timestamp("2018-09-01", tz="UTC")
    start_2019 = pd.Timestamp("2019-01-01", tz="UTC")

    df_use = df[(df[ts_col] >= start_data) & (df[ts_col] < start_2019)].copy()

    tr = df_use[df_use[ts_col] < start_may].copy()
    va = df_use[(df_use[ts_col] >= start_may) & (df_use[ts_col] < start_sep)].copy()
    te = df_use[df_use[ts_col] >= start_sep].copy()

    # sanity: no overlap
    assert len(tr) + len(va) + len(te) == len(df_use)

    return tr, va, te, df_use
