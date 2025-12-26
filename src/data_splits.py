# src/data_splits.py
from __future__ import annotations

import pandas as pd


def split_fixed_windows(df: pd.DataFrame, ts_col: str = "tran_timestamp"):
    """
    Fixed time split (UTC):
    - Train: ts < 2018-05-01
    - Val:   2018-05-01 <= ts < 2018-09-01
    - Test:  2018-09-01 <= ts < 2019-01-01
    Returns: (train_df, val_df, test_df, df_used)
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df[df[ts_col].notna()].copy()

    start_date = pd.Timestamp("2017-01-01", tz="UTC")
    start_may = pd.Timestamp("2018-05-01", tz="UTC")
    start_sep = pd.Timestamp("2018-09-01", tz="UTC")
    start_2019 = pd.Timestamp("2019-01-01", tz="UTC")

    df_used = df[(df[ts_col] >= start_date) & (df[ts_col] < start_2019)].copy()

    tr = df_used[df_used[ts_col] < start_may].copy()
    va = df_used[(df_used[ts_col] >= start_may) & (df_used[ts_col] < start_sep)].copy()
    te = df_used[df_used[ts_col] >= start_sep].copy()

    if len(tr) + len(va) + len(te) != len(df_used):
        raise RuntimeError("Time split sanity check failed: overlap or drop detected.")

    return tr, va, te, df_used

import pandas as pd

def build_account_features(tx: pd.DataFrame, ts_col: str = "tran_timestamp") -> pd.DataFrame:
    """
    Minimal account-level aggregation for MVP.
    Required columns: orig_acct, bene_acct, base_amt, tran_timestamp
    Returns: DataFrame indexed by acct_id.
    """
    tx = tx.copy()


    out = (
        tx.groupby("orig_acct")
        .agg(
            out_cnt=("tran_id", "size"),
            out_amt_sum=("base_amt", "sum"),
            out_uniq_bene=("bene_acct", "nunique"),
        )
        .rename_axis("acct_id")
    )

    inn = (
        tx.groupby("bene_acct")
        .agg(
            in_cnt=("tran_id", "size"),
            in_amt_sum=("base_amt", "sum"),
            in_uniq_orig=("orig_acct", "nunique"),
        )
        .rename_axis("acct_id")
    )

    feats = out.join(inn, how="outer").fillna(0)

    feats["net_flow"] = feats["in_amt_sum"] - feats["out_amt_sum"]
    feats["turnover"] = feats["in_amt_sum"] + feats["out_amt_sum"]
    feats["uniq_total"] = feats["out_uniq_bene"] + feats["in_uniq_orig"]

    return feats.sort_index()

