# src/data.py
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

def split_stratified(
    df: pd.DataFrame,
    label_col: str = "is_sar",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    """
    Random stratified split for account-level classification.
    Returns: (train_df, val_df, test_df, df_used)
    """
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")
    if train_frac <= 0 or val_frac <= 0 or test_frac <= 0:
        raise ValueError("train/val/test fractions must be positive")

    df = df.copy()
    df = df[df[label_col].notna()].copy()
    if df.empty:
        raise ValueError("No rows available after dropping missing labels")

    y = df[label_col].astype(int)
    test_size = val_frac + test_frac

    class_counts = y.value_counts()
    can_stratify = len(class_counts) >= 2 and class_counts.min() >= 2

    if can_stratify:
        tr, temp = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=y,
        )
    else:
        tr, temp = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )

    temp_y = temp[label_col].astype(int)
    temp_counts = temp_y.value_counts()
    temp_can_stratify = len(temp_counts) >= 2 and temp_counts.min() >= 2

    if temp_can_stratify:
        va, te = train_test_split(
            temp,
            test_size=test_frac / test_size,
            random_state=seed,
            shuffle=True,
            stratify=temp_y,
        )
    else:
        va, te = train_test_split(
            temp,
            test_size=test_frac / test_size,
            random_state=seed,
            shuffle=True,
        )

    return tr, va, te
