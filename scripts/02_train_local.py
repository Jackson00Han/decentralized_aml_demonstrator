import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

def main():
    seed =42

    # Load the dataset
    df = pd.read_parquet('data/processed/bank_small/bank_small_merged.parquet')

    df = df.drop(columns=['tran_id', 'orig_acct', 'bene_acct'])

    df = df[df['tran_timestamp'] <= pd.Timestamp("2018-01-01", tz="UTC")].reset_index(drop=True)

    df = df.drop(columns=['orig_acct', 'bene_acct'], errors='ignore')

    trainval_test_cutoff = pd.Timestamp("2017-11-01", tz="UTC")
    train_val_cutoff = pd.Timestamp("2017-09-01", tz="UTC")
    print(f"Train/Val/Test cutoffs: {train_val_cutoff} / {trainval_test_cutoff}")

    test_mask = df['tran_timestamp'] >= trainval_test_cutoff
    val_mask = (df['tran_timestamp'] >= train_val_cutoff) & (df['tran_timestamp'] < trainval_test_cutoff)
    train_mask = df['tran_timestamp'] < train_val_cutoff

    df_val = df[val_mask].reset_index(drop=True)
    df_train = df[train_mask].reset_index(drop=True)

    cat_cols = ["orig_state", "bene_state"]
    num_cols = ["base_amt", "orig_initial_deposit", "bene_initial_deposit"]

    numeric_pipe = Pipeline([
        ("scaler", StandardScaler()),
        
    ])

    categorical_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )


    X_train = df_train[num_cols + cat_cols]
    y_train = df_train['y']

    X_val = df_val[num_cols + cat_cols]
    y_val = df_val['y']

    print("train pos:", int(y_train.sum()), "val pos:", int(y_val.sum()))


    X_train_processed = preprocess.fit_transform(X_train)
    X_val_processed = preprocess.transform(X_val)

    print(f"Training set size: {X_train_processed.shape}, Validation set size: {X_val_processed.shape}")

    # load logistic regression model
    model = LogisticRegression(
        max_iter=1000, 
        C=0.1,
        class_weight='balanced', 
        random_state=seed,
        n_jobs=-1
        )

    model.fit(X_train_processed, y_train)

    # Evaluate on validation set
    y_val_proba = model.predict_proba(X_val_processed)[:, 1]

    K = 200  # adjust to your "review capacity", e.g. select 200 top suspicious transactions and have them reviewed by analysts
    idx = np.argsort(-y_val_proba)[:K]
    precision_at_k = y_val.iloc[idx].mean()
    recall_at_k = y_val.iloc[idx].sum() / max(1, y_val.sum())

    # add baseline precison@k and recall@k

    # ---- Random baseline (without replacement) ----
    N = len(y_val)
    P = int(y_val.sum())
    base_rate = y_val.mean()

    baseline_precision_at_k = base_rate                 # E[Precision@K]
    baseline_recall_at_k = K / N                        # E[Recall@K]

    # model "hits" (for clarity)
    hits = int(y_val.iloc[idx].sum())

    print(f"\nRandom baseline (expected):")
    print(f"  Base rate: {base_rate:.6f}  (P={P}, N={N})")
    print(f"  E[Precision@{K}] = {baseline_precision_at_k:.6f}")
    print(f"  E[Recall@{K}]    = {baseline_recall_at_k:.6f}")

    print(f"\nModel Top-{K}:")
    print(f"  hits = {hits}/{P} positives in Top-{K}")
    print(f"  Precision@{K} = {precision_at_k:.6f}  (lift={precision_at_k / max(1e-12, baseline_precision_at_k):.2f}x)")
    print(f"  Recall@{K}    = {recall_at_k:.6f}     (lift={recall_at_k / max(1e-12, baseline_recall_at_k):.2f}x)")

    roc_auc = roc_auc_score(y_val, y_val_proba)
    ap = average_precision_score(y_val, y_val_proba)

    print(f"Validation ROC AUC: {roc_auc:.4f}")
    print(f"Validation Average Precision: {ap:.4f}")
    print()

if __name__ == "__main__":
    main()

