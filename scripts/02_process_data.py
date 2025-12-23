import pandas as pd
import os

def main():
    import sys
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(REPO_ROOT))
    from src.config import load_config
    cfg = load_config()

    bank_lists = cfg.banks.names
    for bank in bank_lists:
        # create directory
        out_dir = cfg.paths.data_processed / bank; os.makedirs(out_dir, exist_ok=True)
        df_trans = pd.read_csv(cfg.paths.data_raw / bank / "transactions.csv")
        df_accs = pd.read_csv(cfg.paths.data_raw / bank / "accounts.csv")
        df_alerts = pd.read_csv(cfg.paths.data_raw / bank / "alert_transactions.csv")

        df = df_trans.copy()
        df['tran_timestamp'] = pd.to_datetime(df['tran_timestamp'], utc=True, errors='coerce')

        df['y'] = df['tran_id'].isin(df_alerts['tran_id']).astype('int8') # binary label

        # select only relevant columns
        keep_cols = cfg.preprocess.keep_cols
        df_accs = df_accs[keep_cols].copy()
        
        df_org = df_accs.rename(columns={c: f"orig_{c}" for c in keep_cols})
        df_bene = df_accs.rename(columns={c: f"bene_{c}" for c in keep_cols})

        df = df.merge(df_org, how='left', left_on='orig_acct', right_on='orig_acct_id')
        df = df.merge(df_bene, how='left', left_on='bene_acct', right_on='bene_acct_id')

        df = df.drop(columns=['is_sar', 'alert_id'], errors='ignore')
        df = df.drop(columns=['orig_acct_id', 'bene_acct_id'], errors='ignore')
        df = df.sort_values(by='tran_timestamp').reset_index(drop=True)

        assert len(df) == len(df_trans), f"{bank}: Length mismatch after merge!"
        assert df["y"].sum() == len(df_alerts), f"{bank}: label count mismatch!"
        assert df["tran_timestamp"].notna().all(), f"{bank}: some tran_timestamp are NaT!"

        df.to_parquet(os.path.join(out_dir, f"{bank}_merged.parquet"), index=False)

        pos_rate = df['y'].mean()
        print(f"{bank}: Processed {len(df)} transactions with positive rate {pos_rate:.4f}")

if __name__ == "__main__":
    main()