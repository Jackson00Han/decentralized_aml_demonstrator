import pandas as pd
import os
import sys
import shutil
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))


def main():

    from src.config import load_config
    cfg = load_config()
    # process each bank's data
    bank_lists = cfg.banks.names
    for bank in bank_lists:
        # create directory
        out_dir = cfg.paths.data_processed / bank
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        df_accs = pd.read_csv(cfg.paths.data_raw / bank / "accounts.csv")
        df_trans = pd.read_csv(cfg.paths.data_raw / bank / "transactions.csv")
        df_sar = pd.read_csv(cfg.paths.data_raw / bank / "sar_accounts.csv")

        # clean data
        df_accs = df_accs.drop_duplicates().reset_index(drop=True)
        df_trans = df_trans.drop_duplicates().reset_index(drop=True)
        df_sar = df_sar.drop_duplicates().reset_index(drop=True)

        # process accounts data
        ac_used = df_accs[['acct_id', 'initial_deposit', 'state']].copy()
        ac_used['acct_id'] = pd.to_numeric(ac_used['acct_id'], errors='coerce').astype('Int32')
        ac_used['initial_deposit'] = pd.to_numeric(ac_used['initial_deposit'], errors='coerce').astype('float32')
        ac_used['state'] = ac_used['state'].astype('category')

        # process transactions
        df_trans = df_trans[['tran_id', 'tran_timestamp', 'orig_acct', 'bene_acct', 'base_amt']]
        df_trans['tran_timestamp'] = pd.to_datetime(df_trans['tran_timestamp'], utc=True, errors='coerce')
        tx_used = df_trans.sort_values('tran_timestamp').reset_index(drop=True)
        tx_used['orig_acct'] = pd.to_numeric(tx_used['orig_acct'], errors='coerce').astype('Int32')
        tx_used['bene_acct'] = pd.to_numeric(tx_used['bene_acct'], errors='coerce').astype('Int32')
        tx_used['base_amt'] = pd.to_numeric(tx_used['base_amt'], errors='coerce').astype('float32')

        # process SAR accounts
        sar_used = df_sar[['ACCOUNT_ID', 'IS_SAR']].copy()
        sar_used['ACCOUNT_ID'] = pd.to_numeric(sar_used['ACCOUNT_ID'], errors='coerce').astype('Int32')
        sar_used = sar_used[sar_used['IS_SAR'] == "YES"]
        sar_set = set(sar_used['ACCOUNT_ID'].unique())

        # add label to accounts
        ac_used['is_sar'] = ac_used['acct_id'].isin(sar_set)

        # aggregation
        ac_out = tx_used.groupby('orig_acct').agg(
            out_cnt = ('tran_id', 'size'),
            out_sum = ('base_amt', 'sum'),
            out_nuniq_bene = ('bene_acct', 'nunique'),
        ).rename_axis('acct_id').reset_index()

        ac_in = tx_used.groupby('bene_acct').agg(
            in_cnt = ('tran_id', 'size'),
            in_sum = ('base_amt', 'sum'),
            in_nuniq_orig = ('orig_acct', 'nunique') 
        ).rename_axis('acct_id').reset_index()

        # merge in and out features
        ac_features = ac_used.merge(ac_out, how='left', on='acct_id')
        ac_features = ac_features.merge(ac_in, how='left', on='acct_id')

        # fill NaN with 0 for transaction features
        tx_feature_cols = ['out_cnt', 'out_sum', 'out_nuniq_bene', 'in_cnt', 'in_sum', 'in_nuniq_orig']
        ac_features[tx_feature_cols] = ac_features[tx_feature_cols].fillna(0)

        # derived features
        ac_features['net_flow'] = ac_features['in_sum'] - ac_features['out_sum']

        # burst features, aggregate transactions over each day
        tx_used['date'] = tx_used['tran_timestamp'].dt.floor('D')
        out_daily = tx_used.groupby(['orig_acct', 'date']).agg(
            daily_out_sum = ('base_amt', 'sum'),
            daily_out_cnt = ('tran_id', 'size')
        ).reset_index()
        out_burst = out_daily.groupby('orig_acct').agg(
            out_max_daily_cnt = ('daily_out_cnt', 'max'),
            out_max_daily_sum = ('daily_out_sum', 'max')
        ).rename_axis('acct_id').reset_index()

        in_daily = tx_used.groupby(['bene_acct', 'date']).agg(
            daily_in_sum = ('base_amt', 'sum'),
            daily_in_cnt = ('tran_id', 'size')
        ).reset_index()
        in_burst = in_daily.groupby('bene_acct').agg(
            in_max_daily_cnt = ('daily_in_cnt', 'max'),
            in_max_daily_sum = ('daily_in_sum', 'max')
        ).rename_axis('acct_id').reset_index()

        ac_features = ac_features.merge(out_burst, how='left', on='acct_id')
        ac_features = ac_features.merge(in_burst, how='left', on='acct_id')
        ac_features[['out_max_daily_cnt', 'out_max_daily_sum', 'in_max_daily_cnt', 'in_max_daily_sum']] = ac_features[['out_max_daily_cnt', 'out_max_daily_sum', 'in_max_daily_cnt', 'in_max_daily_sum']].fillna(0)

        # sanity check
        assert ac_features['acct_id'].nunique() == ac_used['acct_id'].nunique(), "Account count mismatch after feature engineering"

        # save processed data as parquet format
        ac_features.to_parquet(out_dir / "processed.parquet", index=False)
        
        print(f"Processed data for {bank} saved to {out_dir / 'processed.parquet'}")


if __name__ == "__main__":
    main()
