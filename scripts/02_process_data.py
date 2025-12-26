import pandas as pd
import os

def main():
    import sys
    import shutil
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(REPO_ROOT))
    from src.config import load_config
    from src.data_splits import split_fixed_windows, build_account_features
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
        df_accs = df_accs[['acct_id', 'initial_deposit', 'state']]

        # process transactions
        df_trans = df_trans[['tran_id', 'tran_timestamp', 'orig_acct', 'bene_acct', 'base_amt', 'is_sar']]
        df_trans['tran_timestamp'] = pd.to_datetime(df_trans['tran_timestamp'], utc=True, errors='coerce')
        df_trans['orig_acct'] = pd.to_numeric(df_trans['orig_acct'], errors='coerce').astype('Int32')
        df_trans['bene_acct'] = pd.to_numeric(df_trans['bene_acct'], errors='coerce').astype('Int32')
        df_trans['base_amt'] = pd.to_numeric(df_trans['base_amt'], errors='coerce').astype('float32')

        # split df_trans into train, val, test
        from src.data_splits import split_fixed_windows
        tr, va, te, df_2018_used = split_fixed_windows(df_trans, ts_col="tran_timestamp")
        print(f"Bank {bank}: {len(tr)} train, {len(va)} val, {len(te)} test transactions.")

        # make account features
        from src.data_splits import build_account_features
        tr_acct_feats = build_account_features(tr, ts_col="tran_timestamp").reset_index()
        va_acct_feats = build_account_features(va, ts_col="tran_timestamp").reset_index()
        te_acct_feats = build_account_features(te, ts_col="tran_timestamp").reset_index()
        print(f"Bank {bank}: {len(tr_acct_feats)} train, {len(va_acct_feats)} val, {len(te_acct_feats)} test account features.")

        # merge account features with accounts data
        accs_tr = df_accs.merge(tr_acct_feats, on='acct_id', how='left')
        accs_va = df_accs.merge(va_acct_feats, on='acct_id', how='left')
        accs_te = df_accs.merge(te_acct_feats, on='acct_id', how='left')
        feat_cols = [
            "out_cnt",
            "out_amt_sum",
            "out_uniq_bene",
            "in_cnt",
            "in_amt_sum",
            "in_uniq_orig",
            "net_flow",
            "turnover",
            "uniq_total",
        ]
        accs_tr[feat_cols] = accs_tr[feat_cols].fillna(0)
        accs_va[feat_cols] = accs_va[feat_cols].fillna(0)
        accs_te[feat_cols] = accs_te[feat_cols].fillna(0)
        print(f"Bank {bank}: {len(accs_tr)} train, {len(accs_va)} val, {len(accs_te)} test accounts after merge.")

        # add sar label to accounts (use SAR event date cutoffs)
        df_sar = df_sar.copy()

        # Ensure datetime (AMLSim SAR dates are YYYYMMDD)
        df_sar["EVENT_DATE"] = pd.to_datetime(
            df_sar["EVENT_DATE"].astype(str),
            format="%Y%m%d",
            utc=True,
            errors="coerce",
        )

        # Ensure ids comparable
        sar_ids = pd.to_numeric(df_sar["ACCOUNT_ID"], errors="coerce").astype("Int32")

        cut_tr = pd.Timestamp("2018-05-01", tz="UTC")
        cut_va = pd.Timestamp("2018-09-01", tz="UTC")
        cut_te = pd.Timestamp("2019-01-01", tz="UTC")  # end exclusive for 2018-12-31

        sar_tr_set = set(sar_ids[df_sar["EVENT_DATE"] < cut_tr].dropna().tolist())
        sar_va_set = set(sar_ids[df_sar["EVENT_DATE"] < cut_va].dropna().tolist())
        sar_te_set = set(sar_ids[df_sar["EVENT_DATE"] < cut_te].dropna().tolist())

        accs_tr["is_sar"] = accs_tr["acct_id"].astype("Int32").isin(sar_tr_set)
        accs_va["is_sar"] = accs_va["acct_id"].astype("Int32").isin(sar_va_set)
        accs_te["is_sar"] = accs_te["acct_id"].astype("Int32").isin(sar_te_set)


        # save processed data
        accs_tr.to_csv(out_dir / "accounts_train.csv", index=False)
        accs_va.to_csv(out_dir / "accounts_val.csv", index=False)
        accs_te.to_csv(out_dir / "accounts_test.csv", index=False)

        print(f"Processed data for bank {bank} saved to {out_dir}")


if __name__ == "__main__":
    main()
