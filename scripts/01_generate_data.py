
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

def run(cmd: list[str], cwd: Path) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)

def sim_name_from_conf(conf: Path) -> str:
    data = json.loads(conf.read_text(encoding="utf-8"))
    return data.get("general", {}).get("simulation_name", conf.stem)

def main() -> None:
    import pandas as pd
    import sys
    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(REPO_ROOT))
    from src.config import load_config

    cfg = load_config()
    DATA_RAW = cfg.paths.data_raw; DATA_RAW.mkdir(parents=True, exist_ok=True)

    amlsim_dir = os.environ.get("AMLSIM_DIR", "").strip()
    if not amlsim_dir:
        raise RuntimeError("Please set AMLSIM_DIR, e.g. export AMLSIM_DIR=/path/to/AMLSim")
    amlsim_dir = Path(amlsim_dir).expanduser().resolve()
    print("Using AMLSIM_DIR:", amlsim_dir)
    scripts = amlsim_dir / "scripts"
    outputs = amlsim_dir / "outputs"

    conf = REPO_ROOT / "configs" / "amlsim" / "conf.json"
    if not conf.exists():
        raise FileNotFoundError(f"Missing config file")

    sim_name = sim_name_from_conf(conf)

    run([sys.executable, str(scripts / "transaction_graph_generator.py"), str(conf)], cwd=amlsim_dir)
    run(["bash", str(scripts / "run_AMLSim.sh"), str(conf)], cwd=amlsim_dir)
    run([sys.executable, str(scripts / "convert_logs.py"), str(conf)], cwd=amlsim_dir)

    # Copy generated data to the current data/raw folder from AMLSim outputs folder
    src_out = outputs / sim_name
    if not src_out.exists():
        raise FileNotFoundError(f"Expected outputs folder not found: {src_out} (check conf name)")

    dst_out = DATA_RAW / f"three_banks_simulation"
    if dst_out.exists():
        shutil.rmtree(dst_out)
    shutil.copytree(src_out, dst_out)
    print(f"[OK] {src_out} -> {dst_out}")

    BANKS = cfg.banks.names
    print(f"Banks used in simulation: {BANKS}")

    accounts = pd.read_csv(dst_out / "accounts.csv")
    tx = pd.read_csv(dst_out / "transactions.csv")
    sx = pd.read_csv(dst_out / "sar_accounts.csv")
    for bank in BANKS:
        out_dir = DATA_RAW / bank
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)


        # accounts.csv
        acc_b = accounts[accounts["bank_id"] == bank].copy()
        acc_b.to_csv(out_dir / "accounts.csv", index=False)
        print(f"[OK] accounts.csv for bank {bank}: {len(acc_b)} accounts")

        # transactions.csv
        acct_set = set(pd.to_numeric(acc_b["acct_id"], errors="coerce").dropna().astype(int))
        tx_b = tx[tx["orig_acct"].isin(acct_set) | tx["bene_acct"].isin(acct_set)].copy()
        tx_b.to_csv(out_dir / "transactions.csv", index=False)
        print(f"[OK] transactions.csv for bank {bank}: {len(tx_b)} transactions")

        # sar_accounts.csv
        sx_b = sx[pd.to_numeric(sx["ACCOUNT_ID"], errors="coerce").fillna(-1).astype(int).isin(acct_set)].copy()
        sx_b.to_csv(out_dir / "sar_accounts.csv", index=False)
        print(f"[OK] sar_accounts.csv for bank {bank}: {len(sx_b)} SAR accounts")

        assert (acc_b["bank_id"] == bank).all()
        assert set(sx_b["ACCOUNT_ID"]).issubset(acct_set)
        bad = tx_b[~(tx_b["orig_acct"].isin(acct_set) | tx_b["bene_acct"].isin(acct_set))]
        assert len(bad) == 0
        cross = tx_b[~(tx_b["orig_acct"].isin(acct_set) & tx_b["bene_acct"].isin(acct_set))]
        print("cross-bank visible tx:", len(cross), "total tx:", len(tx_b))


    print("Data generation completed.")
    

if __name__ == "__main__":
    main()
