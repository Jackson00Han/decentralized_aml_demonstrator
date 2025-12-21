
from __future__ import annotations

import json
import os
import sys
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"

BANK_CONFS = {
    "bank_a": REPO_ROOT / "configs" / "amlsim" / "bank_a.json",
    "bank_b": REPO_ROOT / "configs" / "amlsim" / "bank_b.json",
    "bank_c": REPO_ROOT / "configs" / "amlsim" / "bank_c.json",
}


def run(cmd: list[str], cwd: Path) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def sim_name_from_conf(conf: Path) -> str:
    data = json.loads(conf.read_text(encoding="utf-8"))
    return data.get("general", {}).get("simulation_name", conf.stem)


def main() -> None:
    amlsim_dir = os.environ.get("AMLSIM_DIR", "").strip()
    if not amlsim_dir:
        raise RuntimeError("Please set AMLSIM_DIR, e.g. export AMLSIM_DIR=/path/to/AMLSim")
    amlsim_dir = Path(amlsim_dir).expanduser().resolve()
    print("Using AMLSIM_DIR:", amlsim_dir)

    scripts = amlsim_dir / "scripts"
    outputs = amlsim_dir / "outputs"

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Generate data for each bank configuration
    for bank, conf in BANK_CONFS.items():
        if not conf.exists():
            raise FileNotFoundError(f"Missing config for {bank}: {conf}")

        sim_name = sim_name_from_conf(conf)
        print(f"\n=== {bank} | sim_name={sim_name} ===")

        run([sys.executable, str(scripts / "transaction_graph_generator.py"), str(conf)], cwd=amlsim_dir)
        run(["bash", str(scripts / "run_AMLSim.sh"), str(conf)], cwd=amlsim_dir)
        run([sys.executable, str(scripts / "convert_logs.py"), str(conf)], cwd=amlsim_dir)


        # Copy generated data to the current data/raw folder from AMLSim outputs folder
        src_out = outputs / sim_name
        if not src_out.exists():
            raise FileNotFoundError(f"Expected outputs folder not found: {src_out} (check conf name)")

        dst_out = DATA_RAW / f"{bank}"
        if dst_out.exists():
            shutil.rmtree(dst_out)
        shutil.copytree(src_out, dst_out)
        print(f"[OK] {src_out} -> {dst_out}")

    print("\nDONE. Data in:", DATA_RAW)


if __name__ == "__main__":
    main()
