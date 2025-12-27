# 04f_fl_round1_val_eval.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.config import load_config
from src.fl_adapters import SkLogRegSGD
from src.fl_protocol import load_params_npz
from src.metrics import ap, safe_roc_auc
from src.utils import load_dataset, plan_hash

cfg = load_config()
PY = [sys.executable]


def run(cmd, *, quiet: bool = True) -> None:
    disp = []
    for c in cmd:
        s = str(c)
        disp.append(Path(s).relative_to(REPO_ROOT) if s.startswith(str(REPO_ROOT)) else s)
    if not quiet:
        print("\n>>>", " ".join(map(str, disp)))

    if quiet:
        proc = subprocess.run(list(map(str, cmd)), capture_output=True, text=True)
        if proc.returncode != 0:
            print("\n>>>", " ".join(map(str, disp)))
            print(proc.stdout, end="")
            print(proc.stderr, end="", file=sys.stderr)
            proc.check_returncode()
        return
    subprocess.run(list(map(str, cmd)), check=True)


def find_dataset_dir(bank: str, client_out: Path, expected_plan_hash: str) -> Path:
    base = client_out / bank / "datasets"
    if not base.exists():
        raise FileNotFoundError(f"Missing datasets dir: {base} (run scripts/04c_fl_client_initialize.py)")

    prefix = f"plan_{expected_plan_hash[:8]}"
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(
            f"No dataset matching {prefix} under {base} (run scripts/04c_fl_client_initialize.py)"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def fmt(x: float | None, nd: int = 6) -> str:
    if x is None:
        return "NA"
    return f"{float(x):.{nd}f}"


def eval_on_val(*, params: dict, X_val, y_val: np.ndarray, alpha: float, seed: int) -> tuple[float, float | None]:
    model = SkLogRegSGD(d=X_val.shape[1], alpha=alpha, seed=seed)
    model.set_params(params)
    scores = model.predict_scores(X_val)
    return float(ap(y_val, scores)), safe_roc_auc(y_val, scores)


def main(round_id: int = 1, alpha: float | None = None, verbose: bool = False) -> None:
    server_out = cfg.paths.out_fl_server
    client_out = cfg.paths.out_fl_clients
    banks = list(cfg.banks.names)

    alpha = float(alpha) if alpha is not None else float(cfg.fl.alpha)
    seed = int(cfg.project.seed)

    # Ensure this is "round 1 from scratch" unless caller intentionally left artifacts.
    global_latest = server_out / "global_model_latest.npz"
    if global_latest.exists():
        global_latest.unlink()

    quiet = not verbose

    # 1) client train-only stats
    print("[1/6] Client train-only stats (04a)")
    for bank in banks:
        run(PY + [str(REPO_ROOT / "scripts" / "04a_fl_client_report_stats.py"), "--client", bank], quiet=quiet)

    # 2) server builds global plan
    print("[2/6] Server builds global plan (04b)")
    run(PY + [str(REPO_ROOT / "scripts" / "04b_fl_server_build_global_plan.py")], quiet=quiet)

    # 3) clients build aligned datasets
    print("[3/6] Clients build aligned datasets (04c)")
    for bank in banks:
        run(
            PY
            + [
                str(REPO_ROOT / "scripts" / "04c_fl_client_initialize.py"),
                "--client",
                bank,
                "--overwrite",
            ],
            quiet=quiet,
        )

    # 4) clients train one round (local updates)
    print(f"[4/6] Clients train one round (04d) | round_id={round_id} alpha={alpha:g}")
    for bank in banks:
        run(
            PY
            + [
                str(REPO_ROOT / "scripts" / "04d_fl_client_train_round.py"),
                "--client",
                bank,
                "--round_id",
                round_id,
                "--alpha",
                alpha,
            ],
            quiet=quiet,
        )

    # 5) server aggregates (FedAvg) -> global_model_latest.npz
    print("[5/6] Server aggregates to global model (04e)")
    run(PY + [str(REPO_ROOT / "scripts" / "04e_fl_server_aggregate.py"), "--round_id", round_id], quiet=quiet)

    # 6) each client evaluates local update vs aggregated global model on its own val
    print("[6/6] Client val eval: local update vs global model")
    plan_path = server_out / "global_plan.json"
    expected_hash = plan_hash(plan_path)

    global_params = load_params_npz(global_latest)

    total_n = 0
    local_num = 0.0
    global_num = 0.0

    for bank in banks:
        ds_dir = find_dataset_dir(bank, client_out, expected_hash)
        data = load_dataset(ds_dir, expected_hash)
        X_val = data["X_val"]
        y_val = data["y_val"].astype(int)

        val_n = int(len(y_val))
        val_pos = int(y_val.sum())

        local_upd_path = client_out / bank / "updates" / f"round_{round_id:03d}_update.npz"
        local_params = load_params_npz(local_upd_path)

        local_ap, local_auc = eval_on_val(params=local_params, X_val=X_val, y_val=y_val, alpha=alpha, seed=seed)
        global_ap, global_auc = eval_on_val(params=global_params, X_val=X_val, y_val=y_val, alpha=alpha, seed=seed)

        print(
            f"{bank}: val_n={val_n} pos={val_pos} | "
            f"local_ap={fmt(local_ap)} local_auc={fmt(local_auc)} | "
            f"global_ap={fmt(global_ap)} global_auc={fmt(global_auc)}"
        )

        total_n += val_n
        local_num += local_ap * val_n
        global_num += global_ap * val_n

    if total_n > 0:
        print(
            f"\nWeighted avg (by val_n): local_ap={fmt(local_num / total_n)} | global_ap={fmt(global_num / total_n)}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--round_id", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=None, help="override cfg.fl.alpha for this run")
    parser.add_argument("--verbose", action="store_true", help="print subcommands and their output")
    args = parser.parse_args()

    main(round_id=args.round_id, alpha=args.alpha, verbose=args.verbose)

