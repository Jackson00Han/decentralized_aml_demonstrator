# 03d_fl_client_train_round.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import load_config
from src.fl_adapters import SkLogRegSGD
from src.fl_protocol import GlobalPlan, load_params_npz, save_params_npz
from src.utils import load_dataset, plan_hash


def find_dataset_dir(bank: str, client_out: Path, expected_plan_hash: str) -> Path:
    base = client_out / bank / "datasets"
    if not base.exists():
        raise FileNotFoundError(f"Missing datasets dir: {base} (run scripts/03c_fl_client_initialize.py)")

    prefix = f"plan_{expected_plan_hash[:8]}"
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No dataset matching {prefix} under {base} (run scripts/03c_fl_client_initialize.py)")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main(
    bank: str,
    round_id: int,
    data_dir: str | None,
    alpha_override: float | None = None,
    use_trainval: bool = False,
) -> None:
    cfg = load_config()
    server_out = cfg.paths.out_fl_server
    client_out = cfg.paths.out_fl_clients

    plan_path = server_out / "global_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing global plan: {plan_path} (run scripts/03b_server_build_global_plan.py)")
    expected_hash = plan_hash(plan_path)
    plan = GlobalPlan.load(plan_path)

    ds_dir = Path(data_dir) if data_dir else find_dataset_dir(bank, client_out, expected_hash)
    data = load_dataset(ds_dir, expected_hash)

    local_epochs = int(cfg.fl.local_epochs)
    alpha = float(alpha_override) if alpha_override is not None else float(cfg.fl.alpha)
    seed = int(cfg.project.seed)

    if use_trainval:
        X_tr = data["X_trainval"]
        y_tr = data["y_trainval"].astype(int)
    else:
        X_tr = data["X_train"]
        y_tr = data["y_train"].astype(int)

    n_train = int(len(y_tr))

    global_model_path = server_out / "global_model_latest.npz"
    if global_model_path.exists():
        params = load_params_npz(global_model_path)
    else:
        d = int(X_tr.shape[1])
        params = {"coef": np.zeros((1, d), dtype=float), "intercept": np.zeros((1,), dtype=float)}

    model = SkLogRegSGD(d=X_tr.shape[1], alpha=alpha, seed=seed + int(round_id))
    model.set_params(params)
    model.train_one_round(X_tr, y_tr, local_epochs=local_epochs, seed=seed + int(round_id))
    update_params = model.get_params()

    out_dir = client_out / bank / "updates"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_params_npz(
        out_dir / f"round_{round_id:03d}_update.npz",
        update_params,
        meta={
            "bank": bank,
            "round": int(round_id),
            "alpha": float(alpha),
            "n_train": int(n_train),
            "use_trainval": bool(use_trainval),
            "local_epochs": int(local_epochs),
            "schema_version": plan.schema_version,
            "plan_hash": expected_hash,
        },
    )
    print(f"Client {bank} round {round_id} model update saved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default="bank_l")
    parser.add_argument("--round_id", type=int, default=1)
    parser.add_argument("--data_dir", default=None, help="dataset dir from scripts/03c_fl_client_initialize.py")
    parser.add_argument("--alpha", type=float, default=None, help="override cfg.fl.alpha for this run")
    parser.add_argument("--use_trainval", action="store_true", help="train on train+val (skip validation)")
    args = parser.parse_args()

    main(
        args.client,
        args.round_id,
        data_dir=args.data_dir,
        alpha_override=args.alpha,
        use_trainval=args.use_trainval,
    )
