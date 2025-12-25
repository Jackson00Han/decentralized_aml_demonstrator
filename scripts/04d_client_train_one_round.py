# 04d_client_train_one_round.py
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import load_config
from src.fl_adapters import SkLogRegSGD
from src.fl_protocol import GlobalPlan, load_params_npz, save_params_npz
from src.metrics import ap
from src.utils import plan_hash, load_dataset
from src.secure_agg import mask_metric_fraction


def find_dataset_dir(bank: str, client_out: Path, expected_plan_hash: str) -> Path:
    base = client_out / bank / "datasets"
    if not base.exists():
        raise FileNotFoundError(f"Missing datasets dir: {base} (run 04c_1_client_initialization.py)")

    prefix = f"plan_{expected_plan_hash[:8]}"
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(
            f"No dataset matching {prefix} under {base} (run 04c_1_client_initialization.py)"
        )
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
        raise FileNotFoundError(f"Missing global plan: {plan_path} (run 04b_server_build_global_plan.py)")
    plan = GlobalPlan.load(plan_path)

    ds_dir = Path(data_dir) if data_dir else find_dataset_dir(bank, client_out, plan_hash(plan_path))
    data = load_dataset(ds_dir, plan_hash(plan_path))

    local_epochs = cfg.fl.local_epochs
    alpha = float(alpha_override) if alpha_override is not None else cfg.fl.alpha
    seed = cfg.project.seed

    if use_trainval:
        X_tr = data["X_trainval"]
        y_tr = data["y_trainval"].astype(int)
        val_n = 0
        val_pos = 0
        val_ap = None
        n_train = int(len(y_tr))
    else:
        X_tr = data["X_train"]
        y_tr = data["y_train"].astype(int)
        X_va = data["X_val"]
        y_va = data["y_val"].astype(int)
        val_n = int(len(y_va))
        val_pos = int(y_va.sum())
        n_train = int(len(y_tr))

    global_model_path = server_out / "global_model_latest.npz"
    if global_model_path.exists():
        params = load_params_npz(global_model_path)
    else:
        d = X_tr.shape[1]
        params = {"coef": np.zeros((1, d), dtype=float), "intercept": np.zeros((1,), dtype=float)}

################################## Core ################################################
    model = SkLogRegSGD(d=X_tr.shape[1], alpha=alpha, seed=seed + round_id)
    model.set_params(params)
    model.train_one_round(X_tr, y_tr, local_epochs=local_epochs, seed=seed + round_id)
    if not use_trainval:
        scores_va = model.predict_scores(X_va)
        val_ap = ap(y_va, scores_va)
########################################################################################
    participants = list(cfg.banks.names)

    # Use weighted fraction components
    num = float(val_ap) * float(val_n) if (val_ap is not None and int(val_n) > 0) else 0.0
    den = float(val_n)

    use_fk = bool(getattr(cfg.fl, "fk_key", False))
    secret = None
    if use_fk:
        # Insecure simulation mode: store key in config
        secret = str(getattr(cfg.fl, "secure_agg_key", "dev-only-insecure-key"))

    # Mask numerator and denominator (server must NOT have FL_SECURE_AGG_KEY)
    val_num_masked, val_den_masked = mask_metric_fraction(
        numerator=num,
        denominator=den,
        me=bank,
        participants=participants,
        round_id=round_id,
        scale=float(getattr(cfg.fl, "metrics_mask_scale", 1000.0)),
        secret=secret,
    )

    out_dir = client_out / bank / "updates"
    out_dir.mkdir(parents=True, exist_ok=True)
    update_params = model.get_params()
    save_params_npz(
        out_dir / f"round_{round_id:03d}_update.npz",
        update_params,
        meta={
            "bank": bank,
            "round": int(round_id),
            "alpha": float(alpha),
            "n_train": int(n_train),
            
            "val_num_masked": val_num_masked,
            "val_den_masked": val_den_masked,

            "schema_version": plan.schema_version,
            "split_rule": data["meta"].get("split_rule", "fixed_windows_2017_to_2018"),
        },
    )
    print(f"Client {bank} round {round_id} model update saved. val_ap={val_ap}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default="bank_a")
    parser.add_argument("--round_id", type=int, default=1)
    parser.add_argument("--data_dir", default=None, help="dataset dir from 04c_1_client_initialization.py")
    parser.add_argument("--alpha", type=float, default=None, help="override fl.alpha for this run")
    parser.add_argument("--use_trainval", action="store_true", help="train on train+val (skip val metrics)")
    args = parser.parse_args()

    main(
        args.client,
        args.round_id,
        data_dir=args.data_dir,
        alpha_override=args.alpha,
        use_trainval=args.use_trainval,
    )
