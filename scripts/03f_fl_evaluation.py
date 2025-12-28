# 04f_fl_evaluation.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import load_config
from src.fl_adapters import SkLogRegSGD
from src.fl_protocol import GlobalPlan, load_params_npz
from src.fl_secure_agg import mask_value
from src.utils import load_dataset, plan_hash


def find_dataset_dir(bank: str, client_out: Path, expected_plan_hash: str) -> Path:
    base = client_out / bank / "datasets"
    if not base.exists():
        raise FileNotFoundError(f"Missing datasets dir: {base} (run scripts/04c_fl_client_initialize.py)")

    prefix = f"plan_{expected_plan_hash[:8]}"
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No dataset matching {prefix} under {base} (run scripts/04c_fl_client_initialize.py)")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def logloss_sum(y_true: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-6
    p = np.clip(p.astype(float), eps, 1.0 - eps)
    y = y_true.astype(float)
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(loss.sum())


def main(
    bank: str,
    round_id: int,
    data_dir: str | None,
    alpha_override: float | None = None,
) -> None:
    cfg = load_config()
    server_out = cfg.paths.out_fl_server
    client_out = cfg.paths.out_fl_clients

    plan_path = server_out / "global_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing global plan: {plan_path} (run scripts/04b_fl_server_build_global_plan.py)")
    expected_hash = plan_hash(plan_path)
    plan = GlobalPlan.load(plan_path)

    ds_dir = Path(data_dir) if data_dir else find_dataset_dir(bank, client_out, expected_hash)
    data = load_dataset(ds_dir, expected_hash)

    model_path = server_out / "global_models" / f"round_{round_id:03d}.npz"
    if not model_path.exists():
        model_path = server_out / "global_model_latest.npz"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing global model for round {round_id}: {model_path} (run scripts/04e_fl_server_aggregate.py)")

    params = load_params_npz(model_path)

    local_update_path = client_out / bank / "updates" / f"round_{round_id:03d}_update.npz"
    if not local_update_path.exists():
        raise FileNotFoundError(
            f"Missing local update for round {round_id}: {local_update_path} (run scripts/04d_fl_client_train_round.py)"
        )
    local_params = load_params_npz(local_update_path)

    X_val = data["X_val"]
    y_val = data["y_val"].astype(int)
    val_n = int(len(y_val))

    alpha = float(alpha_override) if alpha_override is not None else float(cfg.fl.alpha)
    seed = int(cfg.project.seed)

    global_model = SkLogRegSGD(d=X_val.shape[1], alpha=alpha, seed=seed + int(round_id))
    global_model.set_params(params)
    global_p_val = global_model.predict_scores(X_val)
    val_logloss_sum = logloss_sum(y_val, global_p_val) if val_n > 0 else 0.0

    local_model = SkLogRegSGD(d=X_val.shape[1], alpha=alpha, seed=seed + int(round_id))
    local_model.set_params(local_params)
    local_p_val = local_model.predict_scores(X_val)
    local_val_logloss_sum = logloss_sum(y_val, local_p_val) if val_n > 0 else 0.0

    participants = list(cfg.banks.names)
    use_fk = bool(getattr(cfg.fl, "fk_key", False))
    secret = str(getattr(cfg.fl, "secure_agg_key", "")) if use_fk else None
    scale = float(getattr(cfg.fl, "metrics_mask_scale", 1000.0))

    num_masked = mask_value(
        val_logloss_sum,
        me=bank,
        participants=participants,
        round_id=round_id,
        tag="global_val_logloss_sum",
        scale=scale,
        secret=secret,
    )
    den_masked = mask_value(
        float(val_n),
        me=bank,
        participants=participants,
        round_id=round_id,
        tag="val_logloss_n",
        scale=scale,
        secret=secret,
    )

    local_num_masked = mask_value(
        local_val_logloss_sum,
        me=bank,
        participants=participants,
        round_id=round_id,
        tag="local_val_logloss_sum",
        scale=scale,
        secret=secret,
    )
    local_den_masked = den_masked

    out_dir = client_out / bank / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"round_{round_id:03d}_val_logloss.meta.json"
    out_path.write_text(
        json.dumps(
            {
                "bank": bank,
                "round": int(round_id),
                "alpha": float(alpha),
                "metric": "val_logloss",
                "metric_num_masked": float(num_masked),
                "metric_den_masked": float(den_masked),
                "local_metric_num_masked": float(local_num_masked),
                "local_metric_den_masked": float(local_den_masked),
                "schema_version": plan.schema_version,
                "plan_hash": expected_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Client {bank} round {round_id} evaluation saved: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default="bank_l")
    parser.add_argument("--round_id", type=int, default=1)
    parser.add_argument("--data_dir", default=None, help="dataset dir from scripts/04c_fl_client_initialize.py")
    parser.add_argument("--alpha", type=float, default=None, help="override cfg.fl.alpha for meta tracking")
    args = parser.parse_args()

    main(
        args.client,
        args.round_id,
        data_dir=args.data_dir,
        alpha_override=args.alpha,
    )
