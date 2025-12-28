# 03f_fl_evaluation.py
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import load_config
from src.fl_adapters import SkLogRegSGD
from src.fl_protocol import GlobalPlan, load_params_npz
from src.fl_secure_agg import mask_value
from src.utils import load_dataset, plan_hash
from src.metrics import (
    ap,
    safe_roc_auc,
    best_f1_threshold,
    weighted_logloss_sums,
    class_balance_weights,
)


def find_dataset_dir(bank: str, client_out: Path, expected_plan_hash: str) -> Path:
    base = client_out / bank / "datasets"
    if not base.exists():
        raise FileNotFoundError(f"Missing datasets dir: {base} (run scripts/03c_fl_client_initialize.py)")

    prefix = f"plan_{expected_plan_hash[:8]}"
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No dataset matching {prefix} under {base} (run scripts/03c_fl_client_initialize.py)")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def compute_val_metrics(y_val, scores) -> dict:
    val_n = int(len(y_val))
    val_pos = int(y_val.sum())
    val_ap = float(ap(y_val, scores)) if val_pos > 0 else 0.0
    val_auc = safe_roc_auc(y_val, scores)
    if val_n > 0:
        thr, p_val, r_val, f1_val = best_f1_threshold(y_val, scores)
    else:
        thr, p_val, r_val, f1_val = 0.5, 0.0, 0.0, 0.0
    return {
        "ap": float(val_ap),
        "auc": float(val_auc) if val_auc is not None else None,
        "thr": float(thr),
        "p": float(p_val),
        "r": float(r_val),
        "f1": float(f1_val),
        "n": val_n,
    }


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
        raise FileNotFoundError(f"Missing global plan: {plan_path} (run scripts/03b_fl_server_build_global_plan.py)")
    expected_hash = plan_hash(plan_path)
    plan = GlobalPlan.load(plan_path)

    ds_dir = Path(data_dir) if data_dir else find_dataset_dir(bank, client_out, expected_hash)
    data = load_dataset(ds_dir, expected_hash)

    model_path = server_out / "global_models" / f"round_{round_id:03d}.npz"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing global model for round {round_id}: {model_path} (run scripts/03e_fl_server_aggregate.py)")

    params = load_params_npz(model_path)

    local_update_path = client_out / bank / "updates" / f"round_{round_id:03d}_update.npz"
    if not local_update_path.exists():
        raise FileNotFoundError(
            f"Missing local update for round {round_id}: {local_update_path} (run scripts/03d_fl_client_train_round.py)"
        )
    local_params = load_params_npz(local_update_path)

    X_val = data["X_val"]
    y_val = data["y_val"].astype(int)
    val_n = int(len(y_val))
    val_pos = int(y_val.sum())
    w_pos, w_neg = class_balance_weights(val_n, val_pos)

    alpha = float(alpha_override) if alpha_override is not None else float(cfg.fl.alpha)
    seed = int(cfg.project.seed)

    global_model = SkLogRegSGD(d=X_val.shape[1], alpha=alpha, seed=seed + int(round_id))
    global_model.set_params(params)
    global_p_val = global_model.predict_scores(X_val)
    global_metrics = compute_val_metrics(y_val, global_p_val)

    global_sum_wloss, global_sum_w = weighted_logloss_sums(
        y_true=y_val,
        p_pred=global_p_val,
        w_pos=w_pos,
        w_neg=w_neg,
    )

    local_model = SkLogRegSGD(d=X_val.shape[1], alpha=alpha, seed=seed + int(round_id))
    local_model.set_params(local_params)
    local_p_val = local_model.predict_scores(X_val)
    local_metrics = compute_val_metrics(y_val, local_p_val)
    local_sum_wloss, local_sum_w = weighted_logloss_sums(
        y_true=y_val,
        p_pred=local_p_val,
        w_pos=w_pos,
        w_neg=w_neg,
    )

    participants = sorted(cfg.banks.names)
    use_fk = bool(getattr(cfg.fl, "fk_key", False))
    secret = str(getattr(cfg.fl, "secure_agg_key", "")) if use_fk else None
    scale = float(getattr(cfg.fl, "metrics_mask_scale", 1000.0))

    def mask_metric(metric: float, denom: float, tag: str) -> tuple[float, float]:
        num = float(metric) * float(denom)
        num_masked = mask_value(
            num,
            me=bank,
            participants=participants,
            round_id=round_id,
            tag=f"{tag}_num",
            scale=scale,
            secret=secret,
        )
        den_masked = mask_value(
            float(denom),
            me=bank,
            participants=participants,
            round_id=round_id,
            tag=f"{tag}_den",
            scale=scale,
            secret=secret,
        )
        return num_masked, den_masked

    # Aggregate non-additive metrics as val_n-weighted averages across clients.
    val_weight = float(val_n)

    num_masked = mask_value(
        global_sum_wloss,
        me=bank,
        participants=participants,
        round_id=round_id,
        tag="global_val_wlogloss_sum",
        scale=scale,
        secret=secret,
    )
    den_masked = mask_value(
        global_sum_w,
        me=bank,
        participants=participants,
        round_id=round_id,
        tag="global_val_wsum",
        scale=scale,
        secret=secret,
    )

    global_ap_num_masked, global_ap_den_masked = mask_metric(
        global_metrics["ap"], val_weight, "global_val_ap"
    )
    global_p_num_masked, global_p_den_masked = mask_metric(
        global_metrics["p"], val_weight, "global_val_p"
    )
    global_r_num_masked, global_r_den_masked = mask_metric(
        global_metrics["r"], val_weight, "global_val_r"
    )
    global_auc_value = global_metrics["auc"]
    global_auc_weight = val_weight if global_auc_value is not None else 0.0
    global_auc_num_masked, global_auc_den_masked = mask_metric(
        0.0 if global_auc_value is None else float(global_auc_value),
        global_auc_weight,
        "global_val_auc",
    )

    local_num_masked = mask_value(
        local_sum_wloss,
        me=bank,
        participants=participants,
        round_id=round_id,
        tag="local_val_wlogloss_sum",
        scale=scale,
        secret=secret,
    )
    local_den_masked = mask_value(
        local_sum_w,
        me=bank,
        participants=participants,
        round_id=round_id,
        tag="local_val_wsum",
        scale=scale,
        secret=secret,
    )

    local_ap_num_masked, local_ap_den_masked = mask_metric(local_metrics["ap"], val_weight, "local_val_ap")
    local_p_num_masked, local_p_den_masked = mask_metric(local_metrics["p"], val_weight, "local_val_p")
    local_r_num_masked, local_r_den_masked = mask_metric(local_metrics["r"], val_weight, "local_val_r")
    local_auc_value = local_metrics["auc"]
    local_auc_weight = val_weight if local_auc_value is not None else 0.0
    local_auc_num_masked, local_auc_den_masked = mask_metric(
        0.0 if local_auc_value is None else float(local_auc_value),
        local_auc_weight,
        "local_val_auc",
    )

    out_dir = client_out / bank / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"round_{round_id:03d}_val_wlogloss.meta.json"
    out_path.write_text(
        json.dumps(
            {
                "bank": bank,
                "round": int(round_id),
                "alpha": float(alpha),
                "metric": "val_wlogloss",
                "metric_num_masked": float(num_masked),
                "metric_den_masked": float(den_masked),
                "local_metric_num_masked": float(local_num_masked),
                "local_metric_den_masked": float(local_den_masked),
                "global_val_ap_num_masked": float(global_ap_num_masked),
                "global_val_ap_den_masked": float(global_ap_den_masked),
                "global_val_auc_num_masked": float(global_auc_num_masked),
                "global_val_auc_den_masked": float(global_auc_den_masked),
                "global_val_p_num_masked": float(global_p_num_masked),
                "global_val_p_den_masked": float(global_p_den_masked),
                "global_val_r_num_masked": float(global_r_num_masked),
                "global_val_r_den_masked": float(global_r_den_masked),
                "local_val_ap_num_masked": float(local_ap_num_masked),
                "local_val_ap_den_masked": float(local_ap_den_masked),
                "local_val_auc_num_masked": float(local_auc_num_masked),
                "local_val_auc_den_masked": float(local_auc_den_masked),
                "local_val_p_num_masked": float(local_p_num_masked),
                "local_val_p_den_masked": float(local_p_den_masked),
                "local_val_r_num_masked": float(local_r_num_masked),
                "local_val_r_den_masked": float(local_r_den_masked),
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
    parser.add_argument("--data_dir", default=None, help="dataset dir from scripts/03c_fl_client_initialize.py")
    parser.add_argument("--alpha", type=float, default=None, help="override cfg.fl.alpha for meta tracking")
    args = parser.parse_args()

    main(
        args.client,
        args.round_id,
        data_dir=args.data_dir,
        alpha_override=args.alpha,
    )
