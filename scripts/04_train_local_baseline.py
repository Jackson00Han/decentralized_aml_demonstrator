from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import load_config
from src.fl_adapters import SkLogRegSGD
from src.fl_protocol import GlobalPlan, save_params_npz
from src.metrics import (
    ap,
    best_f1_threshold,
    precision_recall_f1_at_threshold,
    safe_roc_auc,
    weighted_logloss_sums,
    class_balance_weights,
)
from src.utils import load_dataset, plan_hash


def fmt(x: float | None, nd: int = 6) -> str:
    if x is None:
        return "NA"
    return f"{float(x):.{nd}f}"


def find_dataset_dir(bank: str, client_out: Path, expected_plan_hash: str) -> Path:
    base = client_out / bank / "datasets"
    if not base.exists():
        raise FileNotFoundError(f"Missing datasets dir: {base} (run scripts/04c_fl_client_initialize.py)")

    prefix = f"plan_{expected_plan_hash[:8]}"
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No dataset matching {prefix} under {base} (run scripts/04c_fl_client_initialize.py)")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def wlogloss_mean(y_true: np.ndarray, p: np.ndarray, w_pos: float, w_neg: float) -> float:
    """Weighted logloss mean using class-balance weights."""
    sum_wloss, sum_w = weighted_logloss_sums(y_true, p, w_pos=w_pos, w_neg=w_neg)
    return float(sum_wloss / sum_w) if sum_w > 0 else 0.0


def train_select_best(
    *,
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    w_pos: float,
    w_neg: float,
    alpha_grid: list[float],
    max_rounds: int,
    patience: int,
    local_epochs: int,
    seed: int,
) -> dict:
    candidates: list[dict] = []
    for alpha in alpha_grid:
        model = SkLogRegSGD(d=X_train.shape[1], alpha=float(alpha), seed=int(seed))
        best_val_wlogloss = float("inf")
        best_val_ap = None
        best_params = None
        best_round = 0
        no_improve = 0

        for r in range(1, int(max_rounds) + 1):
            model.train_one_round(X_train, y_train, local_epochs=int(local_epochs), seed=int(seed) + int(r))
            val_scores = model.predict_scores(X_val)
            val_ll = wlogloss_mean(y_val, val_scores, w_pos=w_pos, w_neg=w_neg)
            val_ap = float(ap(y_val, val_scores)) if y_val.sum() > 0 else 0.0

            if val_ll < best_val_wlogloss - 1e-12:
                best_val_wlogloss = float(val_ll)
                best_val_ap = float(val_ap)
                best_params = model.get_params()
                best_round = int(r)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= int(patience):
                    break

        if best_params is None:
            raise RuntimeError(f"No params saved for alpha={float(alpha):g} (check training loop).")

        candidates.append(
            {
                "alpha": float(alpha),
                "selected_round": int(best_round),
                "val_wlogloss_best": float(best_val_wlogloss),
                "val_ap_at_best": float(best_val_ap) if best_val_ap is not None else None,
                "params": best_params,
            }
        )

    return min(candidates, key=lambda x: x["val_wlogloss_best"])


def main(client: str | None = None, save_model: bool = True) -> None:
    import shutil
    cfg = load_config()
    server_out = cfg.paths.out_fl_server
    client_out = cfg.paths.out_fl_clients

    plan_path = server_out / "global_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing global plan: {plan_path} (run scripts/04b_fl_server_build_global_plan.py)")
    expected_hash = plan_hash(plan_path)
    plan = GlobalPlan.load(plan_path)

    alpha_grid = [float(x) for x in cfg.baseline.alpha_grid]
    max_rounds = int(cfg.baseline.max_rounds)
    patience = int(cfg.baseline.patience)
    local_epochs = int(cfg.fl.local_epochs)
    seed = int(cfg.project.seed)

    out_root = cfg.paths.out_local_baseline
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=False)

    banks = [client] if client is not None else list(cfg.banks.names)
    summary_rows: list[dict] = []

    for bank in banks:
        ds_dir = find_dataset_dir(bank, client_out, expected_hash)
        data = load_dataset(ds_dir, expected_hash)

        X_train = data["X_train"]
        y_train = data["y_train"].astype(int)
        X_val = data["X_val"]
        y_val = data["y_val"].astype(int)
        X_test = data["X_test"]
        y_test = data["y_test"].astype(int)
        val_n = int(len(y_val))
        val_pos = int(y_val.sum())
        test_n = int(len(y_test))
        test_pos = int(y_test.sum())

        w_pos_val, w_neg_val = class_balance_weights(val_n, val_pos)
        w_pos_test, w_neg_test = class_balance_weights(test_n, test_pos)

        print("\n" + "=" * 68)
        print(f"Local baseline (same split as 04c) | {bank}")
        print("=" * 68)
        print(
            f"Split: train={len(y_train)} (pos={int(y_train.sum())}) | "
            f"val={len(y_val)} (pos={int(y_val.sum())}) | "
            f"test={len(y_test)} (pos={int(y_test.sum())})"
        )

        best = train_select_best(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            w_pos=w_pos_val,
            w_neg=w_neg_val,
            alpha_grid=alpha_grid,
            max_rounds=max_rounds,
            patience=patience,
            local_epochs=local_epochs,
            seed=seed,
        )

        best_alpha = float(best["alpha"])
        best_round = int(best["selected_round"])
        best_val_wlogloss = float(best["val_wlogloss_best"])
        best_params = best["params"]

        print(
            f"Selected by val_wlogloss: alpha={best_alpha:g} round={best_round:03d} "
            f"val_wlogloss={best_val_wlogloss:.6f}"
        )

        model = SkLogRegSGD(d=X_train.shape[1], alpha=best_alpha, seed=seed)
        model.set_params(best_params)

        val_scores = model.predict_scores(X_val)
        val_ap = float(ap(y_val, val_scores)) if y_val.sum() > 0 else 0.0
        val_wlogloss = wlogloss_mean(y_val, val_scores, w_pos=w_pos_val, w_neg=w_neg_val)

        if len(y_val) == 0:
            thr = 0.5
            p_val = r_val = f1_val = 0.0
        else:
            thr, p_val, r_val, f1_val = best_f1_threshold(y_val, val_scores)

        test_scores = model.predict_scores(X_test)
        test_ap = float(ap(y_test, test_scores)) if y_test.sum() > 0 else 0.0
        test_auc = safe_roc_auc(y_test, test_scores)
        test_wlogloss = wlogloss_mean(y_test, test_scores, w_pos=w_pos_test, w_neg=w_neg_test)
        p_test, r_test, f1_test = precision_recall_f1_at_threshold(y_test, test_scores, float(thr))

        print(
            f"Val:  wlogloss={val_wlogloss:.6f} ap={val_ap:.6f} "
            f"| thr={float(thr):.6f} P={p_val:.4f} R={r_val:.4f} F1={f1_val:.4f}"
        )
        print(
            f"Test: wlogloss={test_wlogloss:.6f} ap={test_ap:.6f} auc={fmt(test_auc)} "
            f"| P={p_test:.4f} R={r_test:.4f} F1={f1_test:.4f}"
        )

        bank_out = out_root / bank
        bank_out.mkdir(parents=True, exist_ok=True)

        report = {
            "bank": bank,
            "selection_metric": "val_wlogloss",
            "selected_alpha": best_alpha,
            "selected_round": best_round,
            "val_wlogloss_best": best_val_wlogloss,
            "val_ap_at_best": float(best.get("val_ap_at_best")) if best.get("val_ap_at_best") is not None else None,
            "train_n": int(len(y_train)),
            "train_pos": int(y_train.sum()),
            "val_n": val_n,
            "val_pos": val_pos,
            "test_n": test_n,
            "test_pos": test_pos,
            "val_threshold": float(thr),
            "val_p": float(p_val),
            "val_r": float(r_val),
            "val_f1": float(f1_val),
            "val_wlogloss": float(val_wlogloss),
            "val_logloss": float(val_wlogloss),  # backward-compat field name
            "val_ap": float(val_ap),
            "test_wlogloss": float(test_wlogloss),
            "test_logloss": float(test_wlogloss),  # backward-compat field name
            "test_ap": float(test_ap),
            "test_roc_auc": float(test_auc) if test_auc is not None else None,
            "test_p": float(p_test),
            "test_r": float(r_test),
            "test_f1": float(f1_test),
            "plan_hash": expected_hash,
            "schema_version": plan.schema_version,
            "dataset_dir": str(ds_dir),
        }
        (bank_out / "val_test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        if save_model:
            save_params_npz(
                bank_out / "best_model.npz",
                best_params,
                meta={
                    "bank": bank,
                    "selection_metric": "val_logloss",
                    "selected_alpha": best_alpha,
                    "selected_round": best_round,
                    "val_wlogloss_best": best_val_wlogloss,
                    "plan_hash": expected_hash,
                    "schema_version": plan.schema_version,
                },
            )

        summary_rows.append(
            {
                "bank": bank,
                #"sld_alpha": best_alpha,
                "sld_round": best_round,
                "val_wlogloss_best": best_val_wlogloss,
                "val_ap": val_ap,
                "test_wlogloss": test_wlogloss,
                "test_ap": test_ap,
                "test_roc_auc": test_auc if test_auc is not None else np.nan,
                "test_f1": f1_test,
                "test_p": p_test,
                "test_r": r_test,
                "test_n": test_n,
                "test_pos": test_pos,
            }
        )

    df = pd.DataFrame(summary_rows).sort_values("bank")
    out_summary = out_root / "summary.csv"
    df.to_csv(out_summary, index=False)

    print("\n" + "=" * 68)
    print("Baseline summary")
    print("=" * 68)
    print(df.to_string(index=False))
    print(f"\nSaved: {out_summary}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default=None, help="run only one bank (default: all cfg.banks.names)")
    parser.add_argument("--no_save_model", action="store_true", help="do not write best_model.npz")
    args = parser.parse_args()

    main(client=args.client, save_model=not args.no_save_model)
