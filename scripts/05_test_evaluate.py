from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.config import load_config
from src.fl_adapters import SkLogRegSGD
from src.fl_protocol import load_params_npz
from src.metrics import (
    ap,
    safe_roc_auc,
    best_f1_threshold,
    precision_recall_f1_at_threshold,
    class_balance_weights,
    weighted_logloss_sums,
)
from src.utils import load_dataset, plan_hash

cfg = load_config()


def fmt(x: float | None, nd: int = 6) -> str:
    if x is None:
        return "NA"
    return f"{float(x):.{nd}f}"


def find_dataset_dir(bank: str, client_out: Path, expected_plan_hash: str) -> Path:
    base = client_out / bank / "datasets"
    if not base.exists():
        raise FileNotFoundError(f"Missing datasets dir: {base} (run scripts/03c_fl_client_initialize.py)")

    prefix = f"plan_{expected_plan_hash[:8]}"
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(
            f"No dataset matching {prefix} under {base} (run scripts/03c_fl_client_initialize.py)"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline_report(baseline_out: Path, bank: str) -> tuple[dict | None, Path | None]:
    p1 = baseline_out / bank / "val_test_report.json"
    if p1.exists():
        return load_json(p1), p1

    p2 = baseline_out / bank / "test_report.json"
    if p2.exists():
        return load_json(p2), p2

    return None, None


def eval_params_on_test(
    *,
    params: dict,
    X_val,
    y_val: np.ndarray,
    X_test,
    y_test: np.ndarray,
    alpha: float,
    seed: int,
) -> dict:
    model = SkLogRegSGD(d=X_test.shape[1], alpha=float(alpha), seed=int(seed))
    model.set_params(params)
    val_scores = model.predict_scores(X_val)
    thr, p_val, r_val, f1_val = best_f1_threshold(y_val, val_scores)

    test_scores = model.predict_scores(X_test)

    test_ap = float(ap(y_test, test_scores)) if y_test.sum() > 0 else 0.0
    test_auc = safe_roc_auc(y_test, test_scores)
    p_test, r_test, f1_test = precision_recall_f1_at_threshold(y_test, test_scores, thr)
    w_pos_test, w_neg_test = class_balance_weights(int(len(y_test)), int(y_test.sum()))
    sum_wloss, sum_w = weighted_logloss_sums(y_true=y_test, p_pred=test_scores, w_pos=w_pos_test, w_neg=w_neg_test)
    test_wlogloss = float(sum_wloss / sum_w) if sum_w > 0 else None

    return {
        "val_threshold": float(thr),
        "val_p": float(p_val),
        "val_r": float(r_val),
        "val_f1": float(f1_val),
        "test_ap": float(test_ap),
        "test_roc_auc": float(test_auc) if test_auc is not None else None,
        "test_p": float(p_test),
        "test_r": float(r_test),
        "test_f1": float(f1_test),
        "test_wlogloss": test_wlogloss,
    }


def main() -> None:
    server_out = cfg.paths.out_fl_server
    client_out = cfg.paths.out_fl_clients
    baseline_out = cfg.paths.out_local_baseline

    fl_model_path = server_out / "global_model_best.npz"
    if not fl_model_path.exists():
        raise FileNotFoundError(
            f"Missing best model: {fl_model_path} (save best model as global_model_best.npz first)"
        )
    fl_params = load_params_npz(fl_model_path)
    seed = int(cfg.project.seed)

    plan_path = server_out / "global_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing global plan: {plan_path} (run scripts/03b_fl_server_build_global_plan.py)")
    expected_hash = plan_hash(plan_path)

    print(f"Using FL model: {fl_model_path}")

    out_root = server_out / "test_eval"
    out_root.mkdir(parents=True, exist_ok=True)

    rows_fl: list[dict] = []
    rows_cmp: list[dict] = []

    for bank in cfg.banks.names:
        ds_dir = find_dataset_dir(bank, client_out, expected_hash)
        data = load_dataset(ds_dir, expected_hash)
        X_val = data["X_val"]
        y_val = data["y_val"].astype(int)
        X_test = data["X_test"]
        y_test = data["y_test"].astype(int)
        test_n = int(len(y_test))
        test_pos = int(y_test.sum())

        fl_alpha = float(cfg.fl.alpha)
        fl_metrics = eval_params_on_test(
            params=fl_params,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            alpha=fl_alpha,
            seed=seed,
        )

        baseline_report, baseline_path = load_baseline_report(baseline_out, bank)

        print("\n" + "=" * 68)
        print(f"Test evaluation | {bank}")
        print("=" * 68)
        print(f"Test split: n={test_n} pos={test_pos}")
        if baseline_report is None:
            print("Base:        NA (missing baseline report)")
        else:
            base_n = baseline_report.get("test_n", None)
            base_pos = baseline_report.get("test_pos", None)
            if base_n is None or base_pos is None:
                base_topk = baseline_report.get("topk", {}) if isinstance(baseline_report.get("topk", {}), dict) else {}
                base_n = base_topk.get("n", None)
                base_pos = base_topk.get("pos", None)
            if base_n is not None and base_pos is not None:
                base_n = int(base_n)
                base_pos = int(base_pos)
                if base_n != test_n or base_pos != test_pos:
                    print(
                        f"WARNING: baseline split differs (baseline n={base_n} pos={base_pos}) "
                        f"vs current test (n={test_n} pos={test_pos}); deltas may be misleading."
                    )
            base_wll = baseline_report.get("test_wlogloss", baseline_report.get("test_logloss"))
            print(
                f"Base:        wlogloss={fmt(base_wll)} AP={fmt(baseline_report.get('test_ap'))} "
                f"AUC={fmt(baseline_report.get('test_roc_auc'))} "
                f"F1={fmt(baseline_report.get('test_f1'), nd=4)}"
            )
        print(
            f"FL(03):       wlogloss={fmt(fl_metrics['test_wlogloss'])} "
            f"AP={fmt(fl_metrics['test_ap'])} AUC={fmt(fl_metrics['test_roc_auc'])} "
            f"F1={fmt(fl_metrics['test_f1'], nd=4)} (thr={fl_metrics['val_threshold']:.6f} from val)"
        )
        if baseline_report is not None:
            try:
                delta_p = float(fl_metrics["test_p"]) - float(baseline_report.get("test_p", 0.0))
            except Exception:
                delta_p = None
            try:
                delta_r = float(fl_metrics["test_r"]) - float(baseline_report.get("test_r", 0.0))
            except Exception:
                delta_r = None
            try:
                delta_ap = float(fl_metrics["test_ap"]) - float(baseline_report.get("test_ap", 0.0))
            except Exception:
                delta_ap = None
            try:
                b_auc = baseline_report.get("test_roc_auc", None)
                delta_auc = (
                    float(fl_metrics["test_roc_auc"]) - float(b_auc)
                    if fl_metrics["test_roc_auc"] is not None and b_auc is not None
                    else None
                )
            except Exception:
                delta_auc = None
            try:
                delta_wll = float(fl_metrics["test_wlogloss"]) - float(
                    baseline_report.get("test_wlogloss", baseline_report.get("test_logloss", 0.0))
                )
            except Exception:
                delta_wll = None
            try:
                delta_f1 = float(fl_metrics["test_f1"]) - float(baseline_report.get("test_f1", 0.0))
            except Exception:
                delta_f1 = None
            print(
                f"Delta(FL - Base): P={fmt(delta_p)} R={fmt(delta_r)} "
                f"AP={fmt(delta_ap)} AUC={fmt(delta_auc)} wlogloss={fmt(delta_wll)} "
                f"F1={fmt(delta_f1, nd=4)}"
            )

        bank_out = out_root / bank
        bank_out.mkdir(parents=True, exist_ok=True)

        fl_out = {
            "bank": bank,
            "model_path": str(fl_model_path),
            "test_n": int(test_n),
            "test_pos": int(test_pos),
            **fl_metrics,
        }
        (bank_out / "fl_test_report.json").write_text(json.dumps(fl_out, indent=2), encoding="utf-8")

        cmp_out = {
            "bank": bank,
            "base_path": str(baseline_path) if baseline_path is not None else None,
            "base": (
                None
                if baseline_report is None
                else {
                    "bank": baseline_report.get("bank", bank),
                    "selected_alpha": baseline_report.get("selected_alpha"),
                    "selected_round": baseline_report.get("selected_round"),
                    "val_ap_best": baseline_report.get("val_ap_best"),
                    "val_wlogloss_best": baseline_report.get("val_wlogloss_best"),
                    "ap": baseline_report.get("test_ap"),
                    "roc_auc": baseline_report.get("test_roc_auc"),
                    "wlogloss": baseline_report.get("test_wlogloss", baseline_report.get("test_logloss")),
                    "p": baseline_report.get("test_p"),
                    "r": baseline_report.get("test_r"),
                    "f1": baseline_report.get("test_f1"),
                }
            ),
            "fl": fl_out,
        }
        (bank_out / "compare_report.json").write_text(json.dumps(cmp_out, indent=2), encoding="utf-8")

        rows_fl.append(
            {
                "bank": bank,
                "test_ap": fl_metrics["test_ap"],
                "roc_auc": fl_metrics["test_roc_auc"] if fl_metrics["test_roc_auc"] is not None else np.nan,
                "test_f1": fl_metrics["test_f1"],
                "test_p": fl_metrics["test_p"],
                "test_r": fl_metrics["test_r"],
                "val_thr": fl_metrics["val_threshold"],
                "test_wlogloss": fl_metrics["test_wlogloss"] if fl_metrics["test_wlogloss"] is not None else np.nan,
            }
        )

        base = baseline_report or {}

        def _f(x):
            return float(x) if x is not None else np.nan

        def _delta(a, b):
            if a is None or b is None:
                return np.nan
            return float(a) - float(b)

        rows_cmp.append(
            {
                "bank": bank,
                "base_p": _f(base.get("test_p")),
                "fl_p": _f(fl_metrics.get("test_p")),
                "delta_p": _delta(fl_metrics.get("test_p"), base.get("test_p")),
                "base_r": _f(base.get("test_r")),
                "fl_r": _f(fl_metrics.get("test_r")),
                "delta_r": _delta(fl_metrics.get("test_r"), base.get("test_r")),
                "base_ap": _f(base.get("test_ap")),
                "fl_ap": float(fl_metrics["test_ap"]),
                "delta_ap": _delta(fl_metrics["test_ap"], base.get("test_ap")),
                "base_auc": _f(base.get("test_roc_auc")),
                "fl_auc": _f(fl_metrics["test_roc_auc"]),
                "delta_auc": _delta(fl_metrics["test_roc_auc"], base.get("test_roc_auc")),
                "base_wlogloss": _f(base.get("test_wlogloss", base.get("test_logloss"))),
                "fl_wlogloss": _f(fl_metrics.get("test_wlogloss")),
                "delta_wlogloss": _delta(
                    fl_metrics.get("test_wlogloss"),
                    base.get("test_wlogloss", base.get("test_logloss")),
                ),
                "base_f1": _f(base.get("test_f1")),
                "fl_f1": _f(fl_metrics.get("test_f1")),
                "delta_f1": _delta(fl_metrics.get("test_f1"), base.get("test_f1")),
            }
        )

    df_fl = pd.DataFrame(rows_fl).sort_values("bank")
    out_fl = out_root / "summary_fl.csv"
    df_fl.to_csv(out_fl, index=False)

    df_cmp = pd.DataFrame(rows_cmp).sort_values("bank")
    out_cmp = out_root / "summary_compare.csv"
    df_cmp.to_csv(out_cmp, index=False)

    print("\n" + "=" * 68)
    print("FL Summary")
    print("=" * 68)
    print(df_fl.to_string(index=False))
    print(f"\nSaved: {out_fl}")

    print("\n" + "=" * 68)
    print("Compare Summary (FL vs Baseline)")
    print("=" * 68)
    print(df_cmp.to_string(index=False))
    print(f"\nSaved: {out_cmp}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main()
