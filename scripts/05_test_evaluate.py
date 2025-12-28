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
from src.metrics import ap, safe_roc_auc, best_f1_threshold, precision_recall_f1_at_threshold
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


def find_best_fl_model(server_out: Path, alpha: float | None = None) -> dict:
    meta_files = sorted(server_out.glob("global_model_best_alpha*.meta.json"))
    if not meta_files:
        raise RuntimeError(f"No best-alpha meta files under {server_out}. Run scripts/03_fl_train.py first.")

    candidates: list[dict] = []
    for mf in meta_files:
        d = load_json(mf)
        if "alpha" not in d or "round_id" not in d:
            continue
        a = float(d["alpha"])
        if alpha is not None and abs(a - float(alpha)) > 1e-12:
            continue

        metric_name = None
        metric_value = None
        if d.get("avg_val_logloss", None) is not None:
            metric_name = "avg_val_logloss"
            metric_value = float(d["avg_val_logloss"])
        elif d.get("avg_val_ap", None) is not None:
            metric_name = "avg_val_ap"
            metric_value = float(d["avg_val_ap"])
        else:
            continue

        model_path = mf.with_suffix("").with_suffix(".npz")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file for meta: {model_path}")

        candidates.append(
            {
                "alpha": a,
                "round_id": int(d["round_id"]),
                "selection_metric": metric_name,
                "selection_value": metric_value,
                "model_path": model_path,
                "meta_path": mf,
                "meta": d,
            }
        )

    if not candidates:
        if alpha is None:
            raise RuntimeError(f"No usable FL meta found under {server_out}.")
        raise RuntimeError(f"No usable FL meta found for alpha={alpha:g} under {server_out}.")

    has_logloss = any(c["selection_metric"] == "avg_val_logloss" for c in candidates)
    if has_logloss:
        cand = [c for c in candidates if c["selection_metric"] == "avg_val_logloss"]
        return min(cand, key=lambda x: x["selection_value"])

    return max(candidates, key=lambda x: x["selection_value"])


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
    }


def main(alpha: float | None = None, model_path: str | None = None, baseline_dir: str | None = None) -> None:
    server_out = cfg.paths.out_fl_server
    client_out = cfg.paths.out_fl_clients
    baseline_out = Path(baseline_dir) if baseline_dir is not None else cfg.paths.out_local_baseline

    if model_path is not None:
        fl_model_path = Path(model_path)
        if not fl_model_path.exists():
            raise FileNotFoundError(f"Missing --model_path: {fl_model_path}")
        fl_info = {
            "alpha": float(alpha) if alpha is not None else None,
            "round_id": None,
            "selection_metric": "manual_model_path",
            "selection_value": None,
            "model_path": fl_model_path,
            "meta_path": None,
            "meta": None,
        }
    else:
        fl_info = find_best_fl_model(server_out, alpha=alpha)
        fl_model_path = fl_info["model_path"]

    fl_params = load_params_npz(fl_model_path)
    seed = int(cfg.project.seed)

    plan_path = server_out / "global_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing global plan: {plan_path} (run scripts/03b_fl_server_build_global_plan.py)")
    expected_hash = plan_hash(plan_path)

    sel_metric = fl_info["selection_metric"]
    sel_value = fl_info["selection_value"]
    if fl_info["alpha"] is None:
        print(f"Using FL model: {fl_model_path}")
    else:
        extra = "" if sel_value is None else f" | {sel_metric}={sel_value:.6f}"
        rid = "NA" if fl_info["round_id"] is None else f"{int(fl_info['round_id']):03d}"
        print(f"Using FL model: alpha={fl_info['alpha']:g} round={rid}{extra} | {fl_model_path}")

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

        fl_alpha = float(fl_info["alpha"]) if fl_info["alpha"] is not None else float(cfg.fl.alpha)
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
        if baseline_report is None:
            print("Baseline(03): NA (missing baseline report)")
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
                        f"WARNING: baseline report split differs (baseline n={base_n} pos={base_pos}) "
                        f"vs current test (n={test_n} pos={test_pos}); deltas are not apples-to-apples."
                    )
            print(
                f"Baseline(03): AP={fmt(baseline_report.get('test_ap'))} "
                f"AUC={fmt(baseline_report.get('test_roc_auc'))}"
            )
        print(
            f"FL(03):       AP={fmt(fl_metrics['test_ap'])} AUC={fmt(fl_metrics['test_roc_auc'])} "
            f"F1={fmt(fl_metrics['test_f1'], nd=4)} (thr={fl_metrics['val_threshold']:.6f} from val)"
        )

        if baseline_report is not None:
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
            print(f"Delta(FL - Baseline): AP={fmt(delta_ap)} AUC={fmt(delta_auc)}")

        bank_out = out_root / bank
        bank_out.mkdir(parents=True, exist_ok=True)

        fl_out = {
            "bank": bank,
            "selected_alpha": fl_info["alpha"],
            "selected_round": fl_info["round_id"],
            "selection_metric": fl_info["selection_metric"],
            "selection_value": fl_info["selection_value"],
            "model_path": str(fl_model_path),
            "test_n": int(test_n),
            "test_pos": int(test_pos),
            **fl_metrics,
        }
        (bank_out / "fl_test_report.json").write_text(json.dumps(fl_out, indent=2), encoding="utf-8")

        cmp_out = {
            "bank": bank,
            "baseline_path": str(baseline_path) if baseline_path is not None else None,
            "baseline": (
                None
                if baseline_report is None
                else {
                    "bank": baseline_report.get("bank", bank),
                    "selected_alpha": baseline_report.get("selected_alpha"),
                    "selected_round": baseline_report.get("selected_round"),
                    "val_ap_best": baseline_report.get("val_ap_best"),
                    "test_ap": baseline_report.get("test_ap"),
                    "test_roc_auc": baseline_report.get("test_roc_auc"),
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
                "pos": int(test_pos),
                "n": int(test_n),
                "selected_alpha": fl_info["alpha"] if fl_info["alpha"] is not None else np.nan,
                "selected_round": fl_info["round_id"] if fl_info["round_id"] is not None else np.nan,
                "selection_metric": fl_info["selection_metric"],
                "selection_value": fl_info["selection_value"] if fl_info["selection_value"] is not None else np.nan,
            }
        )

        # Comparison row (baseline + FL + delta)
        base = baseline_report or {}
        base_n = base.get("test_n", None)
        base_pos = base.get("test_pos", None)
        if base_n is None or base_pos is None:
            base_topk = base.get("topk", {}) if isinstance(base.get("topk", {}), dict) else {}
            base_n = base_topk.get("n", None)
            base_pos = base_topk.get("pos", None)

        def _f(x):
            return float(x) if x is not None else np.nan

        def _i(x):
            return int(x) if x is not None else np.nan

        def _delta(a, b):
            if a is None or b is None:
                return np.nan
            return float(a) - float(b)

        rows_cmp.append(
            {
                "bank": bank,
                "baseline_n": _i(base_n),
                "baseline_pos": _i(base_pos),
                "baseline_test_ap": _f(base.get("test_ap")),
                "fl_test_ap": float(fl_metrics["test_ap"]),
                "delta_test_ap": _delta(fl_metrics["test_ap"], base.get("test_ap")),
                "baseline_roc_auc": _f(base.get("test_roc_auc")),
                "fl_roc_auc": _f(fl_metrics["test_roc_auc"]),
                "delta_roc_auc": _delta(fl_metrics["test_roc_auc"], base.get("test_roc_auc")),
                "fl_n": int(test_n),
                "fl_pos": int(test_pos),
                "baseline_selected_alpha": _f(base.get("selected_alpha")),
                "baseline_selected_round": _i(base.get("selected_round")),
                "fl_selected_alpha": float(fl_info["alpha"]) if fl_info["alpha"] is not None else np.nan,
                "fl_selected_round": int(fl_info["round_id"]) if fl_info["round_id"] is not None else np.nan,
                "fl_selection_metric": fl_info["selection_metric"],
                "fl_selection_value": float(fl_info["selection_value"]) if fl_info["selection_value"] is not None else np.nan,
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
    parser.add_argument("--alpha", type=float, default=0.001, help="evaluate best model for this alpha (default: best overall)")
    parser.add_argument("--model_path", default=None, help="override FL model path (.npz)")
    parser.add_argument("--baseline_dir", default=None, help="baseline output dir (default: cfg.paths.out_local_baseline)")
    args = parser.parse_args()

    main(alpha=args.alpha, model_path=args.model_path, baseline_dir=args.baseline_dir)
