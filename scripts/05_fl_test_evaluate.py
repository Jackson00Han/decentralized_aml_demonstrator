from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.config import load_config
from src.fl_adapters import SkLogRegSGD
from src.fl_protocol import load_params_npz, save_params_npz
from src.metrics import ap, safe_roc_auc, topk_report
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


def find_best_model_meta(server_out: Path) -> dict:
    meta_files = sorted(server_out.glob("global_model_best_alpha*.meta.json"))
    if not meta_files:
        raise RuntimeError(
            f"No best-alpha meta files under {server_out}. Run scripts/04_fl_train.py first."
        )

    best = None
    for mf in meta_files:
        d = json.loads(mf.read_text(encoding="utf-8"))
        if "alpha" not in d or "round_id" not in d:
            continue
        val_ap = float(d.get("avg_val_ap", -float("inf")))
        cand = {
            "alpha": float(d["alpha"]),
            "round_id": int(d["round_id"]),
            "val_ap": val_ap,
            "meta_path": mf,
        }
        if best is None or cand["val_ap"] > best["val_ap"]:
            best = cand

    if best is None:
        raise RuntimeError(f"No usable meta found under {server_out} (missing alpha/round_id).")
    return best


def find_dataset_dir(bank: str, client_out: Path, expected_plan_hash: str) -> Path:
    base = client_out / bank / "datasets"
    if not base.exists():
        raise FileNotFoundError(f"Missing datasets dir: {base} (run 04c_fl_client_initialize.py)")

    prefix = f"plan_{expected_plan_hash[:8]}"
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(
            f"No dataset matching {prefix} under {base} (run 04c_fl_client_initialize.py)"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def fmt(x: float | None, nd: int = 6) -> str:
    if x is None:
        return "NA"
    return f"{float(x):.{nd}f}"


def main(alpha: float | None = None, round_id: int | None = None) -> None:
    server_out = cfg.paths.out_fl_server
    client_out = cfg.paths.out_fl_clients

    if alpha is None or round_id is None:
        best = find_best_model_meta(server_out)
        best_alpha = best["alpha"]
        best_round = best["round_id"]
        best_val_ap = best["val_ap"]
    else:
        best_alpha = float(alpha)
        best_round = int(round_id)
        best_val_ap = None

    if best_round <= 0:
        raise ValueError(f"Invalid best_round={best_round} (must be >=1)")

    print(f"Using best params: alpha={best_alpha:g}, round_id={best_round}")

    # 1) client train-only stats
    for bank in cfg.banks.names:
        run(PY + [str(REPO_ROOT / "scripts" / "04a_fl_client_report_stats.py"), "--client", bank], quiet=True)

    # 2) server builds global plan
    run(PY + [str(REPO_ROOT / "scripts" / "04b_fl_server_build_global_plan.py")], quiet=True)

    # 3) clients build aligned datasets (train/val/test + trainval)
    for bank in cfg.banks.names:
        run(
            PY
            + [
                str(REPO_ROOT / "scripts" / "04c_fl_client_initialize.py"),
                "--client",
                bank,
                "--overwrite",
            ],
            quiet=True,
        )

    # reset global model pointer
    delete_path = server_out / "global_model_latest.npz"
    if delete_path.exists():
        delete_path.unlink()

    # 4) train for best_round using train+val (no early stopping)
    for r in range(1, best_round + 1):
        for bank in cfg.banks.names:
            run(
                PY
                + [
                    str(REPO_ROOT / "scripts" / "04d_fl_client_train_round.py"),
                    "--client",
                    bank,
                    "--round_id",
                    r,
                    "--alpha",
                    best_alpha,
                    "--use_trainval",
                ],
                quiet=True,
            )
        run(PY + [str(REPO_ROOT / "scripts" / "04e_fl_server_aggregate.py"), "--round_id", r], quiet=True)

    # 5) save final model
    final_model_path = server_out / f"global_model_final_alpha{best_alpha:.6f}_round{best_round:03d}.npz"
    params = load_params_npz(server_out / "global_model_latest.npz")
    save_params_npz(
        final_model_path,
        params,
        meta={
            "alpha": float(best_alpha),
            "round_id": int(best_round),
            "val_ap_best": float(best_val_ap) if best_val_ap is not None else None,
        },
    )
    print(f"Saved final model: {final_model_path}")

    # 6) evaluate on test sets
    plan_path = server_out / "global_plan.json"
    expected_hash = plan_hash(plan_path)
    top_k = cfg.baseline.top_k
    seed = cfg.project.seed

    out_root = server_out / "test_eval"
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for bank in cfg.banks.names:
        ds_dir = find_dataset_dir(bank, client_out, expected_hash)
        data = load_dataset(ds_dir, expected_hash)
        X_test = data["X_test"]
        y_test = data["y_test"].astype(int)

        model = SkLogRegSGD(d=X_test.shape[1], alpha=best_alpha, seed=seed)
        model.set_params(params)
        test_scores = model.predict_scores(X_test)

        test_ap = float(ap(y_test, test_scores)) if y_test.sum() > 0 else 0.0
        test_auc = safe_roc_auc(y_test, test_scores)
        rep = topk_report(y_test, test_scores, k=top_k)

        print("\n" + "=" * 68)
        print(f"FL test | {bank}")
        print("=" * 68)
        print(f"Test AP: {fmt(test_ap)} | Test ROC AUC: {fmt(test_auc)}")
        print("Top-k report:")
        print(json.dumps(rep, indent=2, sort_keys=True))

        bank_out = out_root / bank
        bank_out.mkdir(parents=True, exist_ok=True)
        metrics = {
            "bank": bank,
            "selected_alpha": best_alpha,
            "selected_round": best_round,
            "val_ap_best": best_val_ap,
            "test_ap": test_ap,
            "test_roc_auc": test_auc,
            "topk": rep,
        }
        out_json = bank_out / "test_report.json"
        out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved: {out_json}")

        summary_rows.append(
            {
                "bank": bank,
                "test_ap": test_ap,
                "roc_auc": test_auc if test_auc is not None else np.nan,
                "pos": rep["pos"],
                "n": rep["n"],
                f"hits@{rep['k']}": rep["hits"],
                f"expected_hits_random@{rep['k']}": rep["expected_hits_random"],
                f"P@{rep['k']}": rep["precision_at_k"],
                f"base_P@{rep['k']}": rep["baseline_precision_at_k"],
                f"R@{rep['k']}": rep["recall_at_k"],
                f"base_R@{rep['k']}": rep["baseline_recall_at_k"],
                f"liftP@{rep['k']}": rep["lift_precision_at_k"] if rep["lift_precision_at_k"] is not None else np.nan,
                "selected_alpha": best_alpha,
                "selected_round": best_round,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("bank")
    out_csv = out_root / "summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print("\n" + "=" * 68)
    print("Summary")
    print("=" * 68)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {out_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=None, help="override best alpha from 04_fl_train")
    parser.add_argument("--round_id", type=int, default=None, help="override best round from 04_fl_train")
    args = parser.parse_args()

    if (args.alpha is None) ^ (args.round_id is None):
        raise SystemExit("Provide both --alpha and --round_id, or neither to use best from 04_fl_train.")

    main(alpha=args.alpha, round_id=args.round_id)
