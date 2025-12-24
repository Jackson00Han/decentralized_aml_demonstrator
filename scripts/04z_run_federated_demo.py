from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
from src.config import load_config
from src.fl_protocol import load_meta_json

PY = [sys.executable]


def run(cmd):
    disp = []
    for c in cmd:
        s = str(c)
        if s.startswith(str(REPO_ROOT)):
            disp.append(Path(s).relative_to(REPO_ROOT))
        else:
            disp.append(s)
    print("\n>>>", " ".join(map(str, disp)))
    subprocess.run(list(map(str, cmd)), check=True)

def load_state(path: Path) -> dict:
    if not path.exists():
        return {"best_val_ap": -1.0, "best_round": -1, "no_improve": 0}
    return json.loads(path.read_text(encoding="utf-8"))

def save_state(path: Path, st: dict) -> None:
    path.write_text(json.dumps(st, indent=2), encoding="utf-8")

def copy_model_with_meta(src: Path, dst: Path) -> None:
    shutil.copyfile(src, dst)
    src_meta = src.with_suffix(".meta.json")
    if src_meta.exists():
        shutil.copyfile(src_meta, dst.with_suffix(".meta.json"))

def compute_round_metrics(client_out: Path, round_id: int) -> tuple[float | None, int, list[dict]]:
    updates = []
    for upd in sorted(client_out.glob(f"*/updates/round_{round_id:03d}_update.npz")):
        meta = load_meta_json(upd)
        if not meta:
            raise RuntimeError(f"Missing update meta: {upd}")
        bank = meta.get("bank", upd.parts[-3])
        updates.append(
            {
                "bank": bank,
                "n_train": int(meta.get("n_train", 0)),
                "val_n": int(meta.get("val_n", 0)),
                "val_ap": float(meta.get("val_ap", 0.0)),
            }
        )
    if not updates:
        raise RuntimeError(f"No client updates found for round {round_id}")

    val_n_total = sum(u["val_n"] for u in updates)
    if val_n_total > 0:
        val_ap_w = sum(u["val_ap"] * u["val_n"] for u in updates) / val_n_total
    else:
        val_ap_w = None
    return val_ap_w, val_n_total, updates

def clean_round_artifacts(server_out: Path, client_out: Path) -> None:
    server_out.mkdir(parents=True, exist_ok=True)
    for name in [
        "global_model_latest.npz",
        "best_model.npz",
        "server_state.json",
        "round_logs.json",
        "summary.csv",
    ]:
        p = server_out / name
        if p.exists():
            p.unlink()
    shutil.rmtree(server_out / "global_models", ignore_errors=True)

    for upd in client_out.glob("*/updates"):
        shutil.rmtree(upd, ignore_errors=True)
    for metrics in client_out.glob("*/metrics_test.json"):
        if metrics.exists():
            metrics.unlink()


def run_stats_and_plan(banks: list[str], client_out: Path, server_out: Path, force: bool) -> None:
    stats_missing = any(not (client_out / b / "stats.json").exists() for b in banks)
    plan_missing = not (server_out / "global_plan.json").exists()
    if force or stats_missing:
        for b in banks:
            run(PY + [str(REPO_ROOT / "scripts/04a_client_report_stats.py"), "--client", b])
    if force or plan_missing:
        run(PY + [str(REPO_ROOT / "scripts/04b_server_build_global_plan.py")])


def run_rounds(
    banks: list[str],
    num_rounds: int,
    server_out: Path,
    client_out: Path,
    patience: int,
    alpha: float | None,
    use_trainval: bool,
    early_stop: bool,
) -> dict | None:
    state_path = server_out / "server_state.json"
    log_path = server_out / "round_logs.json"
    st = load_state(state_path)
    logs = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []

    for r in range(1, num_rounds + 1):
        for b in banks:
            cmd = [
                str(REPO_ROOT / "scripts/04d_client_one_round_update.py"),
                "--client",
                b,
                "--round_id",
                str(r),
            ]
            if alpha is not None:
                cmd += ["--alpha", str(alpha)]
            if use_trainval:
                cmd += ["--use_trainval"]
            run(PY + cmd)
        run(PY + [str(REPO_ROOT / "scripts/04e_server_aggregate_round.py"), "--round_id", str(r)])

        val_ap_w, val_n_total, client_rows = compute_round_metrics(client_out, r)
        logs.append(
            {
                "round": r,
                "val_ap_weighted": val_ap_w,
                "val_n_total": val_n_total,
                "clients": client_rows,
            }
        )
        log_path.write_text(json.dumps(logs, indent=2), encoding="utf-8")

        if early_stop:
            if val_ap_w is None:
                print(f"[WARN] round {r:03d}: val_n_total=0, early stop skipped.")
                continue

            improved = val_ap_w > st["best_val_ap"] + 1e-9
            if improved:
                st["best_val_ap"] = float(val_ap_w)
                st["best_round"] = int(r)
                st["no_improve"] = 0
                copy_model_with_meta(
                    server_out / "global_model_latest.npz",
                    server_out / "best_model.npz",
                )
            else:
                st["no_improve"] += 1

            save_state(state_path, st)
            print(
                f"[Round {r:03d}] val_ap_weighted={val_ap_w:.6f} | "
                f"best={st['best_val_ap']:.6f} @r{st['best_round']} | "
                f"no_improve={st['no_improve']}"
            )

            if st["no_improve"] >= patience:
                best_model = server_out / "best_model.npz"
                print(
                    f"EARLY STOP: no_improve={st['no_improve']} >= patience={patience} | "
                    f"best_round={st['best_round']} best_val_ap={st['best_val_ap']:.6f} | "
                    f"model={best_model}"
                )
                break

    return st


def evaluate_and_summary(
    banks: list[str],
    client_out: Path,
    server_out: Path,
    use_latest: bool,
) -> None:
    for b in banks:
        cmd = [str(REPO_ROOT / "scripts/04f_client_eval_best.py"), "--client", b]
        if use_latest:
            cmd += ["--use_latest"]
        run(PY + cmd)

    summary_rows = []
    for b in banks:
        metrics_path = client_out / b / "metrics_test.json"
        if not metrics_path.exists():
            print(f"[WARN] missing metrics: {metrics_path}")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        topk = metrics.get("topk", {})
        k = topk.get("k")
        summary_rows.append(
            {
                "bank": b,
                "test_ap": metrics.get("test_ap"),
                "roc_auc": metrics.get("test_roc_auc"),
                "pos": topk.get("pos"),
                "n": topk.get("n"),
                f"hits@{k}": topk.get("hits"),
                f"expected_hits_random@{k}": topk.get("expected_hits_random"),
                f"P@{k}": topk.get("precision_at_k"),
                f"base_P@{k}": topk.get("baseline_precision_at_k"),
                f"R@{k}": topk.get("recall_at_k"),
                f"base_R@{k}": topk.get("baseline_recall_at_k"),
                f"liftP@{k}": topk.get("lift_precision_at_k"),
            }
        )

    print("\n" + "=" * 68)
    print("FL Summary")
    print("=" * 68)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("bank")
        out_csv = server_out / "summary.csv"
        summary_df.to_csv(out_csv, index=False)
        print(summary_df.to_string(index=False))
        print(f"Saved: {out_csv}")
    else:
        print("No metrics found; summary skipped.")


def main(clean: bool = True, sweep_alpha: bool = False):
    cfg = load_config()
    banks = cfg.banks.names
    num_rounds = cfg.fl.num_rounds
    server_out = cfg.paths.out_fl_server
    client_out = cfg.paths.out_fl_clients
    patience = cfg.fl.patience

    if clean:
        shutil.rmtree(server_out, ignore_errors=True)

    run_stats_and_plan(banks, client_out, server_out, force=clean)

    if not sweep_alpha:
        clean_round_artifacts(server_out, client_out)
        run_rounds(
            banks=banks,
            num_rounds=num_rounds,
            server_out=server_out,
            client_out=client_out,
            patience=patience,
            alpha=None,
            use_trainval=False,
            early_stop=True,
        )
        evaluate_and_summary(banks, client_out, server_out, use_latest=False)
        return

    alpha_grid = cfg.fl.alpha_grid
    if not alpha_grid:
        print("No alpha_grid specified for sweep; exiting.")
        return

    results = []
    for alpha in alpha_grid:
        print(f"\n=== Sweep alpha={alpha:g} ===")
        clean_round_artifacts(server_out, client_out)
        st = run_rounds(
            banks=banks,
            num_rounds=num_rounds,
            server_out=server_out,
            client_out=client_out,
            patience=patience,
            alpha=alpha,
            use_trainval=False,
            early_stop=True,
        )
        best_val_ap = st.get("best_val_ap", -1.0) if st else -1.0
        best_round = st.get("best_round", -1) if st else -1
        results.append((alpha, best_val_ap, best_round))
        print(f"[alpha={alpha:g}] best_val_ap={best_val_ap:.6f} best_round={best_round}")

    best_alpha, best_val_ap, best_round = max(results, key=lambda x: x[1])
    if best_round <= 0:
        best_round = num_rounds
    print(f"\n[alpha_sweep] selected alpha={best_alpha:g} (best_val_ap={best_val_ap:.6f}, best_round={best_round})")

    clean_round_artifacts(server_out, client_out)
    run_rounds(
        banks=banks,
        num_rounds=best_round,
        server_out=server_out,
        client_out=client_out,
        patience=patience,
        alpha=best_alpha,
        use_trainval=True,
        early_stop=False,
    )
    evaluate_and_summary(banks, client_out, server_out, use_latest=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_clean", action="store_true", help="skip cleaning server output")
    parser.add_argument("--sweep_alpha", action="store_true", help="sweep fl.alpha_grid and rerun best")
    args = parser.parse_args()
    main(clean=not args.no_clean, sweep_alpha=args.sweep_alpha)
