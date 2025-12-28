# 03_fl_train.py
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.config import load_config
from src.fl_protocol import load_params_npz, save_params_npz

cfg = load_config()
PY = [sys.executable]


def run(cmd, *, quiet: bool = True):
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


def aggregate_round_val_logloss_secure(client_out: Path, round_id: int, alpha: float, banks: list[str]) -> float:
    """
    Secure aggregation of validation weighted logloss (mean = Σsum_wloss / Σsum_w).
    Clients write masked components:
      - metric_num_masked = sum_wloss + mask
      - metric_den_masked = sum_w     + mask
    Server only sums masked values; masks cancel if all clients participate.
    """
    meta_files = sorted(client_out.glob(f"*/eval/round_{round_id:03d}_val_wlogloss.meta.json"))
    if not meta_files:
        raise RuntimeError(f"No evaluation meta files found for round {round_id:03d}")

    seen = set()
    total_num = 0.0
    total_den = 0.0

    for mf in meta_files:
        d = json.loads(mf.read_text(encoding="utf-8"))

        if int(d.get("round", -1)) != int(round_id):
            continue
        if "alpha" in d and d["alpha"] is not None:
            if abs(float(d["alpha"]) - float(alpha)) > 1e-12:
                continue

        bank = d.get("bank", mf.parts[-3])
        if bank in seen:
            continue
        seen.add(bank)

        if "metric_num_masked" not in d or "metric_den_masked" not in d:
            raise RuntimeError(f"Missing masked metric fields in {mf}")

        total_num += float(d["metric_num_masked"])
        total_den += float(d["metric_den_masked"])

    missing = [b for b in banks if b not in seen]
    if missing:
        raise RuntimeError(
            f"Secure aggregation requires all clients each round (no dropouts). Missing: {missing}"
        )

    if total_den <= 0:
        raise RuntimeError(f"Invalid aggregated denominator: total_den={total_den}")

    return total_num / total_den


def main():
    import shutil

    server_out = cfg.paths.out_fl_server
    if server_out.exists():
        shutil.rmtree(server_out)
    server_out.mkdir(parents=True, exist_ok=False)

    client_out = cfg.paths.out_fl_clients
    if client_out.exists():
        shutil.rmtree(client_out)
    client_out.mkdir(parents=True, exist_ok=False)

    best_global_path = server_out / "global_model_best.npz"

    # 1) client train-only stats
    for bank in cfg.banks.names:
        run(PY + [str(REPO_ROOT / "scripts" / "03a_fl_client_report_stats.py"), "--client", bank], quiet=True)

    # 2) server builds global plan
    run(PY + [str(REPO_ROOT / "scripts" / "03b_fl_server_build_global_plan.py")], quiet=True)

    # 3) clients build aligned datasets
    for bank in cfg.banks.names:
        run(
            PY
            + [
                str(REPO_ROOT / "scripts" / "03c_fl_client_initialize.py"),
                "--client",
                bank,
                "--overwrite",
            ],
            quiet=True,
        )

    alpha_grid = cfg.fl.alpha_grid
    patience = cfg.fl.patience
    num_rounds = cfg.fl.num_rounds

    # Track best across ALL alphas/rounds
    best_overall = {"alpha": None, "round_id": 0, "val_wlogloss": float("inf"), "model_path": ""}

    for alpha in alpha_grid:
        print(f"\n[alpha={alpha:.0e}]")
        no_improve = 0
        best_val_wlogloss = float("inf")
        best_round = 0

        # reset global model pointer for this alpha run
        delete_path = server_out / "global_model_latest.npz"
        if delete_path.exists():
            delete_path.unlink()

        metrics_dir = server_out / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / f"val_wlogloss_alpha{float(alpha):.6f}.jsonl"
        metrics_path.write_text("", encoding="utf-8")

        for round_id in range(1, num_rounds + 1):
            # 4) clients train one round and write updates
            for bank in cfg.banks.names:
                run(
                    PY
                    + [
                        str(REPO_ROOT / "scripts" / "03d_fl_client_train_round.py"),
                        "--client",
                        bank,
                        "--round_id",
                        round_id,
                        "--alpha",
                        alpha,
                    ],
                    quiet=True,
                )

            # 5) server aggregates model params (FedAvg)
            run(PY + [str(REPO_ROOT / "scripts" / "03e_fl_server_aggregate.py"), "--round_id", round_id], quiet=True)

            # 6) clients evaluate global model and write masked metrics
            for bank in cfg.banks.names:
                run(
                    PY
                    + [
                        str(REPO_ROOT / "scripts" / "03f_fl_evaluation.py"),
                        "--client",
                        bank,
                        "--round_id",
                        round_id,
                        "--alpha",
                        alpha,
                    ],
                    quiet=True,
                )

            # 7) federated evaluation: server aggregates ONLY masked metrics
            avg_val_wlogloss = aggregate_round_val_logloss_secure(
                client_out, round_id, alpha, banks=list(cfg.banks.names)
            )
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "alpha": float(alpha),
                            "round": int(round_id),
                            "avg_val_wlogloss": float(avg_val_wlogloss),
                            "avg_val_logloss": float(avg_val_wlogloss),  # backward compatibility
                        }
                    )
                    + "\n"
                )

            # 8) early stopping (within this alpha)
            is_new_best_alpha = avg_val_wlogloss < best_val_wlogloss
            if is_new_best_alpha:
                best_val_wlogloss = avg_val_wlogloss
                best_round = round_id
                no_improve = 0
            else:
                no_improve += 1

            # 9) overwrite ONLY ONE best global model file (across all alphas/rounds)
            is_new_best_overall = avg_val_wlogloss < best_overall["val_wlogloss"]
            if is_new_best_overall:
                global_model_path = server_out / "global_model_latest.npz"
                params = load_params_npz(global_model_path)

                save_params_npz(
                    best_global_path,
                    params,
                    meta={
                        "alpha": float(alpha),
                        "round_id": int(round_id),
                        "avg_val_wlogloss": float(avg_val_wlogloss),
                        "avg_val_logloss": float(avg_val_wlogloss),  # backward compatibility
                    },
                )
                best_overall = {
                    "alpha": float(alpha),
                    "round_id": int(round_id),
                    "val_wlogloss": float(avg_val_wlogloss),
                    "model_path": str(best_global_path),
                }


            tag_overall = " new_best" if is_new_best_overall else ""
            print(
                f"alpha={alpha:.0e} round={round_id:02d}/{num_rounds}: "
                f"avg_val_wlogloss={avg_val_wlogloss:.6f} "
                f"best_alpha={best_val_wlogloss:.6f} no_improve={no_improve}"
                f"{tag_overall}"
            )

            if no_improve >= patience:
                print(
                    f"[alpha={alpha:.0e}] Early stopping at round {round_id:02d} "
                    f"with best_val_wlogloss={best_val_wlogloss:.6f} at round {best_round:02d}"
                )
                break

        print(
            f"[alpha={alpha:.0e}] best_round={best_round:02d} best_val_wlogloss={best_val_wlogloss:.6f} "
            f"rounds_run={round_id:02d} early_stop={no_improve>=patience}"
        )

    if best_overall["alpha"] is None:
        raise RuntimeError("No best model was selected; check that evaluation produced valid metrics.")

    print(
        f"Best overall model: alpha={best_overall['alpha']:.6f}, "
        f"round={best_overall['round_id']}, val wlogloss={best_overall['val_wlogloss']:.6f}, "
        f"model path={best_overall['model_path']}"
    )
    print("Training completed.")


if __name__ == "__main__":
    main()
