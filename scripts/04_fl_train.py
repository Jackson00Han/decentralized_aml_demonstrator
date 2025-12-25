# 04_fl_train.py
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

def aggregate_round_val_ap_secure(client_out: Path, round_id: int, alpha: float, banks: list[str]) -> float:
    """
    Secure aggregation of validation AP:
      Each client writes masked components:
        - val_num_masked = (val_ap * val_n) + mask
        - val_den_masked = (val_n) + mask
      Server only sums masked values; masks cancel if all clients participate.
    """
    meta_files = sorted(client_out.glob(f"*/updates/round_{round_id:03d}_update.meta.json"))
    if not meta_files:
        raise RuntimeError(f"No update meta files found for round {round_id:03d}")

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

        if "val_num_masked" not in d or "val_den_masked" not in d:
            raise RuntimeError(f"Missing masked metric fields in {mf}")

        total_num += float(d["val_num_masked"])
        total_den += float(d["val_den_masked"])

    missing = [b for b in banks if b not in seen]
    if missing:
        raise RuntimeError(
            f"Secure aggregation requires all clients each round (no dropouts). Missing: {missing}"
        )

    if total_den <= 0:
        raise RuntimeError(f"Invalid aggregated denominator: total_den={total_den}")

    return total_num / total_den



def main():
    server_out = cfg.paths.out_fl_server
    client_out = cfg.paths.out_fl_clients

    # 1) client train-only stats
    for bank in cfg.banks.names:
        run(PY + [str(REPO_ROOT / "scripts" / "04a_fl_client_report_stats.py"), "--client", bank], quiet=True)

    # 2) server builds global plan
    run(PY + [str(REPO_ROOT / "scripts" / "04b_fl_server_build_global_plan.py")], quiet=True)

    # 3) clients build aligned datasets
    for bank in cfg.banks.names:
        run(
            PY
            + [
                str(REPO_ROOT / "scripts" / "04c_fl_client_initialize.py"),
                "--client",
                bank,
                "--overwrite",
            ],
            quiet=True
        )

    alpha_grid = cfg.fl.alpha_grid
    patience = cfg.fl.patience
    num_rounds = cfg.fl.num_rounds

    best_overall = {"alpha": None, "round_id": 0, "val_ap": -float("inf"), "model_path": ""}

    for alpha in alpha_grid:
        print(f"\n[alpha={alpha:.0e}]")
        no_improve = 0
        best_val_ap = -float("inf")
        best_round = 0

        # reset global model pointer
        delete_path = server_out / "global_model_latest.npz"
        if delete_path.exists():
            delete_path.unlink()
        best_alpha_model = {"alpha": alpha, "round_id": 0, "val_ap": -float("inf"), "model_path": ""}
        for round_id in range(1, num_rounds + 1):
            # 4) clients train one round and write updates + metric meta
            for bank in cfg.banks.names:
                run(
                    PY +
                    [
                        str(REPO_ROOT / "scripts" / "04d_fl_client_train_round.py"),
                        "--client", bank,
                        "--round_id", round_id,
                        "--alpha", alpha,
                    ],
                    quiet=True,
                )

            # 5) server aggregates model params (FedAvg)
            run(PY + [str(REPO_ROOT / "scripts" / "04e_fl_server_aggregate.py"), "--round_id", round_id], quiet=True)

            # 6) federated evaluation: server aggregates ONLY metrics from meta.json
            avg_val_ap = aggregate_round_val_ap_secure(client_out, round_id, alpha, banks=list(cfg.banks.names))

            # 7) early stopping
            is_new_best = avg_val_ap > best_val_ap
            if is_new_best:
                best_val_ap = avg_val_ap
                best_round = round_id
                no_improve = 0

                # save best model for this alpha
                global_model_path = server_out / "global_model_latest.npz"
                params = load_params_npz(global_model_path)

                best_model_path = server_out / f"global_model_best_alpha{alpha:.6f}.npz"
                save_params_npz(
                    best_model_path,
                    params,
                    meta={
                        "alpha": float(alpha),
                        "round_id": int(round_id),
                        "avg_val_ap": float(best_val_ap),
                    },
                )
                best_alpha_model.update(
                    {"alpha": alpha, "round_id": round_id, "val_ap": best_val_ap, "model_path": best_model_path}
                )
            else:
                no_improve += 1

            tag = " new_best" if is_new_best else ""
            print(
                f"alpha={alpha:.0e} round={round_id:02d}/{num_rounds} "
                f"avg_val_ap={avg_val_ap:.6f} best={best_val_ap:.6f} no_improve={no_improve}{tag}"
            )

            if no_improve >= patience:
                print(
                    f"[alpha={alpha:.0e}] early_stop round={round_id:02d} best_val_ap={best_val_ap:.6f}"
                )
                break

        print(
            f"[alpha={alpha:.0e}] best_round={best_round:02d} best_val_ap={best_val_ap:.6f} "
            f"rounds_run={round_id:02d} early_stop={no_improve>=patience}"
        )
        if best_alpha_model["val_ap"] > best_overall["val_ap"]:
            best_overall = best_alpha_model

    print(
        f"Best overall model: alpha={best_overall['alpha']:.6f}, "
        f"round={best_overall['round_id']}, val AP={best_overall['val_ap']:.6f}, "
        f"model path={best_overall['model_path']}"
    )
    print("Training completed.")


if __name__ == "__main__":
    main()
