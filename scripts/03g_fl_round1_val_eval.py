# 04g_fl_round1_val_eval.py
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.config import load_config

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


def aggregate_round_val_logloss_secure(
    client_out: Path,
    round_id: int,
    alpha: float,
    banks: list[str],
    *,
    num_key: str = "metric_num_masked",
    den_key: str = "metric_den_masked",
) -> float:
    """
    Server-side secure aggregation of validation logloss (weighted mean by val_n).
    Requires each client to have run scripts/03f_fl_evaluation.py and written:
      outputs/fl_clients/{bank}/eval/round_XXX_val_logloss.meta.json
    """
    meta_files = sorted(client_out.glob(f"*/eval/round_{round_id:03d}_val_logloss.meta.json"))
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

        if num_key not in d or den_key not in d:
            raise RuntimeError(f"Missing masked metric fields ({num_key}, {den_key}) in {mf}")

        total_num += float(d[num_key])
        total_den += float(d[den_key])

    missing = [b for b in banks if b not in seen]
    if missing:
        raise RuntimeError(
            f"Secure aggregation requires all clients each round (no dropouts). Missing: {missing}"
        )

    if total_den <= 0:
        raise RuntimeError(f"Invalid aggregated denominator: total_den={total_den}")

    return total_num / total_den


def main(round_id: int = 1, alpha: float | None = None, verbose: bool = False, from_scratch: bool = False) -> None:
    server_out = cfg.paths.out_fl_server
    client_out = cfg.paths.out_fl_clients
    banks = list(cfg.banks.names)

    alpha = float(alpha) if alpha is not None else float(cfg.fl.alpha)

    quiet = not verbose

    # This script is primarily a convenience wrapper for running evaluation AFTER aggregation (03e),
    # without re-implementing metric logic (it calls 03f).
    if from_scratch:
        if int(round_id) != 1:
            raise ValueError(
                f"--from_scratch is intended for a round-1 smoke test; got round_id={round_id}. "
                "For later rounds, run scripts/03_fl_train.py or run 03d/03e/03f for each round."
            )
        # Ensure this is "round 1 from scratch" unless caller intentionally left artifacts.
        global_latest = server_out / "global_model_latest.npz"
        if global_latest.exists():
            global_latest.unlink()

        # 1) client train-only stats
        print("[1/6] Client train-only stats (03a)")
        for bank in banks:
            run(PY + [str(REPO_ROOT / "scripts" / "03a_fl_client_report_stats.py"), "--client", bank], quiet=quiet)

        # 2) server builds global plan
        print("[2/6] Server builds global plan (03b)")
        run(PY + [str(REPO_ROOT / "scripts" / "03b_fl_server_build_global_plan.py")], quiet=quiet)

        # 3) clients build aligned datasets
        print("[3/6] Clients build aligned datasets (03c)")
        for bank in banks:
            run(
                PY
                + [
                    str(REPO_ROOT / "scripts" / "03c_fl_client_initialize.py"),
                    "--client",
                    bank,
                    "--overwrite",
                ],
                quiet=quiet,
            )

        # 4) clients train one round (local updates)
        print(f"[4/6] Clients train one round (03d) | round_id={round_id} alpha={alpha:g}")
        for bank in banks:
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
                quiet=quiet,
            )

        # 5) server aggregates (FedAvg) -> global_model_latest.npz
        print("[5/6] Server aggregates to global model (03e)")
        run(PY + [str(REPO_ROOT / "scripts" / "03e_fl_server_aggregate.py"), "--round_id", round_id], quiet=quiet)

    # client evaluation (03f) + server aggregates masked metrics
    step = "[6/6]" if from_scratch else "[1/1]"
    print(f"{step} Client val eval (03f) + aggregate masked metrics (03g)")
    for bank in banks:
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
            quiet=quiet,
        )

    avg_val_logloss_global = aggregate_round_val_logloss_secure(
        client_out, round_id, alpha, banks=list(cfg.banks.names), num_key="metric_num_masked", den_key="metric_den_masked"
    )
    avg_val_logloss_local = aggregate_round_val_logloss_secure(
        client_out,
        round_id,
        alpha,
        banks=list(cfg.banks.names),
        num_key="local_metric_num_masked",
        den_key="local_metric_den_masked",
    )
    print(
        f"\nround={round_id:03d} alpha={alpha:g} "
        f"avg_local_val_logloss={avg_val_logloss_local:.6f} "
        f"avg_global_val_logloss={avg_val_logloss_global:.6f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--round_id", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=None, help="override cfg.fl.alpha for this run")
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="run 03a/03b/03c/03d/03e first, then evaluate (useful for a one-shot round-1 smoke test)",
    )
    parser.add_argument("--verbose", action="store_true", help="print subcommands and their output")
    args = parser.parse_args()

    main(round_id=args.round_id, alpha=args.alpha, verbose=args.verbose, from_scratch=bool(args.from_scratch))
