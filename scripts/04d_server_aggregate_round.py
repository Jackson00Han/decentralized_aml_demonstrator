from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.fl_protocol import save_params_npz, load_params_npz, load_meta_json
from src.fl_aggregators import ClientUpdate, fedavg

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_OUT = REPO_ROOT / "outputs" / "fl_server"
CLIENT_OUT = REPO_ROOT / "outputs" / "fl_clients"

PATIENCE = 3


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"best_val_ap": -1.0, "best_round": -1, "no_improve": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, st: dict) -> None:
    path.write_text(json.dumps(st, indent=2), encoding="utf-8")


def main(round_id: int):
    SERVER_OUT.mkdir(parents=True, exist_ok=True)
    state_path = SERVER_OUT / "server_state.json"
    st = load_state(state_path)

    updates = []
    # gather all banks updates for this round
    for upd in sorted(CLIENT_OUT.glob(f"*/updates/round_{round_id:03d}_update.npz")):
        bank = upd.parts[-3]  # .../fl_clients/{bank}/updates/...
        meta = load_meta_json(upd) or {}
        params = load_params_npz(upd)
        updates.append(
            ClientUpdate(
                bank=bank,
                n_train=int(meta.get("n_train", 1)),
                params=params,
                metrics={"val_ap": float(meta.get("val_ap", 0.0)), "val_n": float(meta.get("val_n", 1.0))},
            )
        )

    if not updates:
        raise RuntimeError(f"No client updates found for round {round_id}")

    # FedAvg aggregate
    new_params = fedavg(updates)

    # weighted val AP for monitoring/early stop
    tot = sum(u.metrics["val_n"] for u in updates)
    val_ap_w = sum(u.metrics["val_ap"] * u.metrics["val_n"] for u in updates) / tot if tot > 0 else 0.0

    # save global model for this round + latest pointer
    save_params_npz(
        SERVER_OUT / "global_models" / f"round_{round_id:03d}.npz",
        new_params,
        meta={"round": round_id, "val_ap_weighted": float(val_ap_w)},
    )
    save_params_npz(
        SERVER_OUT / "global_model_latest.npz",
        new_params,
        meta={"round": round_id, "val_ap_weighted": float(val_ap_w)},
    )

    # update best / patience
    improved = val_ap_w > st["best_val_ap"] + 1e-9
    if improved:
        st["best_val_ap"] = float(val_ap_w)
        st["best_round"] = int(round_id)
        st["no_improve"] = 0
        # snapshot best model
        save_params_npz(
            SERVER_OUT / "best_model.npz",
            new_params,
            meta={"best_round": round_id, "best_val_ap_weighted": float(val_ap_w)},
        )
    else:
        st["no_improve"] += 1

    save_state(state_path, st)

    # append round log
    log_path = SERVER_OUT / "round_logs.json"
    logs = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    logs.append(
        {
            "round": round_id,
            "val_ap_weighted": float(val_ap_w),
            "clients": [{"bank": u.bank, "n_train": u.n_train, **u.metrics} for u in updates],
        }
    )
    log_path.write_text(json.dumps(logs, indent=2), encoding="utf-8")

    print(f"[Round {round_id:03d}] val_ap_weighted={val_ap_w:.6f} | best={st['best_val_ap']:.6f} @r{st['best_round']} | no_improve={st['no_improve']}")

    if st["no_improve"] >= PATIENCE:
        print(f"EARLY STOP suggested (patience={PATIENCE}).")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()
    main(args.round)
