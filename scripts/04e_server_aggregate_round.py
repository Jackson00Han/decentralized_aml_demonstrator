from __future__ import annotations

from pathlib import Path


def main(round_id: int):
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))
    from src.fl_protocol import save_params_npz, load_params_npz, load_meta_json
    from src.fl_aggregators import ClientUpdate, fedavg
    from src.config import load_config
    cfg = load_config()
    SERVER_OUT = cfg.paths.out_fl_server
    CLIENT_OUT = cfg.paths.out_fl_clients

    SERVER_OUT.mkdir(parents=True, exist_ok=True)

    updates = []
    # gather all banks updates for this round
    for upd in sorted(CLIENT_OUT.glob(f"*/updates/round_{round_id:03d}_update.npz")):
        meta = load_meta_json(upd)
        if not meta or "n_train" not in meta:
            raise RuntimeError(f"Missing update meta or n_train: {upd}")
        params = load_params_npz(upd)
        bank = meta.get("bank", upd.parts[-3])  # .../fl_clients/{bank}/updates/...
        updates.append(
            ClientUpdate(
                bank=bank,
                n_train=int(meta["n_train"]),
                params=params,
                metrics={},
            )
        )

    if not updates:
        raise RuntimeError(f"No client updates found for round {round_id}")

    # FedAvg aggregate (weighted by n_train)
    new_params = fedavg(updates)

    # save global model for this round + latest pointer
    save_params_npz(
        SERVER_OUT / "global_models" / f"round_{round_id:03d}.npz",
        new_params,
        meta={"round": round_id},
    )
    save_params_npz(
        SERVER_OUT / "global_model_latest.npz",
        new_params,
        meta={"round": round_id},
    )

    total_train = sum(u.n_train for u in updates)
    print(f"[Round {round_id:03d}] aggregated {len(updates)} clients | total_train={total_train}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--round_id", type=int, default=1)
    args = parser.parse_args()
    main(args.round_id)
