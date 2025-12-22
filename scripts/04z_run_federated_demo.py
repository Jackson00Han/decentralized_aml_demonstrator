from __future__ import annotations

import shutil

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = ["python"]

BANKS = ["bank_a", "bank_b", "bank_c"]
NUM_ROUNDS = 2


def run(cmd):
    print("\n>>>", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True)



def main(clean: bool = True):
    if clean: # remove previous outputs, making sure to start fresh
        shutil.rmtree(REPO_ROOT / "outputs" / "fl_server", ignore_errors=True)

    # clients report stats
    for b in BANKS:
        run(PY + ["scripts/04a_client_report_stats.py", "--client", b])

    # server builds global plan and initializes by absence of global_model_latest
    run(PY + ["scripts/04b_server_build_global_plan.py"])

    # rounds
    for r in range(1, NUM_ROUNDS + 1):
        for b in BANKS:
            run(PY + ["scripts/04c_client_train_round.py", "--client", b, "--round", str(r), "--local_epochs", "2"])
        run(PY + ["scripts/04d_server_aggregate_round.py", "--round", str(r)])

    # evaluate best
    for b in BANKS:
        run(PY + ["scripts/04e_client_eval_best.py", "--client", b])


if __name__ == "__main__":
    main()
