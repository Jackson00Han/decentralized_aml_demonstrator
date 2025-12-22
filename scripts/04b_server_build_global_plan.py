from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.fl_protocol import GlobalPlan, FeatureSchema

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_OUT = REPO_ROOT / "outputs" / "fl_server"
CLIENT_OUT = REPO_ROOT / "outputs" / "fl_clients"

CAT_COLS = ["orig_state", "bene_state"]
NUM_COLS = ["base_amt", "orig_initial_deposit", "bene_initial_deposit"]

SCHEMA_VERSION = "v1"


def main():
    SERVER_OUT.mkdir(parents=True, exist_ok=True)

    # collect stats
    stats_files = sorted(CLIENT_OUT.glob("*/stats.json"))
    if not stats_files:
        raise RuntimeError("No client stats found. Run 04a_client_report_stats.py first.")

    global_cats = {c: set() for c in CAT_COLS}
    agg = {c: {"n": 0, "sum": 0.0, "sumsq": 0.0} for c in NUM_COLS}

    for f in stats_files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for c in CAT_COLS:
            global_cats[c].update(d["cat_sets"][c])
        for c in NUM_COLS:
            st = d["num_stats"][c]
            agg[c]["n"] += int(st["n"])
            agg[c]["sum"] += float(st["sum"])
            agg[c]["sumsq"] += float(st["sumsq"])

    global_categories = {c: sorted(list(v)) for c, v in global_cats.items()}

    global_numeric = {}
    for c, st in agg.items():
        n = st["n"]
        if n <= 1:
            mu, sd = 0.0, 1.0
        else:
            mu = st["sum"] / n
            var = max(st["sumsq"] / n - mu * mu, 0.0)
            sd = float(np.sqrt(var)) if var > 0 else 1.0
        global_numeric[c] = {"mean": float(mu), "std": float(sd)}

    plan = GlobalPlan(
        schema_version=SCHEMA_VERSION,
        feature_schema=FeatureSchema(cat_cols=CAT_COLS, num_cols=NUM_COLS),
        global_categories=global_categories,
        global_numeric=global_numeric,
    )
    plan.dump(SERVER_OUT / "global_plan.json")
    print(f"[OK] wrote {SERVER_OUT/'global_plan.json'}")


if __name__ == "__main__":
    main()
