from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.fl_protocol import (
    FeatureSchema,
    GlobalPlan,
    save_params_npz,
    load_params_npz,
    load_meta_json,
)

def main():
    out = Path("outputs/_debug_protocol")
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1) Test GlobalPlan JSON roundtrip ----
    plan = GlobalPlan(
        schema_version="v1",
        feature_schema=FeatureSchema(
            cat_cols=["orig_state", "bene_state"],
            num_cols=["base_amt", "orig_initial_deposit", "bene_initial_deposit"],
        ),
        global_categories={
            "orig_state": ["CA", "NY"],
            "bene_state": ["TX", "CA"],
        },
        global_numeric={
            "base_amt": {"mean": 10.0, "std": 2.0},
            "orig_initial_deposit": {"mean": 100.0, "std": 50.0},
            "bene_initial_deposit": {"mean": 120.0, "std": 60.0},
        },
    )

    plan_path = out / "global_plan.json"
    plan.dump(plan_path)
    plan2 = GlobalPlan.load(plan_path)

    print("== GlobalPlan roundtrip ==")
    print("schema_version:", plan2.schema_version)
    print("cat_cols:", plan2.feature_schema.cat_cols)
    print("num_cols:", plan2.feature_schema.num_cols)
    print("global_categories:", plan2.global_categories)
    print("global_numeric:", plan2.global_numeric)
    print()

    # ---- 2) Test saving/loading model params ----
    # Example: logistic regression params (coef/intercept)
    d = 5
    params = {
        "coef": np.random.randn(1, d).astype(np.float64),
        "intercept": np.random.randn(1).astype(np.float64),
    }
    meta = {
        "round": 3,
        "val_ap_weighted": 0.0042,
        "schema_version": "v1",
    }

    model_path = out / "global_model_latest.npz"
    save_params_npz(model_path, params, meta=meta)

    params2 = load_params_npz(model_path)
    meta2 = load_meta_json(model_path)

    print("== Params NPZ roundtrip ==")
    print("Saved keys:", list(params.keys()))
    print("Loaded keys:", list(params2.keys()))
    print("coef shape:", params2["coef"].shape)
    print("intercept shape:", params2["intercept"].shape)
    print("meta:", meta2)

    # sanity check numeric equality
    print("\n== Equality checks ==")
    print("coef equal:", np.allclose(params["coef"], params2["coef"]))
    print("intercept equal:", np.allclose(params["intercept"], params2["intercept"]))

if __name__ == "__main__":
    main()
