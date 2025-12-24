from __future__ import annotations
import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.fl_cache import load_or_build_cache
from src.fl_protocol import GlobalPlan, load_params_npz
from src.fl_preprocess import build_preprocessor
from src.fl_adapters import SkLogRegSGD
from src.metrics import ap, safe_roc_auc, topk_report


def main(bank: str, use_best: bool = True):
    cfg = load_config()
    data_processed = cfg.paths.data_processed
    client_out = cfg.paths.out_fl_clients
    server_out = cfg.paths.out_fl_server
    top_k = cfg.baseline.top_k

    plan_path = server_out / "global_plan.json"
    plan = GlobalPlan.load(plan_path)
    preprocess = build_preprocessor(plan)

    model_path = server_out / ("best_model.npz" if use_best else "global_model_latest.npz")
    params = load_params_npz(model_path)

    cache = load_or_build_cache(
        bank=bank,
        data_processed=data_processed,
        client_out=client_out,
        plan_path=plan_path,
        plan=plan,
        preprocess=preprocess,
        ts_col="tran_timestamp",
    )
    X_te = cache["X_test"]
    y_te = cache["y_test"].astype(int)

    d = X_te.shape[1]
    model = SkLogRegSGD(d=d)
    model.set_params(params)
    scores = model.predict_scores(X_te)

    res = {
        "bank": bank,
        "which_model": "best_model" if use_best else "latest_model",
        "test_n": int(len(y_te)),
        "test_pos": int(y_te.sum()),
        "test_ap": ap(y_te, scores),
        "test_roc_auc": safe_roc_auc(y_te, scores),
        "topk": topk_report(y_te, scores, top_k)
    }

    out_dir = client_out / bank
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_test.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir/'metrics_test.json'} | test_ap={res['test_ap']:.6f}")


if __name__ == "__main__":
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument("--client", default="bank_a")
    parse.add_argument("--use_best", action="store_true", help="evaluate best_model.npz (default)")
    parse.add_argument("--use_latest", action="store_true", help="evaluate global_model_latest.npz")
    args = parse.parse_args()

    use_best = True
    if args.use_latest:
        use_best = False

    main(args.client, use_best=use_best)
