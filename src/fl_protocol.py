# src/fl_protocol.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


@dataclass(frozen=True)
class FeatureSchema:
    cat_cols: List[str]
    num_cols: List[str]


@dataclass(frozen=True)
class GlobalPlan:
    schema_version: str
    feature_schema: FeatureSchema
    global_categories: Dict[str, List[str]]
    global_numeric: Dict[str, Dict[str, float]]  # {col: {"mean":..., "std":...}}

    @staticmethod
    def load(path: Path) -> "GlobalPlan":
        d = json.loads(path.read_text(encoding="utf-8"))
        fs = d["feature_schema"]
        return GlobalPlan(
            schema_version=d["schema_version"],
            feature_schema=FeatureSchema(cat_cols=fs["cat_cols"], num_cols=fs["num_cols"]),
            global_categories=d["global_categories"],
            global_numeric=d["global_numeric"],
        )

    def dump(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "feature_schema": {
                        "cat_cols": self.feature_schema.cat_cols,
                        "num_cols": self.feature_schema.num_cols,
                    },
                    "global_categories": self.global_categories,
                    "global_numeric": self.global_numeric,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


# ---- Model params: use .npz for extensibility ----
# For LogReg: keys = {"coef", "intercept"}
# For Deep/GNN: keys = arbitrary param_name -> ndarray (state_dict-like)
def save_params_npz(path: Path, params: Dict[str, np.ndarray], meta: Dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **params)
    if meta is not None:
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_params_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_meta_json(path: Path) -> Dict[str, Any] | None:
    m = path.with_suffix(".meta.json")
    if not m.exists():
        return None
    return json.loads(m.read_text(encoding="utf-8"))
