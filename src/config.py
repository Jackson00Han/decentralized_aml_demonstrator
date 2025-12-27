from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

def find_repo_root(start: Path | None = None) -> Path:
    """
    Find the repository root by walking up directories until config.yaml is found.
    """

    cur = (start or Path(__file__)).resolve()
    for p in [cur, *cur.parents]:
        if (p / "config.yaml").exists():
            return p
    raise FileNotFoundError("Could not find repository root (config.yaml not found).")

def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}

def _p(repo_root: Path, rel: str) -> Path:
    """
    Resolve a repo-relative path into an absolute Path.
    """
    return (repo_root / rel).resolve()

@dataclass(frozen=True)
class Paths:
    repo_root: Path
    data_raw: Path
    data_processed: Path
    outputs_root: Path

    out_local_baseline: Path
    out_fl_clients: Path
    out_fl_server: Path

@dataclass(frozen=True)
class Banks:
    names: List[str]
    amlsim_config_dir: Path

    def amlsim_json(self, bank: str) -> Path:
        """
        Return the AMLSIM config JSON path for a given bank name.
        """

        return self.amlsim_config_dir / f"{bank}.json"
    
@dataclass(frozen=True)
class PreprocessCfg:
    keep_cols: List[str]

@dataclass(frozen=True)
class BaselineCfg:
    top_k: int
    alpha_grid: List[float]
    max_rounds: int
    patience: int
    out_dir: Path



@dataclass(frozen=True)
class FLCfg:
    num_rounds: int
    local_epochs: int
    patience: int
    alpha: float = 1e-4
    alpha_grid: List[float] = field(default_factory=list)
    fk_key: bool = False
    secure_agg_key: str = "simulated_secure_agg_key"
    

@dataclass(frozen=True)
class SchemaCfg:
    version: str
    cat_cols: List[str]
    num_cols: List[str]
    train_frac: float
    val_frac: float
    test_frac: float
    

@dataclass(frozen=True)
class ProjectCfg:
    seed: int

@dataclass(frozen=True)
class Config:
    project: ProjectCfg
    paths: Paths
    banks: Banks
    preprocess: PreprocessCfg
    baseline: BaselineCfg
    fl: FLCfg
    schema: SchemaCfg

def load_config(config_path: Optional[str | Path] = None) -> Config:
    """
    Load config.yaml and return a structured Config object.
    """
    repo_root = find_repo_root()
    cfg_path = Path(config_path) if config_path else (repo_root / "config.yaml")
    raw = _read_yaml(cfg_path)

    # Project
    project = raw.get("project", {})
    seed = int(project.get("seed", 42))

    # Banks
    banks = raw.get("banks", {})
    bank_names = list(banks.get("names", ["bank_a", "bank_b", "bank_c"]))
    amlsim_dir = _p(repo_root, banks.get("amlsim_config_dir", "configs/amlsim"))

    # Paths
    paths = raw.get("paths", {})
    data_raw = _p(repo_root, paths.get("data_raw", "data/raw"))
    data_processed = _p(repo_root, paths.get("data_processed", "data/processed"))
    outputs_root = _p(repo_root, paths.get("outputs_root", "outputs"))

    # Preprocess
    preprocess = raw.get("preprocess", {})
    keep_cols = list(preprocess.get("keep_cols", []))

    # Baseline
    baseline = raw.get("baseline", {})
    top_k = int(baseline.get("top_k", 500))
    baseline_alpha_grid = [float(x) for x in baseline.get("alpha_grid", [1e-3, 1e-4, 1e-5])]
    max_rounds = int(baseline.get("max_rounds", 10))
    baseline_patience = int(baseline.get("patience", 3))
    baseline_out = _p(repo_root, baseline.get("out_dir", "outputs/local_baseline"))

    # FL
    fl = raw.get("fl", {})
    num_rounds = int(fl.get("num_rounds", 2))
    local_epochs = int(fl.get("local_epochs", 2))
    fl_patience = int(fl.get("patience", 3))
    alpha = float(fl.get("alpha", 1e-5))
    fl_alpha_grid = [float(x) for x in fl.get("alpha_grid", baseline_alpha_grid)]
    secure_agg_key = fl.get("secure_agg_key", "")
    fk_key = fl.get("fk_key", False)
    

    out_fl_clients = _p(
        repo_root,
        paths.get(
            "out_fl_clients",
            fl.get("client_out_dir", str(outputs_root / "fl_clients")),
        ),
    )
    out_fl_server = _p(
        repo_root,
        paths.get(
            "out_fl_server",
            fl.get("server_out_dir", str(outputs_root / "fl_server")),
        ),
    )

    # Schema
    schema = raw.get("schema", {})
    schema_version = str(schema.get("version", "v1"))
    cat_cols = list(schema.get("cat_cols", []))
    num_cols = list(schema.get("num_cols", []))
    data_splits = raw.get("data_splits", {})
    train_frac = float(data_splits.get("train_frac", 0.7))
    val_frac = float(data_splits.get("val_frac", 0.2))
    test_frac = float(data_splits.get("test_frac", 0.1))
    

    return Config(
        project=ProjectCfg(seed=seed),
        banks=Banks(names=bank_names, amlsim_config_dir=amlsim_dir),
        paths=Paths(
            repo_root=repo_root,
            data_raw=data_raw,
            data_processed=data_processed,
            outputs_root=outputs_root,
            out_local_baseline=baseline_out,
            out_fl_clients=out_fl_clients,
            out_fl_server=out_fl_server,
        ),
        preprocess=PreprocessCfg(keep_cols=keep_cols),
        baseline=BaselineCfg(
            top_k=top_k,
            alpha_grid=baseline_alpha_grid,
            max_rounds=max_rounds,
            patience=baseline_patience,
            out_dir=baseline_out,
        ),
        fl=FLCfg(
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            patience=fl_patience,
            alpha=alpha,
            alpha_grid=fl_alpha_grid,
            fk_key=fk_key,
            secure_agg_key=secure_agg_key,
        ),
        schema=SchemaCfg(
            version=schema_version, 
            cat_cols=cat_cols,
            num_cols=num_cols,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
        ),
    )
