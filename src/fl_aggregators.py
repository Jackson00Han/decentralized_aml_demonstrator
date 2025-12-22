from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class ClientUpdate:
    bank: str
    n_train: int
    params: Dict[str, np.ndarray]
    metrics: Dict[str, float] # e.g. {"val_ap":0.85, "val_n":...}

def fedavg(updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
    if not updates:
        raise ValueError("No updates to aggregate.")
    
    weights = np.array([u.n_train for u in updates], dtype=float)
    weights = weights / weights.sum()

    keys = list(updates[0].params.keys())
    out: Dict[str, np.ndarray] = {}

    for k in keys:
        acc = np.zeros_like(updates[0].params[k])
        for w, u in zip(weights, updates):
            acc += w * u.params[k]
        out[k] = acc
    return out