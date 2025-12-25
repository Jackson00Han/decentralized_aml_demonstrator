# src/fl_aggregators.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class ClientUpdate:
    bank: str
    n_train: int
    params: Dict[str, np.ndarray]
    #metrics: Dict[str, float] # e.g. {"val_ap":0.85, "val_n":...}

def fedavg(updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
    if not updates:
        raise ValueError("No updates to aggregate.")
    
    valid_updates = [u for u in updates if u.n_train > 0]
    if not valid_updates:
        raise ValueError("All n_train values are zero, cannot aggregate.")
    weights = np.array([u.n_train for u in valid_updates], dtype=float)
    weights /= weights.sum()

    keys = list(valid_updates[0].params.keys())
    out: Dict[str, np.ndarray] = {}

    for k in keys:
        acc = np.zeros_like(valid_updates[0].params[k]) # initialize accumulator
        for w, u in zip(weights, valid_updates):
            acc += w * u.params[k]
        out[k] = acc
    return out