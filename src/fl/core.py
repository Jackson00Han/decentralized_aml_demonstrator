from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

Params = Dict[str, np.ndarray]


@dataclass
class ClientUpdate:
    client_id: str
    n_train: int
    params: Params


class FedAvgServer:
    """
    MVP FedAvg: model-agnostic parameter averaging with minimal safety checks.
    """

    def __init__(self, init_params: Params):
        if not init_params:
            raise ValueError("init_params is empty")

        # copy to avoid aliasing
        self.params: Params = {k: np.array(v, copy=True) for k, v in init_params.items()}

        self._keys = tuple(sorted(self.params.keys()))
        self._shapes = {k: self.params[k].shape for k in self.params}

    def get_params(self) -> Params:
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, new_params: Params) -> None:
        self._validate_params(new_params, where="set_params")
        self.params = {k: np.array(v, copy=True) for k, v in new_params.items()}

    def aggregate(self, updates: List[ClientUpdate]) -> Params:
        if not updates:
            raise ValueError("No client updates.")

        total_n = 0
        for u in updates:
            if u.n_train <= 0:
                raise ValueError(f"Invalid n_train from {u.client_id}: {u.n_train}")
            self._validate_params(u.params, where=f"client:{u.client_id}")
            total_n += u.n_train

        if total_n <= 0:
            raise ValueError("total n_train is zero; cannot aggregate.")

        agg: Params = {k: np.zeros_like(self.params[k]) for k in self.params}

        for u in updates:
            w = u.n_train / total_n
            for k in agg:
                agg[k] += w * u.params[k]

        self.set_params(agg)
        return self.get_params()

    def _validate_params(self, params: Params, where: str) -> None:
        keys = tuple(sorted(params.keys()))
        if keys != self._keys:
            raise ValueError(f"[{where}] params keys mismatch")

        for k in self._keys:
            v = params[k]
            if not isinstance(v, np.ndarray):
                raise TypeError(f"[{where}] params['{k}'] must be np.ndarray")
            if v.shape != self._shapes[k]:
                raise ValueError(f"[{where}] params['{k}'] shape mismatch: {v.shape} != {self._shapes[k]}")
