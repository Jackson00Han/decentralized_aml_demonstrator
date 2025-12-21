from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from .core import Params
from sklearn.linear_model import SGDClassifier



class ModelAdapter(ABC):
    """
    Minimal contract required by our FL loop.
    """

    @abstractmethod
    def initialize(self, X, y) -> None:
        raise NotImplementedError("Adapter not implemented yet")

    @abstractmethod
    def get_params(self) -> Params:
        raise NotImplementedError("Adapter not implemented yet")

    @abstractmethod
    def set_params(self, params: Params) -> None:
        raise NotImplementedError("Adapter not implemented yet")

    @abstractmethod
    def fit_local(self, X, y, local_epochs: int = 1) -> None:
        raise NotImplementedError("Adapter not implemented yet")

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError("Adapter not implemented yet")



class SklearnSGDLogRegAdapter(ModelAdapter):
    def __init__(self, seed: int = 42, alpha: float = 1e-4, class_weight="balanced"):
        self.seed = seed
        self.alpha = alpha
        self.class_weight = class_weight
        self.model: Optional[SGDClassifier] = None
        self._initialized = False

    def initialize(self, X, y) -> None:
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=self.alpha,
            learning_rate="optimal",
            random_state=self.seed,
            class_weight=self.class_weight,
        )
        n_init = min(256, X.shape[0])
        self.model.partial_fit(X[:n_init], y[:n_init], classes=np.array([0, 1], dtype=int))
        self._initialized = True

    def get_params(self) -> Params:
        if not self._initialized or self.model is None:
            raise RuntimeError("Adapter not initialized. Call initialize(X, y) first.")
        return {
            "coef": self.model.coef_.copy(),
            "intercept": self.model.intercept_.copy(),
        }

    def set_params(self, params: Params) -> None:
        if not self._initialized or self.model is None:
            raise RuntimeError("Adapter not initialized. Call initialize(X, y) first.")
        self.model.coef_ = params["coef"].copy()
        self.model.intercept_ = params["intercept"].copy()

    def fit_local(self, X, y, local_epochs: int = 1) -> None:
        if not self._initialized or self.model is None:
            raise RuntimeError("Adapter not initialized. Call initialize(X, y) first.")
        for _ in range(local_epochs):
            self.model.partial_fit(X, y)

    def predict_proba(self, X) -> np.ndarray:
        if not self._initialized or self.model is None:
            raise RuntimeError("Adapter not initialized. Call initialize(X, y) first.")
        return self.model.predict_proba(X)[:, 1]
