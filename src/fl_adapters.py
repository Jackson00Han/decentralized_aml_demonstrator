from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from scipy.special import expit
from sklearn.linear_model import SGDClassifier


class ModelAdapter(ABC):
    @abstractmethod
    def get_params(self) -> Dict[str, np.ndarray]:
        ...

    @abstractmethod
    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        ...

    @abstractmethod
    def train_one_round(self, X_train, y_train: np.ndarray, local_epochs: int, seed: int) -> None:
        ...

    @abstractmethod
    def predict_scores(self, X) -> np.ndarray:
        ...


class SkLogRegSGD(ModelAdapter):
    """
    Logistic regression trained by SGD (partial_fit), suitable for FL FedAvg.

    Params format:
      - "coef": (1, d)
      - "intercept": (1,)
    """

    def __init__(
        self,
        d: int,
        alpha: float = 1e-5,
        eta0: float = 0.01,
        learning_rate: str = "constant",
        seed: int = 42,
    ):
        self.d = int(d)
        self.seed = int(seed)
        self.clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            learning_rate=learning_rate,
            eta0=eta0,
            fit_intercept=True,
            random_state=seed,
            average=False,
        )
        self._inited = False

    def _assert_dim(self, X):
        if X.shape[1] != self.d:
            raise ValueError(f"Feature dim mismatch: X has {X.shape[1]} features but adapter d={self.d}")

    def _init_if_needed(self, X, y):
        if self._inited:
            return
        self._assert_dim(X)
        self.clf.partial_fit(X[:1], y[:1], classes=np.array([0, 1], dtype=int))
        self._inited = True
        # ensure shapes
        if self.clf.coef_.shape != (1, self.d):
            self.clf.coef_ = np.zeros((1, self.d), dtype=float)
        if self.clf.intercept_.shape != (1,):
            self.clf.intercept_ = np.zeros((1,), dtype=float)

    def get_params(self) -> Dict[str, np.ndarray]:
        if not self._inited:
            raise RuntimeError("Model not initialized. Call set_params() or train_one_round() first.")
        return {"coef": self.clf.coef_.copy(), "intercept": self.clf.intercept_.copy()}

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        if "coef" not in params or "intercept" not in params:
            raise KeyError(f"Expected keys ['coef','intercept'], got {list(params.keys())}")

        coef = np.asarray(params["coef"], dtype=float)
        intercept = np.asarray(params["intercept"], dtype=float)

        if coef.shape != (1, self.d) or intercept.shape != (1,):
            raise ValueError(f"Bad shapes: coef {coef.shape}, intercept {intercept.shape}; expected (1,{self.d}) and (1,)")

        # initialize internal sklearn state
        if not self._inited:
            X0 = np.zeros((1, self.d))
            y0 = np.array([0], dtype=int)
            self.clf.partial_fit(X0, y0, classes=np.array([0, 1], dtype=int))
            self._inited = True

        self.clf.coef_ = coef.copy()
        self.clf.intercept_ = intercept.copy()

    def train_one_round(self, X_train, y_train: np.ndarray, local_epochs: int, seed: int) -> None:
        self._assert_dim(X_train)
        y_train = y_train.astype(int)
        self._init_if_needed(X_train, y_train)

        rng = np.random.default_rng(self.seed + int(seed))
        n = X_train.shape[0]
        for _ in range(int(local_epochs)):
            idx = rng.permutation(n) # like shuffling dataset each epoch
            self.clf.partial_fit(X_train[idx], y_train[idx]) # like backforward step of deep learning

    def predict_scores(self, X) -> np.ndarray:
        self._assert_dim(X)
        # decision_function is stable for sparse/dense
        z = self.clf.decision_function(X)
        return expit(z)


# new adapters go here:
# - torch_mlp_adapter.py (later)
# - torch_gnn_adapter.py (later)
