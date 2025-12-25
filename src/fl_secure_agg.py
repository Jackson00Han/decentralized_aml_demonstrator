# src/secure_agg.py
from __future__ import annotations

import hashlib
import os
from typing import Iterable, List

import numpy as np


ENV_KEY = "FL_SECURE_AGG_KEY"


def get_secure_agg_key() -> str:
    """
    Read the secure aggregation secret from environment variable.
    IMPORTANT: the server must NOT have this key.
    """
    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        raise RuntimeError(
            f"Missing secure aggregation key. Either set environment variable {ENV_KEY}, "
            f"or enable simulation mode fk_key=true and provide fl.secure_agg_key in config."
        )
    return key


def _pair_seed(secret: str, a: str, b: str, round_id: int, tag: str) -> int:
    """
    Deterministically derive a shared seed for pair (a,b) using a shared secret.
    Both clients can compute the same seed locally.

    We sort (a,b) to ensure symmetry.
    """
    x, y = (a, b) if a < b else (b, a)
    msg = f"{secret}|{x}|{y}|{int(round_id)}|{tag}".encode("utf-8")
    h = hashlib.sha256(msg).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def _pairwise_mask_sum(
    secret: str,
    me: str,
    participants: List[str],
    round_id: int,
    tag: str,
    scale: float,
) -> float:
    """
    Pairwise masks that cancel out when summed across ALL participants.

    For each pair (me, other), both can generate the same random mask m.
    - If me < other: +m
    - Else: -m

    When the server sums masked values across all clients, masks cancel out:
      (m_{i,j} added by i) + (-m_{i,j} added by j) = 0
    """
    out = 0.0
    for other in participants:
        if other == me:
            continue
        seed = _pair_seed(secret, me, other, round_id, tag)
        rng = np.random.default_rng(seed)  # deterministic across clients
        m = float(rng.normal(loc=0.0, scale=float(scale)))
        out += m if me < other else -m
    return out


def mask_value(
    value: float,
    me: str,
    participants: List[str],
    round_id: int,
    tag: str,
    scale: float = 1000.0,
    secret: str | None = None,
) -> float:
    """
    Return masked value = value + pairwise_mask_sum(...).
    """
    if secret is None:
        secret = get_secure_agg_key()
    mask = _pairwise_mask_sum(secret, me, participants, round_id, tag, scale)
    return float(value) + float(mask)


def mask_metric_fraction(
    numerator: float,
    denominator: float,
    me: str,
    participants: List[str],
    round_id: int,
    scale: float = 1000.0,
    secret: str | None = None,
) -> tuple[float, float]:
    """
    Securely mask a weighted fraction numerator/denominator.
    Example:
      numerator = val_ap * val_n
      denominator = val_n

    Server can sum masked numerators/denominators across clients and then divide to
    get the global weighted average, without seeing per-client metrics.
    """
    num_m = mask_value(numerator, me, participants, round_id, tag="metric_num", scale=scale, secret=secret)
    den_m = mask_value(denominator, me, participants, round_id, tag="metric_den", scale=scale, secret=secret)
    return num_m, den_m
