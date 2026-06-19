"""Leakage tripwires for the Vol-Engine (A-M0 skeleton; wired at A-M2).

RV-specific adaptation of the v2 tripwires (which were classification-oriented):
* ``label_shuffle_reg`` — shuffle the RV target; forecast R² must collapse to ~0 (chance).
* ``bars_per_day_guard`` — short days (missing hours) bias RV downward; flag if too many.
* reused generically from v2_microstructure: ``t_minus_1_availability_audit`` (exo/HAR vs
  contemporaneous target), ``feature_dominance_flag``, ``summarize``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import config
from .data_spot import audit_bars_per_day

# Reuse the generic, model-agnostic tripwires from the v2 package (config put it on sys.path).
from v2_microstructure.src.tripwires import (  # noqa: E402
    t_minus_1_availability_audit, feature_dominance_flag, summarize,
)


def label_shuffle_reg(X: pd.DataFrame, y: np.ndarray,
                      fit_predict_fn: Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray],
                      *, tol: float = 0.05, seed: int = config.SEED) -> dict:
    """Train on a permuted RV target; OOS R² must NOT be significantly positive.

    ``fit_predict_fn(Xtr,ytr,Xev)->yhat``. A leak would let the model predict the shuffled target
    (R² > tol). A negative R² (worse than the mean) is the expected no-signal outcome and PASSES —
    so the criterion is ``r2 <= tol``, not ``|r2| <= tol``.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    cut = int(0.7 * n)
    y_shuf = np.asarray(y, dtype=float).copy()
    rng.shuffle(y_shuf)
    yhat = fit_predict_fn(X.iloc[:cut], y_shuf[:cut], X.iloc[cut:])
    yt = y_shuf[cut:]
    ss_res = float(np.sum((yt - yhat) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return {"check": "label_shuffle_reg", "passed": r2 <= tol, "shuffled_r2": r2, "tol": tol,
            "note": "leak == shuffled R² > tol; negative R² = no signal (pass)"}


def bars_per_day_guard(spot: pd.DataFrame, min_frac_full: float = 0.95) -> dict:
    """Red if too few full (24-bar) days — RV would be biased by missing hours (R4)."""
    a = audit_bars_per_day(spot)
    return {"check": "bars_per_day", "passed": a["frac_full"] >= min_frac_full,
            "frac_full": a["frac_full"], "short_days": a["short_days"], "min_bars": a["min_bars"]}


def constant_prediction_guard(yhat: np.ndarray, *, tol: float = 1e-8) -> dict:
    """Red if the model emits a (near-)constant forecast — no signal, not an edge."""
    arr = np.asarray(yhat, dtype=float)
    arr = arr[np.isfinite(arr)]
    std = float(arr.std()) if arr.size else 0.0
    return {"check": "constant_prediction", "passed": std > tol, "pred_std": std}


__all__ = ["label_shuffle_reg", "bars_per_day_guard", "constant_prediction_guard",
           "t_minus_1_availability_audit", "feature_dominance_flag", "summarize"]
