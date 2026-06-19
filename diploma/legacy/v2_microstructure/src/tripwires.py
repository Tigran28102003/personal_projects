"""Automatic leakage tripwires, run as part of every experiment (M0 skeleton; wired at M2).

The premise: a "too good" result is a leakage bug until proven otherwise. These checks are
cheap, mechanical, and pre-registered:

* ``label_shuffle_test`` — shuffle the target; forecast quality MUST collapse to ~chance.
  Generic over the modelling harness via a ``fit_predict_fn``.
* ``t_minus_1_availability_audit`` — every feature at row t must be known before t. Flags
  features that are suspiciously equal to (or perfectly correlated with) a same-bar reference
  (i.e. the causal shift was not applied). Trailing sigma_t and funding are explicit cases.
* ``feature_dominance_flag`` — one feature carrying > threshold of importance is a likely leak.
* ``single_class_guard`` — a constant / single-class prediction is "no signal", not an edge.

Each returns a small dict {passed: bool, ...}; the experiment aggregates them and refuses to
trust any result while a tripwire is red.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


def single_class_guard(y_pred: Sequence) -> dict:
    """Red if predictions collapse to a single class / constant value."""
    arr = pd.Series(list(y_pred)).dropna()
    n_unique = int(arr.nunique())
    return {"check": "single_class_guard", "passed": n_unique > 1,
            "n_unique": n_unique, "n": int(len(arr))}


def feature_dominance_flag(importances: Mapping[str, float], threshold: float = 0.5) -> dict:
    """Red if a single feature's normalised importance exceeds ``threshold`` (likely leak)."""
    if not importances:
        return {"check": "feature_dominance", "passed": True, "top_feature": None, "share": 0.0}
    total = float(sum(abs(v) for v in importances.values())) or 1.0
    top = max(importances.items(), key=lambda kv: abs(kv[1]))
    share = abs(top[1]) / total
    return {"check": "feature_dominance", "passed": share <= threshold,
            "top_feature": top[0], "share": float(share), "threshold": threshold}


def label_shuffle_test(X: pd.DataFrame, y: np.ndarray,
                       fit_predict_fn: Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray],
                       metric_fn: Callable[[np.ndarray, np.ndarray], float],
                       chance_level: float, *, tol: float = 0.05, seed: int = 0) -> dict:
    """Train on a permuted target; the metric must collapse to ~chance.

    ``fit_predict_fn(X_train, y_train, X_eval) -> y_pred`` is the harness's own causal
    fit+predict (so the test exercises the real pipeline). ``metric_fn(y_true, y_pred)``
    is a proper score whose chance value is ``chance_level``. Passes if the shuffled-target
    score is within ``tol`` of chance.
    """
    rng = np.random.default_rng(seed)
    y_shuf = np.asarray(y).copy()
    rng.shuffle(y_shuf)
    y_pred = fit_predict_fn(X, y_shuf, X)
    score = float(metric_fn(y_shuf, y_pred))
    return {"check": "label_shuffle", "passed": abs(score - chance_level) <= tol,
            "score": score, "chance_level": chance_level, "tol": tol}


def t_minus_1_availability_audit(features: pd.DataFrame, reference: pd.Series, *,
                                 corr_threshold: float = 0.999) -> dict:
    """Flag features that are (near-)perfectly correlated with a CONTEMPORANEOUS reference.

    After the causal t-1 shift, no feature should track the same-bar reference (e.g. the
    bar return) at correlation ~1; that would mean the shift was missed. Trailing sigma_t and
    funding are audited the same way (they must be known at/before t).
    """
    flagged = {}
    ref = reference.reindex(features.index)
    for col in features.columns:
        s = features[col]
        pair = pd.concat([s, ref], axis=1).dropna()
        if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2:
            continue
        c = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        if np.isfinite(c) and abs(c) >= corr_threshold:
            flagged[col] = c
    return {"check": "t_minus_1_availability", "passed": not flagged,
            "flagged": flagged, "corr_threshold": corr_threshold}


def summarize(results: Sequence[dict]) -> dict:
    """Aggregate tripwire results into a single pass/fail with the red checks listed."""
    red = [r["check"] for r in results if not r.get("passed", False)]
    return {"all_passed": not red, "red": red, "n_checks": len(results)}
