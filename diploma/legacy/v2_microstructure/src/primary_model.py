"""Stage-1 side classifier ladder (M3).

A 3-class {+1, 0, -1} classifier ladder, from a trivial baseline up to LightGBM, each rung
proven against the previous on economics (not accuracy). Every model is trained with the
unified ``sample_weight`` from labeling (uniqueness x |move| x class-balance), so class
imbalance is handled there and never double-counted as a model ``class_weight``.

Leakage safety: linear models use an sklearn ``Pipeline`` (median-impute + standardise) whose
transformers fit on the TRAIN fold only by construction; LightGBM consumes NaNs natively.

The model emits a tradable **side only on a ±1 prediction**; a predicted ``0`` is flat and
does not enter the Stage-2 meta-dataset (handled in meta_model.py / strategy.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

from . import config

LADDER = ("baseline", "logistic", "lgbm")
CLASSES = (-1, 0, 1)


def make_model(name: str, seed: int = config.SEED):
    """Factory for a ladder rung. Final estimator step is always named ``clf``."""
    if name == "baseline":
        # stratified == random side drawn from the train class base-rate (a real null).
        return DummyClassifier(strategy="stratified", random_state=seed)
    if name == "logistic":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ])
    if name == "lgbm":
        return lgb.LGBMClassifier(
            n_estimators=config.lgbm_n_estimators(), num_leaves=31, learning_rate=0.05,
            min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1,
        )
    raise ValueError(f"unknown model {name!r}")


def calibrated(name: str = "lgbm", seed: int = config.SEED,
               method: str = "isotonic", cv: int = 3):
    """Train-only probability calibration around a ladder rung (the probability feeds M4 sizing).

    NOTE: the M3 gate deliberately uses **uncalibrated** models, identically on both arms — what
    matters there is equal treatment, not calibration. Calibration is critical at M4 (sizing).
    ``CalibratedClassifierCV`` does its own internal CV on the training fold (leakage-safe) and
    forwards ``sample_weight`` to the base estimator.
    """
    return CalibratedClassifierCV(make_model(name, seed), method=method, cv=cv)


def fit_model(model, X: pd.DataFrame, y, w=None):
    """Fit with sample_weight, routing to the final pipeline step when needed."""
    if w is None:
        model.fit(X, y)
        return model
    try:
        model.fit(X, y, sample_weight=np.asarray(w))
    except (TypeError, ValueError):
        last = model.steps[-1][0]
        model.fit(X, y, **{f"{last}__sample_weight": np.asarray(w)})
    return model


def predict_side_proba(model, X: pd.DataFrame):
    """Return (side, p_max, proba_df). side == predicted class in {-1,0,1}."""
    side = np.asarray(model.predict(X)).astype(int)
    if hasattr(model, "predict_proba"):
        proba = pd.DataFrame(model.predict_proba(X), index=X.index,
                             columns=[int(c) for c in model.classes_])
        p_max = proba.max(axis=1).to_numpy()
    else:  # pragma: no cover - all ladder rungs expose predict_proba
        proba = pd.DataFrame(index=X.index)
        p_max = np.full(len(X), np.nan)
    return side, p_max, proba


def walk_forward_oof(X: pd.DataFrame, y: pd.Series, w: pd.Series, cols, splits,
                     model_name: str = "lgbm", seed: int = config.SEED) -> pd.DataFrame:
    """Out-of-fold Stage-1 predictions over walk-forward ``splits`` (positions into X/y).

    Returns a frame indexed by the test-event timestamp with columns:
    ``side`` (predicted class), ``p_side`` (calibrated-enough P(predicted class)),
    ``p_up``/``p_flat``/``p_down`` (class probabilities), ``fold``. Feeds Stage-2 (M4).
    """
    Xc = X[cols]
    rows = []
    for fold, (tr, te) in enumerate(splits):
        m = fit_model(make_model(model_name, seed), Xc.iloc[tr], y.iloc[tr], w.iloc[tr])
        side, p_side, proba = predict_side_proba(m, Xc.iloc[te])
        out = pd.DataFrame({"side": side, "p_side": p_side, "fold": fold}, index=Xc.index[te])
        for c, name in zip((1, 0, -1), ("p_up", "p_flat", "p_down")):
            out[name] = proba[c].to_numpy() if c in proba.columns else np.nan
        rows.append(out)
    return pd.concat(rows).sort_index()
