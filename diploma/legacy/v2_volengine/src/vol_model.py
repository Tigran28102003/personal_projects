"""ML ladder for Layer-1 (A-M3): HARQ baseline → Ridge(HAR+exo) → LightGBM(reg).

Regression of log-RV_{t+1} on the perp-era 2019→ subsample (the M2 gate matrix), isolating the
marginal contribution at each rung:
* **HARQ** — the strong HAR baseline ML must beat (LinearRegression on HAR_d/w/m + √RQ);
* **Ridge(HAR+exo)** — does exo/realised flow add value **linearly** over HAR? (L2, std features);
* **LightGBM(reg)** — does **nonlinearity** add over the linear model? (full regressor set).

The ML arms get the **full** candidate regressor set (HAR + realised + exo, lagged t−1) — no separate
feature-selection step, so there is no selection leak and the model is given its best honest chance;
Ridge regularisation / GBM handle the 19 collinear-clustered regressors. All preprocessing is
**train-only** inside each fold (Ridge standardisation fit on train); the level forecast applies the
lognormal/Jensen correction exp(pred + ½·σ²_resid). Both arms share identical folds and rows so Gate 1
(vol_evaluate.gate1_forecasting) is a fair, pre-registered comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from . import config
from .features_vol import HAR_COLS

# legacy/v2 infra (config put it on sys.path)
from walk_forward import expanding_window_splits  # noqa: E402
from validation import cpcv_splits                # noqa: E402

HARQ_COLS = HAR_COLS + ["rq_term"]               # the HAR baseline (best HAR family member = HARQ)
ARMS = ("HARQ", "Ridge", "LGBM")
ML_ARMS = ("Ridge", "LGBM")


def _lgbm_reg():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=config.lgbm_n_estimators(), num_leaves=31,
                             learning_rate=0.05, min_child_samples=50, subsample=0.8,
                             colsample_bytree=0.8, random_state=config.SEED, n_jobs=-1, verbose=-1)


def _fit_predict_arm(arm: str, Xtr: pd.DataFrame, ytr: pd.Series, Xte: pd.DataFrame, ml_cols):
    """Train-only fit of one arm → (logrv_pred on test, σ²_resid on train) for the Jensen back-transform.

    The test forecast is clipped to the plausible log-RV support of the training target (± margin) and
    σ²_resid is capped, so a single wild extrapolation cannot produce a nonsensical RV (and a degenerate
    QLIKE blow-up). Applied **uniformly to every arm** — numerical hygiene, not arm-specific tuning."""
    if arm == "HARQ":
        m = LinearRegression().fit(Xtr[HARQ_COLS], ytr)
        pred, fit_tr = m.predict(Xte[HARQ_COLS]), m.predict(Xtr[HARQ_COLS])
    elif arm == "Ridge":
        sc = StandardScaler().fit(Xtr[ml_cols])
        m = Ridge(alpha=config.RIDGE_ALPHA).fit(sc.transform(Xtr[ml_cols]), ytr)
        pred, fit_tr = m.predict(sc.transform(Xte[ml_cols])), m.predict(sc.transform(Xtr[ml_cols]))
    elif arm == "LGBM":
        m = _lgbm_reg().fit(Xtr[ml_cols], ytr)
        pred, fit_tr = m.predict(Xte[ml_cols]), m.predict(Xtr[ml_cols])
    else:
        raise ValueError(f"unknown arm {arm!r}")
    sd = float(ytr.std())
    lo, hi = float(ytr.min()) - 2.0 * sd, float(ytr.max()) + 2.0 * sd     # plausible log-RV support
    pred = np.clip(np.asarray(pred, dtype=float), lo, hi)
    resid = ytr.to_numpy() - np.asarray(fit_tr)
    s2 = min(float(np.var(resid)), 2.0 * float(np.var(ytr.to_numpy())))   # cap Jensen variance term
    return pred, s2


def wf_oof_forecasts(X: pd.DataFrame, ml_cols, n_splits: int = config.WF_N_SPLITS,
                     embargo: int = 5) -> pd.DataFrame:
    """Expanding walk-forward OOF (clean, non-overlapping) for every arm — the readable headline.

    Returns a frame with y_logrv / y_rv and ``{arm}_logrv`` / ``{arm}_rv`` (Jensen-corrected)."""
    Xc = X.dropna(subset=list(ml_cols) + ["target"])
    y = Xc["target"]
    n = len(Xc)
    out = pd.DataFrame(index=Xc.index)
    out["y_logrv"] = y
    out["y_rv"] = np.exp(y)
    cols = {a: (np.full(n, np.nan), np.full(n, np.nan)) for a in ARMS}
    for tr, te in expanding_window_splits(n, n_splits=n_splits):
        tr = tr[tr < te[0] - embargo]
        if len(tr) < 60:
            continue
        Xtr, ytr, Xte = Xc.iloc[tr], y.iloc[tr], Xc.iloc[te]
        for a in ARMS:
            pred, s2 = _fit_predict_arm(a, Xtr, ytr, Xte, ml_cols)
            cols[a][0][te] = pred
            cols[a][1][te] = np.exp(pred + 0.5 * s2)
    for a in ARMS:
        out[f"{a}_logrv"] = cols[a][0]
        out[f"{a}_rv"] = cols[a][1]
    return out.dropna(subset=[f"{a}_logrv" for a in ARMS])


def cpcv_arm_losses(X: pd.DataFrame, ml_cols, N: int = config.CPCV_N, k: int = config.CPCV_K):
    """Per-CPCV-path QLIKE loss series for each arm (distributional input for Gate 1).

    Each path = one combinatorial-purged split (C(N,k) paths). Returns a list of dicts
    ``{"y_rv", "n_test", "HARQ", "Ridge", "LGBM"}`` where each arm value is the per-obs QLIKE loss on
    that path's purged test block. Fit is strictly train-only."""
    from .vol_evaluate import qlike_series
    Xc = X.dropna(subset=list(ml_cols) + ["target"]).reset_index(drop=True)
    y = Xc["target"]
    paths = []
    for tr, te in cpcv_splits(len(Xc), N=N, k=k, horizon=1):
        Xtr, ytr, Xte = Xc.iloc[tr], y.iloc[tr], Xc.iloc[te]
        y_rv_te = np.exp(y.iloc[te].to_numpy())
        rec = {"y_rv": y_rv_te, "n_test": int(len(te))}
        for a in ARMS:
            pred, s2 = _fit_predict_arm(a, Xtr, ytr, Xte, ml_cols)
            rec[a] = qlike_series(y_rv_te, np.exp(pred + 0.5 * s2))
        paths.append(rec)
    return paths
