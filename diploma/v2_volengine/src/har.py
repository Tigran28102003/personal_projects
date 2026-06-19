"""HAR family + naive RV baselines (A-M1) — the guaranteed Layer-1 deliverable.

Models log-RV_{t+1} on the full spot-era dev series (2017→). All HAR components are past-only
(built from RV up to day t to predict day t+1), in **log space** (the standard log-HAR):
    log-HAR : logRV_{t+1} ~ logRV_d + logRV_w + logRV_m
    HAR-J   : + a jump component  log1p(jump_t)
    HARQ    : + a realized-quarticity term √RQ_t (central; corrects RV measurement error, R2)
Naive baselines HAR must beat (the MASE discipline that caught the persistence illusion in the
original diploma): RW (logRV_{t+1}=logRV_t), AR(1) on log-RV, unconditional mean.

Forecasts are produced out-of-fold via expanding walk-forward with a small embargo; nothing is fit
outside the train block. Level (RV) forecasts apply the lognormal/Jensen correction
exp(pred + ½·σ²_resid) so the back-transformed level is unbiased.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import config

# legacy/v2 infra (config put it on sys.path)
from walk_forward import expanding_window_splits  # noqa: E402

MODELS = ("RW", "MEAN", "AR1", "HAR", "HARJ", "HARQ")
_DESIGN = {
    "AR1": ["har_d"],
    "HAR": ["har_d", "har_w", "har_m"],
    "HARJ": ["har_d", "har_w", "har_m", "jump_d"],
    "HARQ": ["har_d", "har_w", "har_m", "rq_term"],
}


def har_components(frame: pd.DataFrame, windows=config.HAR_WINDOWS) -> pd.DataFrame:
    """Past-only HAR regressors at day t (predicting t+1). ``frame`` from rv.build_rv."""
    logrv = np.log(frame["rv"].clip(lower=config.RV_FLOOR))
    X = pd.DataFrame(index=frame.index)
    X["har_d"] = logrv                                   # RV_d (daily) in log space
    X["har_w"] = logrv.rolling(windows[1]).mean()        # weekly
    X["har_m"] = logrv.rolling(windows[2]).mean()        # monthly
    X["jump_d"] = np.log1p(frame["jump"].clip(lower=0))  # HAR-J jump component
    X["rq_term"] = np.sqrt(frame["rq"].clip(lower=0))    # HARQ quarticity term
    X["target"] = frame["target_next"]                   # logRV_{t+1}
    return X


def _fit_predict(name, Xtr, ytr, Xte):
    """Return (logrv_pred on test, sigma2_resid on train) for one model."""
    if name == "MEAN":
        mu = float(ytr.mean())
        pred = np.full(len(Xte), mu)
        resid = ytr.to_numpy() - mu
    elif name == "RW":                                   # logRV_{t+1} = logRV_t (= har_d)
        pred = Xte["har_d"].to_numpy()
        resid = ytr.to_numpy() - Xtr["har_d"].to_numpy()
    else:
        cols = _DESIGN[name]
        lr = LinearRegression().fit(Xtr[cols], ytr)
        pred = lr.predict(Xte[cols])
        resid = ytr.to_numpy() - lr.predict(Xtr[cols])
    return pred, float(np.var(resid))


def walk_forward_oof(frame: pd.DataFrame, n_splits: int = config.WF_N_SPLITS,
                     embargo: int = 5) -> pd.DataFrame:
    """Out-of-fold log-RV forecasts for every model. Returns a frame indexed by day with columns
    ``{model}_logrv`` (log-scale OOF forecast) and ``{model}_rv`` (Jensen-corrected RV level),
    plus ``y_logrv`` (true logRV_{t+1}) and ``y_rv`` (true RV_{t+1})."""
    X = har_components(frame).dropna(subset=["har_m", "target"])  # drop monthly warmup + last day
    y = X["target"]
    n = len(X)
    out = pd.DataFrame(index=X.index)
    out["y_logrv"] = y
    out["y_rv"] = np.exp(y)
    cols = {m: (np.full(n, np.nan), np.full(n, np.nan)) for m in MODELS}

    for tr, te in expanding_window_splits(n, n_splits=n_splits):
        tr = tr[tr < te[0] - embargo]                    # embargo gap between train and test
        if len(tr) < config.HAR_WINDOWS[2] + 10:
            continue
        Xtr, ytr, Xte = X.iloc[tr], y.iloc[tr], X.iloc[te]
        for m in MODELS:
            pred, s2 = _fit_predict(m, Xtr, ytr, Xte)
            cols[m][0][te] = pred
            cols[m][1][te] = np.exp(pred + 0.5 * s2) if config.TARGET_TRANSFORM == "log" else pred ** 2
    for m in MODELS:
        out[f"{m}_logrv"] = cols[m][0]
        out[f"{m}_rv"] = cols[m][1]
    return out.dropna(subset=[f"{m}_logrv" for m in MODELS])
