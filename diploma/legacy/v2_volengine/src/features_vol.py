"""ML+exo feature matrix for the Vol-Engine (A-M2), perp-era 2019→ subsample.

Predicts log-RV_{t+1} from:
* HAR components (from range-RV): har_d / har_w / har_m (log-RV, past-only);
* realized measures: √RQ, log1p(jump), signed-RV asymmetry, bipower ratio;
* exo flow from the frozen v2_microstructure C-snapshot — funding/basis (+changes), klines-proxy
  micro (tfi/cvd/vpin) and on-chain *_chg — aggregated to daily and **lagged t−1** (known before t).

Exo only exists on the perp era (≈2019-09→), so the matrix is the **gate subsample** for M3: the HAR
arm and the ML+exo arm share identical 2019→ rows. Train-only collinearity clustering + stable
(MDA, dev-fold) selection are thin wrappers over the reused v2 walk_forward utilities — nothing is
fit outside train, and the holdout is never touched here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, rv as rvmod

# reused infra (config put sibling packages on sys.path)
from walk_forward import cluster_collinear, select_stable_features, expanding_window_splits  # noqa: E402
from v2_microstructure.src.data_snapshot import load_snapshot as load_exo_snapshot  # noqa: E402

HAR_COLS = ["har_d", "har_w", "har_m"]
REALIZED_COLS = ["rq_term", "jump_d", "signed_rv", "bpv_ratio"]
EXO_MEAN_COLS = ["funding", "funding_chg", "basis", "basis_chg", "tfi", "cvd", "vpin"]
EXO_LAST_COLS = ["active_addr_chg", "tx_cnt_chg", "addr_bal_chg", "hashrate_chg", "supply_chg"]


def aggregate_exo_daily(exo_hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily exo from the hourly C-snapshot, then lag t−1 (state known before day t)."""
    day = exo_hourly.index.normalize()
    parts = []
    means = [c for c in EXO_MEAN_COLS if c in exo_hourly.columns]
    lasts = [c for c in EXO_LAST_COLS if c in exo_hourly.columns]
    if means:
        parts.append(exo_hourly[means].groupby(day).mean())
    if lasts:
        parts.append(exo_hourly[lasts].groupby(day).last())
    daily = pd.concat(parts, axis=1)
    daily.index.name = "date"
    return daily.shift(1)                       # t-1 availability (no same-day exo leak)


def build_feature_matrix(spot: pd.DataFrame, estimator: str = config.RV_ESTIMATOR,
                         exo_hourly: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Full ML+exo matrix on the perp-era subsample. Returns (X, groups). X has a ``target`` column
    (logRV_{t+1}); ``groups`` maps har/realized/exo → column lists for the M3 arms."""
    frame = rvmod.build_rv(spot, estimator=estimator)
    logrv = np.log(frame["rv"].clip(lower=config.RV_FLOOR))
    X = pd.DataFrame(index=frame.index)
    X["har_d"] = logrv
    X["har_w"] = logrv.rolling(config.HAR_WINDOWS[1]).mean()
    X["har_m"] = logrv.rolling(config.HAR_WINDOWS[2]).mean()
    X["rq_term"] = np.sqrt(frame["rq"].clip(lower=0))
    X["jump_d"] = np.log1p(frame["jump"].clip(lower=0))
    X["signed_rv"] = (frame["rv_plus"] - frame["rv_minus"]) / frame["rv"].clip(lower=config.RV_FLOOR)
    X["bpv_ratio"] = frame["bpv"] / frame["rv"].clip(lower=config.RV_FLOOR)

    exo = aggregate_exo_daily(exo_hourly if exo_hourly is not None else load_exo_snapshot())
    exo_cols = [c for c in (EXO_MEAN_COLS + EXO_LAST_COLS) if c in exo.columns]
    X = X.join(exo[exo_cols])
    X["target"] = frame["target_next"]

    X = X.loc[pd.Timestamp(config.PERP_ERA_START):]      # perp-era subsample (exo exists here)
    groups = {"har": HAR_COLS, "realized": REALIZED_COLS, "exo": exo_cols,
              "all": HAR_COLS + REALIZED_COLS + exo_cols}
    return X, groups


def dev_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Perp-era DEV rows (holdout sealed) with complete features+target (drop warmup/last-day NaN)."""
    dev, _ = rvmod.seal_holdout(X.index)
    cols = [c for c in X.columns]
    return X[dev.to_numpy()].dropna(subset=cols)


def stable_select(X_dev: pd.DataFrame, feature_cols, k: int = 10,
                  method: str = "mda") -> tuple[list[str], list[list[str]]]:
    """Train-only collinearity clustering + median-importance stable selection (dev folds only).

    Returns (selected_features, collinear_clusters). Falls back to 'model_gain' if 'mda' degenerates.
    """
    corr = X_dev[feature_cols].corr().abs().fillna(0.0)
    clusters = [c for c in cluster_collinear(corr, threshold=0.9) if len(c) > 1]
    splits = expanding_window_splits(len(X_dev), n_splits=config.WF_N_SPLITS)
    try:
        sel = select_stable_features(X_dev, feature_cols, "target", splits, k=k, method=method)
    except Exception:
        sel = select_stable_features(X_dev, feature_cols, "target", splits, k=k, method="model_gain")
    return list(sel), clusters
