"""Feature assembly for direction C (M2).

Builds the candidate feature matrix from the frozen snapshot, with every feature placed at
its CAUSAL position (value at row ``t`` depends only on bars ``≤ t-1``). Two arms are exposed
for the M3 ablation gate:

* ``arm_flow``   — flow/derivatives (funding/basis) + klines-proxy microstructure (tfi/cvd/
  vpin) + on-chain, plus deterministic time. The thesis: information not in the price path.
* ``arm_control`` — close-price-only null: return lags, rolling stats, RSI/MACD, fractional-
  differencing memory, plus the same deterministic time.

Both arms share the deterministic cyclical-time block (a neutral baseline), so the contrast
is strictly flow vs price. Leakage controls:
* market/flow/micro features shifted by ``BASE_LAG`` (state at t-1);
* on-chain shifted by an extra ``ONCHAIN_PUB_LAG_BARS`` (free-tier daily publication delay);
* cyclical time is deterministic and NOT shifted (known at t);
* fractional-differencing order ``d*`` is fitted on the DEV slice only (holdout sealed); the
  FFD transform itself is causal (fixed backward window).

Per-fold, train-only collinearity clustering and stable selection are provided as thin
wrappers over ``walk_forward`` (used inside the M3 loop).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .labeling import Labels

# Sibling legacy library in ACTUAL_VERSION (config.LEGACY_LIB put it on sys.path)
from fracdiff import frac_diff_ffd, min_ffd_d  # noqa: E402
from walk_forward import (  # noqa: E402
    add_cyclical_time_features, CYCLICAL_FEATURES,
    select_stable_features, cluster_collinear,
)
from free_features import ORDERFLOW_FEATURES, DERIV_FEATURES, ONCHAIN_FEATURES  # noqa: E402

BASE_LAG = 1
ONCHAIN_PUB_LAG_BARS = 24          # ~1 trading day of free-tier on-chain publication delay
RET_LAGS = (1, 2, 3, 6, 12, 24)
ROLL_WINDOWS = (6, 24, 72)

FLOW_COLS = list(DERIV_FEATURES)   # funding, funding_chg, basis, basis_chg
MICRO_COLS = list(ORDERFLOW_FEATURES)  # tfi, cvd, vpin
ONCHAIN_COLS = list(ONCHAIN_FEATURES)


def _rsi_macd(close: pd.Series):
    """Reuse get_data's RSI/MACD (lazy import: keeps this module light and import-robust)."""
    try:
        from get_data import _rsi, _macd
        return _rsi(close), _macd(close)
    except Exception:  # fallback if get_data's heavy deps are unavailable
        delta = close.diff()
        up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        return rsi, macd


def build_features(snapshot: pd.DataFrame, dev_mask: pd.Series | None = None):
    """Build the causal candidate matrix. Returns ``(X, meta)``.

    ``meta`` carries the group→columns map, the fitted ``d_star``, and the two arm column
    lists (``arm_flow`` / ``arm_control``). ``dev_mask`` (index < holdout cut) restricts the
    fractional-differencing ``d*`` fit to the dev slice; if None, the full series is used.
    """
    idx = snapshot.index
    close = snapshot["perp_close"].astype(float)
    logp = np.log(close)
    r = logp.diff()

    X = pd.DataFrame(index=idx)
    groups = {"flow": [], "micro": [], "onchain": [], "memory": [], "control": [], "time": []}

    # --- flow / derivatives (the hourly core) ---
    for c in FLOW_COLS:
        if c in snapshot.columns and snapshot[c].notna().any():
            X[c] = snapshot[c].shift(BASE_LAG)
            groups["flow"].append(c)
    # --- klines-proxy microstructure (not tick-level) ---
    for c in MICRO_COLS:
        if c in snapshot.columns and snapshot[c].notna().any():
            X[c] = snapshot[c].shift(BASE_LAG)
            groups["micro"].append(c)
    # --- on-chain (daily; extra publication-lag) ---
    for c in ONCHAIN_COLS:
        if c in snapshot.columns and snapshot[c].notna().any():
            X[c] = snapshot[c].shift(BASE_LAG + ONCHAIN_PUB_LAG_BARS)
            groups["onchain"].append(c)

    # --- close-price control block ---
    for j in RET_LAGS:
        col = f"ret_l{j}"
        X[col] = r.shift(j)
        groups["control"].append(col)
    for w in ROLL_WINDOWS:
        cm, cs = f"ret_mean_{w}", f"ret_std_{w}"
        X[cm] = r.rolling(w).mean().shift(BASE_LAG)
        X[cs] = r.rolling(w).std().shift(BASE_LAG)
        groups["control"] += [cm, cs]
    rsi, macd = _rsi_macd(close)
    X["rsi"] = rsi.shift(BASE_LAG)
    X["macd"] = macd.shift(BASE_LAG)
    groups["control"] += ["rsi", "macd"]

    # --- fractional-differencing memory (d* fitted on dev only; FFD transform is causal) ---
    fit_series = logp[dev_mask.to_numpy()] if dev_mask is not None else logp
    d_star, _ = min_ffd_d(fit_series.dropna())
    X["ffd"] = frac_diff_ffd(logp, float(d_star)).shift(BASE_LAG)
    groups["memory"].append("ffd")

    # --- deterministic cyclical time (NOT shifted: known at t) ---
    tf = add_cyclical_time_features(pd.DataFrame(index=idx))
    for c in CYCLICAL_FEATURES:
        X[c] = tf[c]
        groups["time"].append(c)

    meta = {
        "groups": groups,
        "d_star": float(d_star),
        "arm_flow": groups["flow"] + groups["micro"] + groups["onchain"] + groups["time"],
        "arm_control": groups["control"] + groups["memory"] + groups["time"],
    }
    return X, meta


def align_xy(X: pd.DataFrame, labels: Labels, cols: list[str] | None = None):
    """Align features to the label entry timestamps. Returns ``(X_aligned, y, weight)``.

    NaNs are kept (LightGBM handles them natively; linear models get train-only imputation
    via ``walk_forward.preprocess_fold`` inside the fold loop).
    """
    f = labels.frame
    use = cols if cols is not None else list(X.columns)
    Xa = X.reindex(f.index)[use]
    return Xa, f["label"].astype(int), f["sample_weight"].astype(float)


def stable_select(frame: pd.DataFrame, feature_cols, target_col: str,
                  splits, k: int, method: str = "mda") -> list[str]:
    """Leakage-safe stable feature selection (median importance on dev folds + CFI).

    Thin wrapper over ``walk_forward.select_stable_features`` — importance is computed on the
    first dev folds' TRAIN data only, then the set is fixed for the whole walk-forward.
    """
    return select_stable_features(frame, feature_cols, target_col, splits, k=k, method=method)


# re-export for the M3 loop
__all__ = ["build_features", "align_xy", "stable_select", "cluster_collinear"]
