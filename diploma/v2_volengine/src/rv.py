"""Realized-volatility target construction for the Vol-Engine (A-M0).

From the long spot-OHLC snapshot we build daily realized measures and the 1-day-ahead target
(log-RV by default). Estimators (R2 — measurement error):
* ``close``  — RV = Σ intraday (Δlog close)²  (classic, but ≈12× noisier at hourly than 5-min);
* ``range``  — Σ intraday Garman-Klass / Rogers-Satchell over hourly OHLC (better-measured on the
  same low-frequency data when high/low are available — they are, from data_spot).

Auxiliary realized measures for the HAR family: realized quarticity (HARQ), bipower variation &
jump (HAR-J), signed/semi-RV. Cross-day return convention is 24/7 (the first hourly return of a
UTC day is 23:00→00:00 of the prior day — produced naturally by ``log(close).diff()`` over the
continuous series; no overnight gap). Short days (missing hours, R4) are rescaled by
``BARS_PER_DAY / n_bars`` by default and flagged; the bar count is kept for transparency.

The target is for ``RV_{t+1}`` (next day) — labels are built strictly from intraday data up to the
end of day t, so there is no future intraday bleed (see tripwires_vol).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_LOG2 = np.log(2.0)


def realized_measures(spot: pd.DataFrame, short_day_policy: str = "scale") -> pd.DataFrame:
    """Daily realized measures from hourly OHLC. Index = UTC date.

    Columns: rv_close, rv_gk, rv_rs, rq (realized quarticity), bpv, jump, rv_plus, rv_minus, n_bars.
    ``short_day_policy``: 'scale' (×BARS_PER_DAY/n_bars on Σ-over-bar measures; default), 'drop', 'keep'.
    """
    o, h, l, c = (spot[x].astype(float) for x in ("open", "high", "low", "close"))
    r = np.log(c).diff()                                   # 24/7 hourly log-return (cross-day incl.)
    day = spot.index.normalize()

    # per-bar range measures (Garman-Klass, Rogers-Satchell)
    gk_bar = 0.5 * np.log(h / l) ** 2 - (2 * _LOG2 - 1) * np.log(c / o) ** 2
    rs_bar = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

    g_r = r.groupby(day)
    out = pd.DataFrame(index=pd.DatetimeIndex(sorted(set(day)), name="date"))
    out["n_bars"] = c.groupby(day).size()
    out["rv_close"] = g_r.apply(lambda x: np.nansum(x.values ** 2))
    out["rv_gk"] = gk_bar.groupby(day).sum()
    out["rv_rs"] = rs_bar.groupby(day).sum()
    out["rq"] = g_r.apply(lambda x: (np.isfinite(x).sum() / 3.0) * np.nansum(x.values ** 4))
    out["bpv"] = g_r.apply(_bipower)
    out["jump"] = (out["rv_close"] - out["bpv"]).clip(lower=0.0)
    out["rv_plus"] = g_r.apply(lambda x: np.nansum(np.where(x.values > 0, x.values, 0.0) ** 2))
    out["rv_minus"] = g_r.apply(lambda x: np.nansum(np.where(x.values < 0, x.values, 0.0) ** 2))

    sum_cols = ["rv_close", "rv_gk", "rv_rs", "rq", "bpv", "jump", "rv_plus", "rv_minus"]
    if short_day_policy == "drop":
        out = out[out["n_bars"] >= config.BARS_PER_DAY]
    elif short_day_policy == "scale":
        factor = (config.BARS_PER_DAY / out["n_bars"]).clip(upper=config.BARS_PER_DAY)
        out.loc[out["n_bars"] < config.BARS_PER_DAY, sum_cols] = \
            out.loc[out["n_bars"] < config.BARS_PER_DAY, sum_cols].mul(
                factor[out["n_bars"] < config.BARS_PER_DAY], axis=0)
    return out


def _bipower(x: pd.Series) -> float:
    v = np.abs(x.values)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan
    return (np.pi / 2.0) * np.sum(v[1:] * v[:-1])


def to_target(rv: pd.Series, transform: str = config.TARGET_TRANSFORM) -> pd.Series:
    rv = rv.clip(lower=config.RV_FLOOR)
    return np.log(rv) if transform == "log" else np.sqrt(rv)


def from_target(t: pd.Series | np.ndarray, transform: str = config.TARGET_TRANSFORM):
    return np.exp(t) if transform == "log" else np.square(t)


def build_rv(spot: pd.DataFrame, estimator: str = config.RV_ESTIMATOR,
             short_day_policy: str = "scale") -> pd.DataFrame:
    """Daily RV frame with the chosen estimator as ``rv`` and the next-day ``target`` (e.g. log-RV).

    Adds ``target_next`` = target(rv) shifted -1 (predict RV_{t+1}); the last day has NaN target_next.
    """
    m = realized_measures(spot, short_day_policy=short_day_policy)
    rv_col = {"close": "rv_close", "range": "rv_gk" if config.RV_RANGE_METHOD == "garman_klass" else "rv_rs"}[estimator]
    m["rv"] = m[rv_col].clip(lower=config.RV_FLOOR)
    m["target"] = to_target(m["rv"])                      # contemporaneous target of day t
    m["target_next"] = m["target"].shift(-1)             # what we predict at day t (RV_{t+1})
    return m


def seal_holdout(index: pd.Index, cut: str = config.HOLDOUT_CUT):
    """(dev_mask, holdout_mask): holdout = index >= cut (last ~12 months), sealed until the end."""
    cut_ts = pd.Timestamp(cut)
    dev = pd.Series(index < cut_ts, index=index, name="dev")
    hold = pd.Series(index >= cut_ts, index=index, name="holdout")
    return dev, hold


def _daily_rv_close(spot: pd.DataFrame) -> pd.Series:
    """Σ intraday (Δlog close)² per UTC day — no short-day scaling (used for cross-frequency check)."""
    r = np.log(spot["close"].astype(float)).diff()
    return (r ** 2).groupby(spot.index.normalize()).sum()


def cross_check_5min(start, end, hourly_spot: pd.DataFrame | None = None) -> dict:
    """Measurement-error check (R2): daily RV from hourly vs from 5-min over a window.

    Fetches Binance spot 5-min OHLC (network) and compares daily close-RV against the hourly series
    on the intersection of days. Hourly RV is noisier → wider hourly/5-min ratio dispersion at
    similar mean and high log-RV correlation. Returns a small dict (no files written).
    """
    from .data_spot import fetch_spot_ohlc, load_spot_snapshot
    if hourly_spot is None:
        hourly_spot = load_spot_snapshot()
    five = fetch_spot_ohlc(interval="5m", start=start, end=end)
    hourly = hourly_spot.loc[str(start):str(end)]
    rv5, rvh = _daily_rv_close(five), _daily_rv_close(hourly)
    # keep only FULL days in both series (else partial boundary days create spurious ratios)
    n5 = five.groupby(five.index.normalize()).size()
    nh = hourly.groupby(hourly.index.normalize()).size()
    full = n5.index[(n5 >= 288) & (nh.reindex(n5.index, fill_value=0) >= config.BARS_PER_DAY)]
    idx = rv5.index.intersection(rvh.index).intersection(full)
    a = rvh.reindex(idx).clip(lower=config.RV_FLOOR)
    b = rv5.reindex(idx).clip(lower=config.RV_FLOOR)
    ratio = (a / b).replace([np.inf, -np.inf], np.nan).dropna()
    corr = float(np.corrcoef(np.log(a), np.log(b))[0, 1]) if len(idx) > 2 else float("nan")
    q1, q3 = (float(ratio.quantile(0.25)), float(ratio.quantile(0.75))) if len(ratio) else (np.nan, np.nan)
    return {
        "n_full_days": int(len(idx)),
        "logrv_corr_hourly_vs_5min": corr,
        "ratio_hourly_over_5min_median": float(ratio.median()) if len(ratio) else float("nan"),
        "ratio_iqr": [q1, q3],
        "note": "hourly RV vs 5-min RV on full days; ratio≠1 + dispersion quantify hourly measurement error (R2)",
    }


def rv_report(frame: pd.DataFrame) -> dict:
    """Persistence, jump share, short-day share, target stats + the hourly measurement-error caveat."""
    lr = np.log(frame["rv"].clip(lower=config.RV_FLOOR)).dropna()
    acf1 = float(lr.autocorr(lag=1)) if len(lr) > 2 else float("nan")
    return {
        "n_days": int(len(frame)),
        "short_day_frac": float((frame["n_bars"] < config.BARS_PER_DAY).mean()),
        "logRV_acf1": acf1,                               # persistence (cluster) — the source of predictability
        "jump_frac_of_rv": float((frame["jump"] / frame["rv"].clip(lower=config.RV_FLOOR)).mean()),
        "logRV_mean": float(lr.mean()), "logRV_std": float(lr.std()),
        "estimator": config.RV_ESTIMATOR, "transform": config.TARGET_TRANSFORM,
        "measurement_note": ("hourly RV ≈12× noisier than 5-min → temper R² vs literature ~0.5; "
                             "HARQ central, range estimators preferred (R2)"),
    }
