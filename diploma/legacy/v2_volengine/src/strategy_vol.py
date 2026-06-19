"""Layer-2 vol-timing overlay (A-M4) on the validated HARQ h=1 RV forecast.

Sizing: ``position = min(cap, target_vol_ann / forecast_vol_ann)``, long-flat ∈ [0, cap], **no short**,
cap = 1 (no leverage), **daily rebalance**. The forecast is the *sanitised* (clipped) HARQ forecast
(vol_model arm), so a tiny forecast cannot blow up sizing. ``target_vol_ann`` is pre-registered in
config (fixed at M0), NOT tuned to maximise Sharpe (that would be forking).

Because long-flat-0..1 only **de-risks** in high vol (never levers up), the honest question is whether
risk-adjusted return improves vs buy&hold — not whether raw return rises (it generally won't, since
average exposure ≤ 1). Costs are realistic perp costs: taker fee + slippage on **turnover** (routed
through the legacy ``backtest.simulate_strategy`` continuous mode — no legacy edits) plus **funding
carry** on the held leg (a thin extension on top, not a legacy change). Dev period only; holdout sealed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, har as harmod, vol_model as vm

# legacy/v2 infra (config put it on sys.path)
from walk_forward import expanding_window_splits  # noqa: E402
from backtest import simulate_strategy, buy_and_hold, round_trip_cost  # noqa: E402


def harq_rv_forecast(rv_frame: pd.DataFrame, n_splits: int = config.VOL_TIMING_WF_SPLITS,
                     embargo: int = 5, min_train_frac: float = config.VOL_TIMING_MIN_TRAIN_FRAC) -> pd.DataFrame:
    """Walk-forward OOF HARQ RV-level forecast (clipped + Jensen) on ``rv_frame`` (rv.build_rv).

    Reuses vol_model._fit_predict_arm('HARQ', …) so the forecast feeding Layer-2 is the *same
    sanitised* HARQ used by Gate 1. A smaller warmup (``min_train_frac``) + finer refits widen the
    tradeable dev window across more vol regimes. Returns a frame indexed by date (``y_rv``, ``rv_pred``)."""
    X = harmod.har_components(rv_frame).dropna(subset=["har_m", "target"])
    y = X["target"]
    n = len(X)
    rv_pred = np.full(n, np.nan)
    for tr, te in expanding_window_splits(n, n_splits=n_splits, min_train_frac=min_train_frac):
        tr = tr[tr < te[0] - embargo]
        if len(tr) < config.HAR_WINDOWS[2] + 10:
            continue
        pred, s2 = vm._fit_predict_arm("HARQ", X.iloc[tr], y.iloc[tr], X.iloc[te], None)
        rv_pred[te] = np.exp(pred + 0.5 * s2)
    out = pd.DataFrame({"y_rv": np.exp(y.to_numpy()), "rv_pred": rv_pred}, index=X.index)
    return out.dropna()


def daily_funding(exo_hourly: pd.DataFrame) -> pd.Series:
    """Daily long-funding rate from the hourly C-snapshot: sum of the 3 daily 8h settlement rates."""
    f8 = exo_hourly["funding"].resample("8h").last()
    return f8.groupby(f8.index.normalize()).sum().rename("funding_daily")


def vol_timing_positions(fc: pd.DataFrame, target_vol_ann: float = config.VOL_TARGET_ANN,
                         cap: float = config.VOL_TIMING_CAP) -> pd.Series:
    """Inverse-vol sizing decided at day t (from the forecast of RV_{t+1}), in [0, cap]."""
    fvol_ann = np.sqrt(fc["rv_pred"].clip(lower=config.RV_FLOOR)) * np.sqrt(config.DAYS_PER_YEAR)
    pos = np.minimum(cap, target_vol_ann / fvol_ann)
    return pos.clip(lower=0.0).rename("size_raw")


def run_overlay(rv_frame: pd.DataFrame, perp_close_daily: pd.Series, exo_hourly: pd.DataFrame,
                target_vol_ann: float = config.VOL_TARGET_ANN,
                fee_rate: float = config.FEE_RATE, slippage_bps: float = config.SLIPPAGE_BPS,
                cap: float = config.VOL_TIMING_CAP) -> pd.DataFrame:
    """Run the vol-timing overlay through legacy simulate_strategy + a funding-carry extension.

    Returns a daily frame: ``pos`` (held), ``ret`` (perp), ``ret_vt_price`` (price PnL net of
    fee/slippage on turnover), ``funding_cost``, ``ret_vt`` (net incl. funding), ``ret_bh``,
    ``equity_vt`` / ``equity_bh`` (compounded), ``turnover``."""
    fc = harq_rv_forecast(rv_frame)
    size_raw = vol_timing_positions(fc, target_vol_ann, cap)
    size_hold = size_raw.shift(1)                                   # decided at t−1, held into day t
    fund = daily_funding(exo_hourly)

    idx = fc.index.intersection(perp_close_daily.index)
    df = pd.DataFrame(index=idx)
    df["actual"] = perp_close_daily.reindex(idx).astype(float)
    df["size"] = size_hold.reindex(idx)
    df = df.dropna(subset=["actual", "size"])
    df["timestamp"] = df.index
    df["predicted"] = df["actual"]                                  # unused in continuous mode
    df["model"] = "vol_timing"

    sim = simulate_strategy(df, fee_rate=fee_rate, slippage_bps=slippage_bps,
                            size_col="size", cap=cap, allow_short=False)
    bh = buy_and_hold(df)
    sim = sim.set_index("timestamp")
    bh = bh.set_index("timestamp")

    fund_d = fund.reindex(sim.index).fillna(0.0)
    out = pd.DataFrame(index=sim.index)
    out["pos"] = sim["position"]
    out["turnover"] = sim["trade"].abs()
    out["ret"] = bh["bh_ret_pct"]                                   # underlying perp daily return
    out["ret_vt_price"] = sim["ret_pct"]                            # price PnL net of turnover cost
    out["funding_cost"] = out["pos"] * fund_d                       # long pays positive funding
    out["ret_vt"] = out["ret_vt_price"] - out["funding_cost"]       # net incl. funding carry
    out["ret_bh"] = bh["bh_ret_pct"]
    out["equity_vt"] = (1.0 + out["ret_vt"]).cumprod()
    out["equity_bh"] = (1.0 + out["ret_bh"]).cumprod()
    return out


def constant_vol_target(out: pd.DataFrame) -> pd.Series:
    """Constant-exposure benchmark at the overlay's average exposure (Moreira–Muir matched-vol
    spirit, within the long-flat cap): same average notional, no timing. For Gate-2 robustness."""
    avg_pos = float(out["pos"].mean())
    return (avg_pos * out["ret"]).rename("ret_cvt")
