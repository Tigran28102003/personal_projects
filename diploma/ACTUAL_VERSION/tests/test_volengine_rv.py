"""Unit tests for the Vol-Engine RV construction (v2_volengine/src/rv.py) — A-M0.

Hand-built hourly OHLC with a known intraday path verifies close-RV, the next-day target shift,
the holdout split, and the short-day handling.
"""

import numpy as np
import pandas as pd
import pytest

from v2_volengine.src import rv as rvmod, config


def _two_day_hourly():
    # 2 UTC days × 24 hourly bars; close follows a tiny deterministic walk.
    idx = pd.date_range("2021-01-01 00:00", periods=48, freq="h", name="timestamp")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 48)))
    df = pd.DataFrame({
        "open": close * (1 - 1e-4), "high": close * (1 + 5e-4),
        "low": close * (1 - 5e-4), "close": close, "volume": 1.0,
    }, index=idx)
    return df


def test_close_rv_matches_manual_sum_of_squares():
    df = _two_day_hourly()
    m = rvmod.realized_measures(df, short_day_policy="keep")
    r = np.log(df["close"]).diff()
    day = df.index.normalize()
    expected_day2 = float(np.nansum(r[day == day[24]].values ** 2))
    assert m["rv_close"].iloc[1] == pytest.approx(expected_day2, rel=1e-9)
    assert (m["n_bars"] == 24).all()


def test_range_measures_positive_and_present():
    df = _two_day_hourly()
    m = rvmod.realized_measures(df, short_day_policy="keep")
    assert (m["rv_gk"] > 0).all() and (m["rv_rs"] > 0).all()
    assert {"rq", "bpv", "jump", "rv_plus", "rv_minus"} <= set(m.columns)


def test_target_next_is_one_day_ahead():
    df = _two_day_hourly()
    f = rvmod.build_rv(df, estimator="close", short_day_policy="keep")
    # target_next at day t equals target at day t+1; last day NaN
    assert f["target_next"].iloc[0] == pytest.approx(f["target"].iloc[1])
    assert np.isnan(f["target_next"].iloc[-1])


def test_log_target_roundtrip():
    rv = pd.Series([1e-4, 4e-4, 9e-4])
    t = rvmod.to_target(rv, "log")
    np.testing.assert_allclose(rvmod.from_target(t, "log"), rv.values, rtol=1e-9)


def test_short_day_scaling_inflates_partial_day():
    df = _two_day_hourly()
    df_short = df.drop(df.index[24:36])  # day 2 loses 12 of 24 bars
    kept = rvmod.realized_measures(df_short, short_day_policy="keep")
    scaled = rvmod.realized_measures(df_short, short_day_policy="scale")
    # day 2 has 12 bars → scale factor 24/12 = 2 on the sum-of-squares measure
    assert scaled["rv_close"].iloc[1] == pytest.approx(kept["rv_close"].iloc[1] * 2.0, rel=1e-9)


def test_seal_holdout_partition():
    idx = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    dev, hold = rvmod.seal_holdout(idx, cut="2025-06-01")
    assert (dev.values & hold.values).sum() == 0 and (dev.values | hold.values).all()
    assert idx[hold.values].min() >= pd.Timestamp("2025-06-01")
