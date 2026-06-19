"""Unit tests for the Vol-Engine ML+exo feature matrix (A-M2)."""

import numpy as np
import pandas as pd
import pytest

from v2_volengine.src import features_vol as fv, config


def _synthetic_spot(start="2019-09-01", days=420, seed=0):
    rng = np.random.default_rng(seed)
    n = days * 24
    idx = pd.date_range(start, periods=n, freq="h", name="timestamp")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    return pd.DataFrame({"open": close * (1 - 1e-4), "high": close * (1 + 6e-4),
                         "low": close * (1 - 6e-4), "close": close, "volume": 1.0}, index=idx)


def _synthetic_exo(spot, seed=1):
    rng = np.random.default_rng(seed)
    idx = spot.index
    return pd.DataFrame({
        "funding": rng.normal(0, 1e-4, len(idx)), "funding_chg": rng.normal(0, 1e-5, len(idx)),
        "basis": rng.normal(0, 1e-3, len(idx)), "basis_chg": rng.normal(0, 1e-4, len(idx)),
        "tfi": rng.normal(0, 0.3, len(idx)), "cvd": np.cumsum(rng.normal(0, 0.1, len(idx))),
        "vpin": rng.uniform(0, 1, len(idx)),
        "active_addr_chg": rng.normal(0, 0.05, len(idx)), "tx_cnt_chg": rng.normal(0, 0.05, len(idx)),
        "addr_bal_chg": rng.normal(0, 0.01, len(idx)), "hashrate_chg": rng.normal(0, 0.02, len(idx)),
        "supply_chg": rng.normal(0, 1e-4, len(idx)),
    }, index=idx)


def test_exo_aggregation_is_lagged_one_day():
    spot = _synthetic_spot()
    exo = _synthetic_exo(spot)
    daily = fv.aggregate_exo_daily(exo)
    # first row is NaN (shift), and day t equals raw daily mean of day t-1
    assert daily["funding"].iloc[0] != daily["funding"].iloc[0] or np.isnan(daily["funding"].iloc[0])
    raw_mean = exo["funding"].groupby(exo.index.normalize()).mean()
    assert daily["funding"].iloc[5] == pytest.approx(raw_mean.iloc[4])


def test_matrix_perp_era_groups_and_target():
    spot = _synthetic_spot()
    X, groups = fv.build_feature_matrix(spot, estimator="range", exo_hourly=_synthetic_exo(spot))
    assert X.index.min() >= pd.Timestamp(config.PERP_ERA_START)
    assert set(groups["har"]) <= set(X.columns) and "target" in X.columns
    assert "funding" in groups["exo"] and "vpin" in groups["exo"]
    assert set(groups["all"]) == set(groups["har"] + groups["realized"] + groups["exo"])


def test_dev_matrix_has_no_nan_and_excludes_holdout():
    spot = _synthetic_spot(days=500)
    X, _ = fv.build_feature_matrix(spot, estimator="range", exo_hourly=_synthetic_exo(spot))
    Xd = fv.dev_matrix(X)
    assert not Xd.isna().any().any()
    assert Xd.index.max() < pd.Timestamp(config.HOLDOUT_CUT)
