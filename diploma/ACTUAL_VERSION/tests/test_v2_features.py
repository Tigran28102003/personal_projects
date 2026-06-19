"""Unit tests for the direction-C feature assembly (v2_microstructure/src/features.py).

Verifies the causal placement of every feature (value at t depends only on bars ≤ t-1),
the extra on-chain publication lag, the non-shifting of deterministic time features, the
two-arm partition, and align_xy.
"""

import numpy as np
import pandas as pd

from v2_microstructure.src import features as ft, labeling as lb


def _synthetic_snapshot(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="h", name="timestamp")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame(index=idx)
    df["perp_close"] = close
    df["spot_close"] = close * (1 + rng.normal(0, 1e-4, n))
    df["perp_volume"] = rng.uniform(1, 10, n)
    # distinct, monotone series make the shift exactly testable
    df["funding"] = np.arange(n, dtype=float)
    df["funding_chg"] = rng.normal(0, 1, n)
    df["basis"] = rng.normal(0, 1e-3, n)
    df["basis_chg"] = rng.normal(0, 1e-4, n)
    df["tfi"] = rng.normal(0, 0.3, n)
    df["cvd"] = np.cumsum(rng.normal(0, 0.1, n))
    df["vpin"] = rng.uniform(0, 1, n)
    df["active_addr_chg"] = np.arange(n, dtype=float)   # monotone -> testable extra lag
    for c in ("tx_cnt_chg", "addr_bal_chg", "hashrate_chg", "supply_chg"):
        df[c] = rng.normal(0, 1, n)
    return df


def test_flow_feature_is_lagged_one_bar():
    snap = _synthetic_snapshot()
    X, meta = ft.build_features(snap, dev_mask=pd.Series(True, index=snap.index))
    # funding == arange, so X['funding'][t] must equal t-1
    assert X["funding"].iloc[10] == 9.0
    assert np.isnan(X["funding"].iloc[0])  # first row has no t-1


def test_onchain_has_extra_publication_lag():
    snap = _synthetic_snapshot()
    X, meta = ft.build_features(snap, dev_mask=pd.Series(True, index=snap.index))
    lag = ft.BASE_LAG + ft.ONCHAIN_PUB_LAG_BARS
    assert X["active_addr_chg"].iloc[40] == float(40 - lag)
    assert np.isnan(X["active_addr_chg"].iloc[lag - 1])


def test_time_features_not_shifted_and_deterministic():
    snap = _synthetic_snapshot()
    X, meta = ft.build_features(snap, dev_mask=pd.Series(True, index=snap.index))
    hours = pd.DatetimeIndex(X.index).hour
    expected = np.sin(2 * np.pi * hours / 24.0)
    np.testing.assert_allclose(X["hour_sin"].to_numpy(), expected, atol=1e-12)


def test_arms_partition_and_share_time():
    snap = _synthetic_snapshot()
    X, meta = ft.build_features(snap, dev_mask=pd.Series(True, index=snap.index))
    flow, ctrl, time = set(meta["arm_flow"]), set(meta["arm_control"]), set(meta["groups"]["time"])
    assert time <= flow and time <= ctrl                     # time shared by both
    assert (flow - time).isdisjoint(ctrl - time)             # flow vs control disjoint otherwise
    assert "funding" in flow and "funding" not in (ctrl - time)
    assert "rsi" in ctrl and "ffd" in ctrl                   # control owns price-only + memory


def test_d_star_in_unit_interval():
    snap = _synthetic_snapshot()
    _, meta = ft.build_features(snap, dev_mask=pd.Series(True, index=snap.index))
    assert 0.0 <= meta["d_star"] <= 1.0


def test_align_xy_shapes_and_weight_normalised():
    snap = _synthetic_snapshot(n=600)
    X, meta = ft.build_features(snap, dev_mask=pd.Series(True, index=snap.index))
    L = lb.triple_barrier_labels(snap["perp_close"], k=1.5, H=12, sigma_span=20)
    Xa, y, w = ft.align_xy(X, L, cols=meta["arm_flow"])
    assert len(Xa) == len(y) == len(w) == len(L.frame)
    assert list(Xa.columns) == meta["arm_flow"]
    assert abs(float(w.mean()) - 1.0) < 1e-6
