"""Unit tests for the direction-C triple-barrier labeling (v2_microstructure/src/labeling.py).

Verifies first-touch correctness on hand-built paths, the no-future-leak property of the
trailing sigma, average-uniqueness / effective sample on overlapping events, and the unified
sample-weight construction.
"""

import numpy as np
import pandas as pd
import pytest

from v2_microstructure.src import labeling as lb


def _series(vals, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="h")
    return pd.Series(np.asarray(vals, dtype=float), index=idx, name="close")


def test_upper_barrier_first_gives_plus_one():
    # Flat then a sharp jump up well beyond k*sigma within H.
    close = _series([100, 100, 100, 100, 100, 100, 100, 100, 110, 100, 100, 100])
    out = lb.triple_barrier_labels(close, k=1.0, H=4, sigma_span=3).frame
    # entry at index 7 (value 100) sees +10% at next bar -> +1, bars_held=1
    row = out.loc[close.index[7]]
    assert row["label"] == 1
    assert row["bars_held"] == 1
    assert row["t1"] == close.index[8]
    assert row["ret_log"] > 0


def test_lower_barrier_first_gives_minus_one():
    close = _series([100, 100, 100, 100, 100, 100, 100, 100, 90, 100, 100, 100])
    out = lb.triple_barrier_labels(close, k=1.0, H=4, sigma_span=3).frame
    row = out.loc[close.index[7]]
    assert row["label"] == -1
    assert row["bars_held"] == 1
    assert row["ret_log"] < 0


def test_vertical_barrier_gives_zero():
    # Tiny oscillations that never reach k*sigma -> vertical hit -> label 0, held = H.
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.normal(0, 0.01, 60))  # very small drift/noise
    close = _series(base)
    out = lb.triple_barrier_labels(close, k=5.0, H=6, sigma_span=10).frame
    # with k=5 the barriers are very wide; most labels should be vertical (0) with held==H
    zeros = out[out["label"] == 0]
    assert len(zeros) > 0
    assert (zeros["bars_held"] == 6).all()


def test_last_H_rows_unlabeled():
    close = _series(list(range(100, 130)))  # 30 bars
    H = 5
    out = lb.triple_barrier_labels(close, k=1.0, H=H, sigma_span=4).frame
    # no valid entry within the last H bars (need a full forward window)
    assert out.index.max() <= close.index[len(close) - 1 - H]


def test_trailing_sigma_is_causal():
    # sigma_t must not change when FUTURE values change (no look-ahead).
    vals = list(100 + np.cumsum(np.random.default_rng(1).normal(0, 0.5, 50)))
    close_a = _series(vals)
    vals_b = vals.copy()
    vals_b[40:] = [v * 3 for v in vals_b[40:]]  # perturb only the future (>=40)
    close_b = _series(vals_b)
    sa = lb.trailing_sigma(close_a, span=10)
    sb = lb.trailing_sigma(close_b, span=10)
    # sigma up to index 39 must be identical (depends only on the past)
    pd.testing.assert_series_equal(sa.iloc[:40], sb.iloc[:40])


def test_average_uniqueness_bounds_and_effective_sample():
    # Non-overlapping events -> uniqueness 1; heavy overlap -> uniqueness < 1.
    close = _series(100 + np.cumsum(np.random.default_rng(2).normal(0, 1.0, 200)))
    out = lb.triple_barrier_labels(close, k=1.5, H=10, sigma_span=20).frame
    u = out["avg_uniqueness"]
    assert (u > 0).all() and (u <= 1.0 + 1e-9).all()
    eff = u.sum()
    assert 0 < eff <= len(out)  # effective sample never exceeds the number of events


def test_sample_weight_mean_normalised_and_nonnegative():
    close = _series(100 + np.cumsum(np.random.default_rng(3).normal(0, 1.0, 300)))
    L = lb.triple_barrier_labels(close, k=1.5, H=12, sigma_span=20)
    w = L.sample_weight
    assert (w >= 0).all()
    assert not w.isna().any()
    assert w.mean() == pytest.approx(1.0, rel=1e-6)


def test_seal_holdout_partition_is_disjoint_and_complete():
    idx = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    dev, hold = lb.seal_holdout(idx, cut="2025-06-01")
    assert (dev.values & hold.values).sum() == 0          # disjoint
    assert (dev.values | hold.values).all()               # complete
    assert idx[hold.values].min() >= pd.Timestamp("2025-06-01")
