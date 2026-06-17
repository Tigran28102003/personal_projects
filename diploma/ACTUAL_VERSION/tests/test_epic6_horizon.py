"""EPIC 6 — окна и горизонт: forward-метка, reconstruction round-trip, embargo-trim."""

import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from walk_forward import (
    forward_log_return, expanding_window_splits, rolling_window_splits,
    run_walk_forward,
)


# --------------------------------------------------------------------------
# forward_log_return (T6.3)
# --------------------------------------------------------------------------

def test_forward_log_return_h1_equals_diff():
    price = pd.Series([100.0, 101, 103, 102, 105])
    pd.testing.assert_series_equal(forward_log_return(price, 1), np.log(price).diff())


def test_forward_log_return_h_sum_and_reconstruction():
    price = pd.Series([100.0, 101, 103, 102, 105, 110, 108])
    h = 3
    fr = forward_log_return(price, h)
    lp = np.log(price)
    # метка в t = log(P_{t+h-1}) - log(P_{t-1}) = сумма h forward-доходностей
    for t in range(1, len(price) - (h - 1)):
        assert np.isclose(fr.iloc[t], lp.iloc[t + h - 1] - lp.iloc[t - 1])
    # последние h-1 строк -> NaN (нет будущего)
    assert fr.iloc[-(h - 1):].isna().all()
    # reconstruction: P_{t-1} * exp(R_t^{(h)}) == P_{t+h-1}
    t = 2
    assert np.isclose(price.iloc[t - 1] * np.exp(fr.iloc[t]), price.iloc[t + h - 1])


def test_forward_log_return_no_leakage_vs_lagged_returns():
    # forward-метка в t не должна совпадать с лагированной доходностью r_{t-1}
    rng = np.random.default_rng(0)
    price = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, size=60)))
    r = np.log(price).diff()
    fr = forward_log_return(price, 3)
    lag1 = r.shift(1)   # признак ret_lag1
    common = fr.dropna().index.intersection(lag1.dropna().index)
    # forward-метка и r_{t-1} не идентичны (нет тривиального перекрытия)
    assert not np.allclose(fr.loc[common], lag1.loc[common])


def test_forward_log_return_validates_horizon():
    with pytest.raises(ValueError):
        forward_log_return(pd.Series([1.0, 2.0]), 0)


# --------------------------------------------------------------------------
# split generators
# --------------------------------------------------------------------------

def test_expanding_window_train_grows():
    splits = expanding_window_splits(1000, n_splits=5, min_train_frac=0.5)
    train_sizes = [len(tr) for tr, _ in splits]
    assert all(b > a for a, b in zip(train_sizes, train_sizes[1:]))  # строго растёт
    assert train_sizes[0] >= 500                                     # >= min_train_frac*n


def test_rolling_window_fixed_train_size():
    splits = rolling_window_splits(1000, train_size=365, test_size=21)
    assert all(len(tr) == 365 for tr, _ in splits)


# --------------------------------------------------------------------------
# run_walk_forward embargo trim for horizon > 1
# --------------------------------------------------------------------------

def _gb_factory():
    def f():
        return Pipeline([('model', HistGradientBoostingRegressor(random_state=0, max_iter=40))])
    return f


@pytest.fixture
def wf_df():
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.RangeIndex(n)
    f0 = rng.normal(size=n)
    return pd.DataFrame({
        'BTC': 0.01 * f0 + 0.002 * rng.normal(size=n),   # target = returns
        'f0': np.roll(f0, 1),
        'f1': rng.normal(size=n),
    }, index=idx)


def test_run_walk_forward_embargo_trim(wf_df):
    splits = rolling_window_splits(len(wf_df), train_size=80, test_size=20, n_splits=3)
    common = dict(model_family='GB', model_name='HistGB', df=wf_df, target_col='BTC',
                  all_feature_cols=['f0', 'f1'], splits=splits, top_k=2,
                  optuna_n_trials=1, gb_model_factory=_gb_factory(), seed=0, retune_every=1)
    m1, _ = run_walk_forward(horizon=1, **common)
    m3, _ = run_walk_forward(horizon=3, **common)
    # при h=3 из каждого train-фолда отброшены последние 2 строки (embargo = h-1)
    assert np.all(m1['n_train'].to_numpy() - m3['n_train'].to_numpy() == 2)


def test_run_walk_forward_horizon1_unchanged_n_train(wf_df):
    splits = rolling_window_splits(len(wf_df), train_size=80, test_size=20, n_splits=3)
    m1, _ = run_walk_forward(
        model_family='GB', model_name='HistGB', df=wf_df, target_col='BTC',
        all_feature_cols=['f0', 'f1'], splits=splits, top_k=2, optuna_n_trials=1,
        gb_model_factory=_gb_factory(), seed=0, retune_every=1, horizon=1)
    assert (m1['n_train'] == 80).all()   # h=1 -> без обрезки
