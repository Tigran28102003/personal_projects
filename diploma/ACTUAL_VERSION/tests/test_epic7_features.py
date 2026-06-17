"""EPIC 7 — признаки: time-of-day циклические, нелинейный отбор, frac-diff."""

import numpy as np
import pandas as pd
import pytest

from walk_forward import add_cyclical_time_features, select_top_k_features
from fracdiff import ffd_weights, frac_diff_ffd, min_ffd_d


# --------------------------------------------------------------------------
# T7.1 — time-of-day cyclical features
# --------------------------------------------------------------------------

def test_cyclical_features_bounded_and_no_nan():
    idx = pd.date_range('2021-01-01', periods=500, freq='5min')
    df = pd.DataFrame({'x': np.arange(500.0)}, index=idx)
    out = add_cyclical_time_features(df)
    for col in ('tod_sin', 'tod_cos', 'hour_sin', 'hour_cos',
                'dow_sin', 'dow_cos', 'month_sin', 'month_cos'):
        assert col in out.columns
        assert out[col].notna().all()
        assert out[col].between(-1.0 - 1e-9, 1.0 + 1e-9).all()


def test_cyclical_features_deterministic_no_shift():
    # значение детерминировано по timestamp (без лага): полночь -> tod_sin=0, tod_cos=1
    idx = pd.DatetimeIndex(['2021-03-15 00:00', '2021-03-15 06:00'])
    out = add_cyclical_time_features(pd.DataFrame({'x': [1.0, 2.0]}, index=idx))
    assert np.isclose(out['tod_sin'].iloc[0], 0.0) and np.isclose(out['tod_cos'].iloc[0], 1.0)
    # 06:00 -> minute_of_day=360 -> sin(2π*360/1440)=sin(π/2)=1
    assert np.isclose(out['tod_sin'].iloc[1], 1.0, atol=1e-9)


# --------------------------------------------------------------------------
# T7.2 — nonlinear feature selection
# --------------------------------------------------------------------------

@pytest.fixture
def nonlinear_xy():
    rng = np.random.default_rng(0)
    n = 400
    good = rng.uniform(-1, 1, size=n)
    y = pd.Series(good ** 2 + 0.05 * rng.normal(size=n))   # квадратично -> Pearson~0, MI высок
    X = pd.DataFrame({'good': good, 'n1': rng.normal(size=n),
                      'n2': rng.normal(size=n), 'n3': rng.normal(size=n)})
    return X, y


def test_select_pearson_default_runs(nonlinear_xy):
    X, y = nonlinear_xy
    sel = select_top_k_features(X, y, k=2)         # default 'pearson'
    assert len(sel) == 2


@pytest.mark.parametrize('method', ['mutual_info', 'model_gain', 'mda'])
def test_select_nonlinear_finds_informative(nonlinear_xy, method):
    X, y = nonlinear_xy
    sel = select_top_k_features(X, y, k=2, method=method)
    assert 'good' in sel                            # нелинейный отбор ловит квадратичную связь


def test_select_unknown_method_raises(nonlinear_xy):
    X, y = nonlinear_xy
    with pytest.raises(ValueError):
        select_top_k_features(X, y, k=2, method='bogus')


# --------------------------------------------------------------------------
# T7.3 — fractional differentiation
# --------------------------------------------------------------------------

def test_ffd_weights_recursion():
    d = 0.4
    w = ffd_weights(d, thresh=1e-4)
    assert w[0] == 1.0
    assert np.isclose(w[1], -d)                     # w_1 = -w_0*(d-0)/1 = -d
    # рекурсия w_k = -w_{k-1}*(d-k+1)/k
    for k in range(2, len(w)):
        assert np.isclose(w[k], -w[k - 1] * (d - k + 1) / k)


def test_ffd_d0_is_identity_and_d1_is_diff():
    s = pd.Series(np.cumsum(np.random.default_rng(1).normal(size=200)) + 100.0)
    pd.testing.assert_series_equal(frac_diff_ffd(s, 0.0), s.astype(float).copy())
    fd1 = frac_diff_ffd(s, 1.0)
    # d=1 -> первая разность (с точностью до начального NaN)
    assert np.allclose(fd1.dropna().to_numpy(), s.diff().dropna().to_numpy()[-len(fd1.dropna()):])


def test_min_ffd_d_yields_stationary_series():
    from fracdiff import _adf_pvalue
    rng = np.random.default_rng(2)
    price = pd.Series(np.cumsum(rng.normal(size=600)) + 1000.0)   # random walk (нестационарна)
    d_star, fd = min_ffd_d(price, adf_pvalue=0.05)
    assert 0.0 < d_star <= 1.0
    assert _adf_pvalue(fd) < 0.05                    # выбранный ряд проходит ADF (стационарен)
