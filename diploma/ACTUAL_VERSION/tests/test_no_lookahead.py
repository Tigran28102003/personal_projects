"""Тесты отсутствия look-ahead в лагировании экзогенных рядов (EPIC 1, T1.2).

Две группы проверок:
  1. `lag_exogenous` действительно применяет сдвиг — значение exog в строке t
     совпадает с сырым значением в t−lag; BTC/календарь не тронуты; FRED получает
     publication-lag сверх базового.
  2. Грубый детектор синхронной утечки: corr(exog[t], r_t) не должна быть
     аномально выше corr(exog[t], r_{t+1}). Проверяем, что детектор флагует
     синхронно-утекающий ряд и пропускает корректно лагированный.

Запуск: `pytest -q`.
"""

import numpy as np
import pandas as pd
import pytest

import get_data as G


# ==================== 1. Корректность сдвига ====================

def test_lag_exogenous_base_shift_applied():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    raw = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    df = pd.DataFrame({"ETH": raw, "BTC": np.arange(6.0)}, index=idx)
    out = G.lag_exogenous(df, ["ETH"], base_lag=1)
    # exog[t] == raw[t-1]
    assert np.isnan(out["ETH"].iloc[0])
    for t in range(1, 6):
        assert out["ETH"].iloc[t] == raw[t - 1]


def test_lag_exogenous_publication_lag_for_fred():
    idx = pd.date_range("2020-01-01", periods=8, freq="D")
    # CPI наблюдается раз в "период", между наблюдениями NaN -> ffill -> shift
    cpi = np.array([100.0, np.nan, np.nan, np.nan, 104.0, np.nan, np.nan, np.nan])
    df = pd.DataFrame({"CPI": cpi, "BTC": np.arange(8.0)}, index=idx)
    out = G.lag_exogenous(df, ["CPI"], pub_lag_bars={"CPI": 2}, base_lag=1)
    # ffill: 100,100,100,100,104,104,104,104 ; shift на 1+2=3
    expected = np.array([np.nan, np.nan, np.nan, 100, 100, 100, 100, 104])
    np.testing.assert_array_equal(out["CPI"].values, expected)


def test_lag_exogenous_does_not_touch_excluded():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "ETH": np.arange(5.0),
        "BTC": np.arange(10.0, 15.0),
        "RSI": np.arange(20.0, 25.0),
        "Feer_Greed_value": np.arange(30.0, 35.0),
        "day": [1, 2, 3, 4, 5],
    }, index=idx)
    not_exog = {"BTC", "RSI", "MACD", "ATR", "OBV", "Feer_Greed_value", "day"}
    exog = [c for c in df.columns if c not in not_exog]
    out = G.lag_exogenous(df, exog, base_lag=1)
    # ровно ETH сдвинут; остальное идентично исходному
    assert exog == ["ETH"]
    pd.testing.assert_series_equal(out["BTC"], df["BTC"])
    pd.testing.assert_series_equal(out["RSI"], df["RSI"])
    pd.testing.assert_series_equal(out["Feer_Greed_value"], df["Feer_Greed_value"])
    pd.testing.assert_series_equal(out["day"], df["day"])


def test_lag_exogenous_ffill_no_future_leak():
    # ffill смотрит только назад: дырка между наблюдениями заполняется ПРОШЛЫМ,
    # а не будущим значением.
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    s = np.array([1.0, np.nan, np.nan, 9.0, np.nan])
    df = pd.DataFrame({"X": s, "BTC": np.arange(5.0)}, index=idx)
    out = G.lag_exogenous(df, ["X"], base_lag=1)
    # ffill: 1,1,1,9,9 ; shift1: nan,1,1,1,9 — значение 9 не «протекает» в строки до его появления
    expected = np.array([np.nan, 1, 1, 1, 9])
    np.testing.assert_array_equal(out["X"].values, expected)


def test_default_fred_pub_lag_covers_series():
    # publication-lag задан для всех FRED-рядов из конфигурации DataMaker
    dm = G.DataMaker(fred_api_key="x")
    missing = [name for name in dm.fredapi_series if name not in dm.fred_pub_lag]
    assert missing == [], f"нет publication-lag для: {missing}"


# ==================== 2. Детектор синхронной утечки ====================

def synchronous_leak_score(exog: np.ndarray, ret: np.ndarray) -> float:
    """|corr(exog[t], r_t)| − |corr(exog[t], r_{t+1})|.

    Корректно лагированный exog (известен к t−1) одинаково слабо коррелирует с
    r_t и r_{t+1} -> score≈0. Синхронно утекающий exog (содержит r_t) -> высокая
    corr с r_t и почти нулевая с r_{t+1} -> score≫0.
    """
    e = np.asarray(exog, float)
    r = np.asarray(ret, float)
    e_t = e[:-1]
    r_same = r[:-1]
    r_next = r[1:]
    m = np.isfinite(e_t) & np.isfinite(r_same) & np.isfinite(r_next)
    c_same = np.corrcoef(e_t[m], r_same[m])[0, 1]
    c_next = np.corrcoef(e_t[m], r_next[m])[0, 1]
    return float(abs(c_same) - abs(c_next))


@pytest.fixture
def returns():
    rng = np.random.default_rng(7)
    return rng.normal(scale=0.02, size=4000)


def test_detector_flags_synchronous_leak(returns):
    rng = np.random.default_rng(11)
    leaky = returns + rng.normal(scale=0.002, size=returns.size)  # содержит r_t
    assert synchronous_leak_score(leaky, returns) > 0.3


def test_detector_passes_properly_lagged(returns):
    # exog[t] = r_{t-1} (строго прошлое) -> синхронной утечки нет
    lagged = np.concatenate([[np.nan], returns[:-1]])
    assert synchronous_leak_score(lagged, returns) < 0.1


def test_lag_exogenous_removes_synchronous_leak(returns):
    # строим синхронно-утекающий exog, лагируем через lag_exogenous и убеждаемся,
    # что после фикса синхронная утечка исчезает
    rng = np.random.default_rng(3)
    leaky = returns + rng.normal(scale=0.002, size=returns.size)
    idx = pd.date_range("2015-01-01", periods=returns.size, freq="D")
    df = pd.DataFrame({"leaky": leaky, "BTC": np.cumsum(returns)}, index=idx)
    before = synchronous_leak_score(df["leaky"].values, returns)
    out = G.lag_exogenous(df, ["leaky"], base_lag=1)
    after = synchronous_leak_score(out["leaky"].values, returns)
    assert before > 0.3
    assert after < 0.1
