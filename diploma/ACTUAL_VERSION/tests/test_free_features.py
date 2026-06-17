"""T7.4 — бесплатные exogenous-фичи: офлайн-проверки билдеров + анти-утечка funding.

Сетевые вызовы НЕ делаются (кроме одного smoke, который сам пропускается без сети) —
билдеры тестируются на синтетике, чтобы suite был детерминирован и работал в CI.
"""

import numpy as np
import pandas as pd
import pytest

import free_features as ff


# --------------------------------------------------------------------------
# order-flow imbalance
# --------------------------------------------------------------------------

def _klines(volume, taker_buy):
    idx = pd.date_range('2021-01-01', periods=len(volume), freq='5min')
    return pd.DataFrame({
        'close': np.linspace(100, 110, len(volume)),
        'volume': np.asarray(volume, float),
        'quote_volume': np.asarray(volume, float) * 100,
        'n_trades': np.full(len(volume), 10.0),
        'taker_buy_base': np.asarray(taker_buy, float),
        'taker_buy_quote': np.asarray(taker_buy, float) * 100,
    }, index=idx)


def test_tfi_bounds():
    # вся покупка тейкером -> tfi=+1; продажа -> -1; пополам -> 0
    k = _klines([10, 10, 10], [10, 0, 5])
    of = ff.orderflow_features(k, vpin_window=2)
    assert np.isclose(of['tfi'].iloc[0], 1.0)
    assert np.isclose(of['tfi'].iloc[1], -1.0)
    assert np.isclose(of['tfi'].iloc[2], 0.0)
    assert of['vpin'].dropna().between(0, 1).all()


def test_orderflow_empty_returns_named_columns():
    of = ff.orderflow_features(pd.DataFrame())
    assert list(of.columns) == list(ff.ORDERFLOW_FEATURES)
    assert len(of) == 0


# --------------------------------------------------------------------------
# basis
# --------------------------------------------------------------------------

def test_basis_sign_and_zero():
    idx = pd.date_range('2021-01-01', periods=4, freq='h')
    spot = pd.Series([100, 100, 200, 200.0], index=idx)
    perp = pd.Series([100, 101, 200, 198.0], index=idx)   # +0%, +1%, 0%, -1%
    b = ff.basis_features(perp, spot)
    assert np.isclose(b['basis'].iloc[0], 0.0)
    assert np.isclose(b['basis'].iloc[1], 0.01)
    assert np.isclose(b['basis'].iloc[3], -0.01)


# --------------------------------------------------------------------------
# funding — выравнивание на сетку без утечки в будущее
# --------------------------------------------------------------------------

def test_funding_on_grid_no_future_leak():
    fund = pd.DataFrame({'funding': [0.0001, 0.0002, -0.0001]},
                        index=pd.to_datetime(['2021-01-01 00:00', '2021-01-01 08:00', '2021-01-01 16:00']))
    grid = pd.date_range('2020-12-31 23:00', '2021-01-01 18:00', freq='h')
    g = ff.funding_on_grid(fund, grid)
    # бар ДО первого сеттлмента -> NaN (будущее не подтекает)
    assert np.isnan(g.loc['2020-12-31 23:00', 'funding'])
    # после сеттлмента -> последнее известное значение (ffill)
    assert np.isclose(g.loc['2021-01-01 03:00', 'funding'], 0.0001)
    assert np.isclose(g.loc['2021-01-01 10:00', 'funding'], 0.0002)
    assert np.isclose(g.loc['2021-01-01 17:00', 'funding'], -0.0001)


def test_funding_empty_graceful():
    grid = pd.date_range('2021-01-01', periods=5, freq='h')
    g = ff.funding_on_grid(pd.DataFrame(columns=['funding']), grid)
    assert g['funding'].isna().all() and len(g) == 5


# --------------------------------------------------------------------------
# on-chain
# --------------------------------------------------------------------------

def test_onchain_activity_changes():
    idx = pd.date_range('2021-01-01', periods=3, freq='D')
    cm = pd.DataFrame({
        'AdrActCnt': [100.0, 110.0, 121.0],
        'TxCnt': [200.0, 210.0, 220.0],
        'AdrBalCnt': [1000.0, 1010.0, 1020.0],
        'HashRate': [1.0, 1.1, 1.2],
        'SplyCur': [18_000_000.0, 18_000_050.0, 18_000_100.0],
    }, index=idx)
    oc = ff.onchain_features(cm)
    assert np.isclose(oc['active_addr_chg'].iloc[1], 0.10)   # 110/100-1
    assert {'tx_cnt_chg', 'addr_bal_chg', 'hashrate_chg', 'supply_chg'} <= set(oc.columns)
    assert oc['supply_chg'].iloc[1] > 0                       # эмиссия растёт


# --------------------------------------------------------------------------
# attach_free_features — graceful degradation (без сети)
# --------------------------------------------------------------------------

def test_attach_graceful_when_fetch_empty(monkeypatch):
    monkeypatch.setattr(ff, 'fetch_klines', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(ff, 'fetch_funding', lambda *a, **k: pd.DataFrame(columns=['funding']))
    monkeypatch.setattr(ff, 'fetch_coinmetrics', lambda *a, **k: pd.DataFrame())
    frame = pd.DataFrame({'BTC': [1.0, 2, 3]}, index=pd.date_range('2021-01-01', periods=3, freq='D'))
    out = ff.attach_free_features(frame, 'daily', use_cache=False)
    assert list(out.columns) == ['BTC']      # ни одной all-NaN фичи не добавлено


def test_attach_unknown_freq_returns_frame():
    frame = pd.DataFrame({'BTC': [1.0, 2]}, index=pd.date_range('2021-01-01', periods=2, freq='D'))
    assert ff.attach_free_features(frame, 'weekly').equals(frame)


def test_attach_lags_features_at_source(monkeypatch):
    # анти-утечка: фичи лагируются на 1 бар (значение в t = состояние t-1)
    idx = pd.date_range('2021-01-01', periods=10, freq='D')
    frame = pd.DataFrame({'BTC': np.arange(10.0) + 100}, index=idx)
    kl = pd.DataFrame({'close': np.arange(10.0) + 100, 'volume': np.full(10, 10.0),
                       'quote_volume': np.full(10, 1000.0), 'n_trades': np.full(10, 5.0),
                       'taker_buy_base': np.full(10, 10.0),     # tfi = +1 каждый бар
                       'taker_buy_quote': np.full(10, 1000.0)}, index=idx)
    monkeypatch.setattr(ff, 'fetch_klines', lambda *a, **k: kl)
    monkeypatch.setattr(ff, 'fetch_funding', lambda *a, **k: pd.DataFrame(columns=['funding']))
    monkeypatch.setattr(ff, 'fetch_coinmetrics', lambda *a, **k: pd.DataFrame())
    out = ff.attach_free_features(frame, 'daily', use_cache=False, lag=1)
    assert 'tfi' in out.columns
    assert np.isnan(out['tfi'].iloc[0])             # первый бар -> NaN после лага
    assert np.isclose(out['tfi'].iloc[1], 1.0)      # дальше известное t-1 значение


# --------------------------------------------------------------------------
# сетевой smoke (сам пропускается, если нет доступа)
# --------------------------------------------------------------------------

def test_live_smoke_klines_and_funding():
    import datetime
    sess = ff._session()
    end = pd.Timestamp(datetime.datetime.utcnow())
    start = end - pd.Timedelta(days=2)
    try:
        k = ff.fetch_klines('BTCUSDT', '1d', start, end, 'spot', sess, use_cache=False)
    except Exception:
        pytest.skip("нет сетевого доступа к Binance")
    if k.empty:
        pytest.skip("Binance вернул пусто (сеть/лимит)")
    assert {'close', 'volume', 'taker_buy_base'} <= set(k.columns)
    of = ff.orderflow_features(k)
    assert of['tfi'].dropna().between(-1, 1).all()
