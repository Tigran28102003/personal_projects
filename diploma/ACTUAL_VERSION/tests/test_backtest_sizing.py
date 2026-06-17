"""EPIC 5 — непрерывный сайзинг и экономические метрики."""

import numpy as np
import pandas as pd
import pytest

from backtest import (
    simulate_strategy, buy_and_hold, backtest_summary, position_size,
)


def _price_oof(prices, model='M', size=None):
    n = len(prices)
    d = {
        'timestamp': pd.RangeIndex(n),
        'actual': np.asarray(prices, dtype=float),
        'predicted': np.asarray(prices, dtype=float),
        'fold': np.zeros(n, dtype=int),
        'model': [model] * n,
    }
    if size is not None:
        d['size'] = np.asarray(size, dtype=float)
    return pd.DataFrame(d)


# --------------------------------------------------------------------------
# position_size helper
# --------------------------------------------------------------------------

def test_position_size_confidence_and_cap():
    # p_up=0.5 -> conf 0 -> size 0
    assert float(position_size(0.5, 0.01, 0.02)) == 0.0
    # высокая уверенность + сильный сигнал -> упирается в cap
    assert float(position_size(1.0, 1.0, 0.001, kappa=1.0, cap=1.0)) == 1.0
    # отрицательный сигнал при высокой уверенности -> short, клип к -cap
    assert float(position_size(1.0, -1.0, 0.001, cap=1.0)) == -1.0


def test_position_size_vectorised():
    p = np.array([0.5, 0.9, 0.1])
    r = np.array([0.01, 0.01, 0.01])
    s = np.array([0.02, 0.02, 0.02])
    out = position_size(p, r, s, cap=1.0)
    assert out.shape == (3,)
    assert out[0] == 0.0 and out[1] > 0 and out[2] < 0


# --------------------------------------------------------------------------
# continuous sizing vs binary
# --------------------------------------------------------------------------

def test_continuous_matches_binary_when_size_is_0_1():
    # МОНОТОННО растущая цена: бинарный сигнал входит в лонг и НИКОГДА не выходит
    # (predicted всегда > prev), поэтому равен постоянному size=1; без издержек
    # совокупный PnL обоих режимов совпадает (телескопирование разностей цен).
    prices = [100, 101, 103, 106, 110, 115]
    oof_bin = _price_oof(prices)
    oof_cont = _price_oof(prices, size=[1.0] * len(prices))
    bins = simulate_strategy(oof_bin, fee_rate=0.0, slippage_bps=0.0)
    cont = simulate_strategy(oof_cont, fee_rate=0.0, slippage_bps=0.0, size_col='size')
    assert np.isclose(bins['pnl_usd'].sum(), cont['pnl_usd'].sum())


def test_short_flips_pnl_sign():
    prices = [100, 102, 104, 106]           # монотонный рост
    longs = simulate_strategy(_price_oof(prices, size=[1, 1, 1, 1]),
                              fee_rate=0.0, slippage_bps=0.0, size_col='size')
    shorts = simulate_strategy(_price_oof(prices, size=[-1, -1, -1, -1]),
                               fee_rate=0.0, slippage_bps=0.0, size_col='size', allow_short=True)
    assert longs['pnl_usd'].sum() > 0
    assert np.isclose(shorts['pnl_usd'].sum(), -longs['pnl_usd'].sum())


def test_costs_charged_on_delta_position():
    prices = [100, 100, 100, 100]            # без движения -> PnL только издержки
    # позиция 0 -> 0.5 -> 0.5 -> 0: два изменения по 0.5
    oof = _price_oof(prices, size=[0.0, 0.5, 0.5, 0.0])
    res = simulate_strategy(oof, fee_rate=0.001, slippage_bps=0.0, size_col='size')
    # издержки = |Δpos| * entry_price * fee = (0.5+0.5)*100*0.001 = 0.1
    assert np.isclose(res['fee_usd'].sum(), 0.1)


def test_size_clipped_to_cap():
    prices = [100, 101, 102]
    res = simulate_strategy(_price_oof(prices, size=[5.0, 5.0, 5.0]),
                            fee_rate=0.0, slippage_bps=0.0, size_col='size', cap=1.0)
    assert res['position'].abs().max() <= 1.0 + 1e-9


def test_disallow_short_floors_at_zero():
    prices = [100, 101, 102]
    res = simulate_strategy(_price_oof(prices, size=[-1.0, -1.0, -1.0]),
                            fee_rate=0.0, slippage_bps=0.0, size_col='size', allow_short=False)
    assert (res['position'] >= 0).all()


def test_binary_path_unchanged_signature():
    # без size_col работает прежний бинарный режим и выдаёт ожидаемые колонки
    prices = [100, 101, 99, 102, 103]
    res = simulate_strategy(_price_oof(prices))
    for col in ('signal', 'position', 'trade', 'fee_usd', 'slippage_usd',
                'pnl_usd', 'ret_pct', 'equity'):
        assert col in res.columns


# --------------------------------------------------------------------------
# backtest_summary: Sortino / Calmar / PSR
# --------------------------------------------------------------------------

def test_summary_has_new_metrics():
    rng = np.random.default_rng(0)
    prices = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, size=200))
    oof = _price_oof(prices, size=np.clip(rng.normal(0.3, 0.3, size=200), -1, 1))
    strat = simulate_strategy(oof, fee_rate=0.0005, slippage_bps=2.0, size_col='size')
    summ = backtest_summary(strat, buy_and_hold(oof), periods_per_year=252)
    for col in ('sharpe', 'sortino', 'calmar', 'psr'):
        assert col in summ.columns
    # PSR ∈ [0,1]
    assert summ['psr'].dropna().between(0, 1).all()
