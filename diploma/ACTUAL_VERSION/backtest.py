"""
Economic backtesting for BTC forecasting pipeline.

`simulate_strategy` converts out-of-fold price predictions into a long/flat
trading signal and computes PnL with Binance-style maker/taker fees and
slippage. `buy_and_hold` gives the passive benchmark over the same period.
`backtest_summary` produces a comparison table (Total Return, Sharpe,
Max Drawdown, Turnover, PnL USD) for all models + B&H on a per-frequency basis.

Fee model
---------
Binance spot maker/taker: 0.1% per side (0.05% with BNB discount; we use
0.05% as the `fee_rate` default). Slippage is modelled as a fixed number of
basis points on the trade notional (default 2 bps). Both are charged on every
position change (entry and exit).

Signal rule
-----------
`y_pred[t] > y_actual[t-1]`  ->  long BTC (hold 1 unit)
`y_pred[t] <= y_actual[t-1]` ->  flat (no position)

`oof_df` carries the *reconstructed price* OOF (P̂_t = P_{t-1}·exp(r̂_t)), so the
rule above is exactly the directional bet "predicted return > 0 -> long".

A position change at time t incurs one round of fees + slippage on the notional
`y_actual[t-1]` (the entry/exit price proxy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_strategy(
    oof_df: pd.DataFrame,
    fee_rate: float = 0.0005,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    """
    Simulate a long/flat strategy on out-of-fold predictions.

    Parameters
    ----------
    oof_df      : DataFrame with columns ['timestamp','actual','predicted','fold','model']
                  Must be sorted by timestamp within each model.
    fee_rate    : Round-trip fee per trade as fraction of notional.
                  Default 0.05% (Binance maker, 0.001 for taker).
    slippage_bps: One-way slippage in basis points. Default 2 bps (0.02%).

    Returns
    -------
    DataFrame with per-timestamp equity curve columns added:
      signal, position, trade, fee_usd, slippage_usd, pnl_usd, equity
    """
    slippage_rate = slippage_bps / 10_000

    results = []
    for model, grp in oof_df.groupby('model'):
        grp = grp.sort_values('timestamp').copy()
        prices = grp['actual'].values
        preds = grp['predicted'].values
        n = len(grp)

        # Long signal: predict next price higher than previous actual
        signal = np.zeros(n, dtype=int)
        for t in range(1, n):
            if preds[t] > prices[t - 1]:
                signal[t] = 1

        # Detect position changes to apply costs
        position = signal.copy()
        trade = np.diff(position, prepend=0)  # +1 = enter, -1 = exit, 0 = hold

        fee_usd = np.zeros(n)
        slippage_usd = np.zeros(n)
        pnl_usd = np.zeros(n)

        for t in range(n):
            entry_price = prices[t - 1] if t > 0 else prices[t]
            if trade[t] != 0:
                notional = entry_price
                fee_usd[t] = notional * fee_rate * 2  # entry + exit both charged at trade
                slippage_usd[t] = notional * slippage_rate * 2
            if position[t] == 1 and t > 0:
                # PnL from holding one unit: price change
                pnl_usd[t] = prices[t] - prices[t - 1]

        net_pnl = pnl_usd - fee_usd - slippage_usd
        equity = np.cumsum(net_pnl)

        # Перевод PnL в процентную доходность периода относительно предыдущей цены
        # (нотинал = одна единица BTC по цене prices[t-1]). Sharpe считается именно
        # на ret_pct, а не на USD-PnL: процентная доходность масштабо-инвариантна и
        # сопоставима между периодами разной цены BTC (иначе высокоценовые куски
        # ряда искусственно доминировали бы в оценке риска).
        base = np.empty(n)
        base[0] = prices[0]
        base[1:] = prices[:-1]
        ret_pct = np.divide(net_pnl, base, out=np.zeros_like(net_pnl), where=base != 0)

        grp = grp.copy()
        grp['signal'] = signal
        grp['position'] = position
        grp['trade'] = trade
        grp['fee_usd'] = fee_usd
        grp['slippage_usd'] = slippage_usd
        grp['pnl_usd'] = net_pnl
        grp['ret_pct'] = ret_pct
        grp['equity'] = equity
        results.append(grp)

    return pd.concat(results, ignore_index=True)


def buy_and_hold(oof_df: pd.DataFrame) -> pd.DataFrame:
    """
    Passive Buy & Hold benchmark: hold 1 BTC from the first OOF timestamp.

    Returns per-timestamp equity series for each model's time window
    (same timestamps as `oof_df`, so comparison is aligned period-by-period).
    """
    results = []
    for model, grp in oof_df.groupby('model'):
        grp = grp.sort_values('timestamp').copy()
        prices = grp['actual'].values
        pnl = np.concatenate([[0.0], np.diff(prices)])
        base = np.empty(len(prices))
        base[0] = prices[0]
        base[1:] = prices[:-1]
        grp['bh_pnl'] = pnl
        grp['bh_ret_pct'] = np.divide(pnl, base, out=np.zeros_like(pnl), where=base != 0)
        grp['bh_equity'] = np.cumsum(pnl)
        results.append(grp)
    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    """Annualised Sharpe ratio (Rf = 0) on per-period percentage returns."""
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0 or np.std(returns) == 0:
        return float('nan')
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown in USD."""
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    return float(drawdowns.max())


def backtest_summary(
    strat_df: pd.DataFrame,
    bh_df: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Comparison table: strategy vs Buy & Hold per model.

    Parameters
    ----------
    strat_df        : output of `simulate_strategy`
    bh_df           : output of `buy_and_hold` (same oof_df)
    periods_per_year: for Sharpe annualisation (252 daily, 8760 hourly, 105120 5-min)

    Returns
    -------
    DataFrame with one row per (model, 'Strategy'/'Buy&Hold') and columns
    [model, type, total_return_usd, sharpe, max_drawdown_usd, turnover, pnl_usd]
    """
    rows = []
    for model, sg in strat_df.groupby('model'):
        sg = sg.sort_values('timestamp')
        trades = sg['trade'].abs().sum()

        rows.append({
            'model': model,
            'type': 'Strategy',
            'total_return_usd': float(sg['equity'].iloc[-1]),
            'pnl_usd': float(sg['pnl_usd'].sum()),
            'total_fees_usd': float(sg['fee_usd'].sum() + sg['slippage_usd'].sum()),
            'sharpe': _sharpe(sg['ret_pct'].values, periods_per_year),
            'max_drawdown_usd': _max_drawdown(sg['equity'].values),
            'turnover': int(trades),
        })

        bh = bh_df[bh_df['model'] == model].sort_values('timestamp')
        rows.append({
            'model': model,
            'type': 'Buy&Hold',
            'total_return_usd': float(bh['bh_equity'].iloc[-1]),
            'pnl_usd': float(bh['bh_pnl'].sum()),
            'total_fees_usd': 0.0,
            'sharpe': _sharpe(bh['bh_ret_pct'].values, periods_per_year),
            'max_drawdown_usd': _max_drawdown(bh['bh_equity'].values),
            'turnover': 0,
        })

    return pd.DataFrame(rows)
