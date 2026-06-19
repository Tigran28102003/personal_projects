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

from typing import Optional

import numpy as np
import pandas as pd

from metrics import psr


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def round_trip_cost(fee_rate: float = 0.0005, slippage_bps: float = 2.0) -> float:
    """Полные издержки на один цикл вход+выход (доля номинала): комиссия с обеих
    сторон + проскальзывание с обеих сторон. Это естественный минимальный порог:
    сделка осмысленна, только если ожидаемое движение больше этой величины."""
    return 2 * fee_rate + 2 * (slippage_bps / 10_000)


def position_size(
    p_up: float,
    r_hat: float,
    sigma: float,
    kappa: float = 1.0,
    cap: float = 1.0,
):
    """Непрерывный размер позиции по уверенности и обратно — волатильности (T5.1).

    ``conf = 2·p_up − 1 ∈ [−1, 1]`` (уверенность классификатора); сырой размер
    ``kappa · conf · r̂ / (σ + eps)`` масштабирует ставку по сигналу и риску
    (vol-targeting), затем клипуется в ``[−cap, cap]`` (допускается short). Работает
    поэлементно для массивов и для скаляров.
    """
    conf = 2.0 * np.asarray(p_up, dtype=float) - 1.0
    raw = kappa * conf * (np.asarray(r_hat, dtype=float) / (np.asarray(sigma, dtype=float) + 1e-9))
    return np.clip(raw, -cap, cap)


def simulate_strategy(
    oof_df: pd.DataFrame,
    fee_rate: float = 0.0005,
    slippage_bps: float = 2.0,
    enter_threshold: float = 0.0,
    exit_threshold: Optional[float] = None,
    min_hold: int = 1,
    size_col: Optional[str] = None,
    cap: float = 1.0,
    allow_short: bool = True,
) -> pd.DataFrame:
    """
    Simulate a trading strategy on out-of-fold predictions.

    Два режима:
      * **binary long/flat** (по умолчанию, `size_col=None`) — прежний stateful
        сигнал с cost-aware порогом, гистерезисом и min-hold. Экономика НЕ изменена.
      * **continuous sizing** (`size_col` задан, T5.1) — целевая позиция берётся из
        колонки `size_col` (например, `meta_size` из мета-лейблинга): `position_t =
        clip(size, -cap|0, cap)`, издержки начисляются на изменение нотинала
        `|Δposition|` (одна сторона за сделку), допускается short (`allow_short`).

    Parameters
    ----------
    oof_df          : DataFrame ['timestamp','actual','predicted','fold','model'].
                      В continuous-режиме должна содержать колонку `size_col`.
    fee_rate        : fee per side as fraction of notional (0.05% Binance maker).
    slippage_bps    : one-way slippage in bps (default 2 bps).
    enter_threshold : (binary) войти в лонг, только если `r̂ = predicted/prev_price - 1
                      > enter_threshold`. Задавайте `>= round_trip_cost(...)`.
    exit_threshold  : (binary) выйти, когда `r̂ < exit_threshold`. None -> = enter.
    min_hold        : (binary) минимальное число баров удержания после входа.
    size_col        : имя колонки с непрерывным целевым размером позиции; None ->
                      бинарный режим.
    cap             : максимум |position| в continuous-режиме.
    allow_short     : разрешить отрицательную позицию (short) в continuous-режиме.

    Returns
    -------
    DataFrame с колонками: signal, position, trade, fee_usd, slippage_usd,
    pnl_usd, ret_pct, equity.
    """
    slippage_rate = slippage_bps / 10_000
    if exit_threshold is None:
        exit_threshold = enter_threshold

    results = []
    for model, grp in oof_df.groupby('model'):
        grp = grp.sort_values('timestamp').copy()
        prices = grp['actual'].values
        preds = grp['predicted'].values
        n = len(grp)

        fee_usd = np.zeros(n)
        slippage_usd = np.zeros(n)
        pnl_usd = np.zeros(n)

        if size_col is not None and size_col in grp.columns:
            # ---- continuous sizing: цель = size_col, издержки на |Δposition| ----
            lo = -cap if allow_short else 0.0
            position = np.clip(np.nan_to_num(grp[size_col].to_numpy(dtype=float), nan=0.0), lo, cap)
            signal = position.copy()
            trade = np.diff(position, prepend=0.0)  # изменение целевой позиции
            for t in range(n):
                entry_price = prices[t - 1] if t > 0 else prices[t]
                if trade[t] != 0:
                    notional = entry_price * abs(trade[t])     # изменённый нотинал
                    fee_usd[t] = notional * fee_rate           # одна сторона за сделку
                    slippage_usd[t] = notional * slippage_rate
                if t > 0:
                    pnl_usd[t] = position[t] * (prices[t] - prices[t - 1])
        else:
            # ---- binary long/flat (без изменений в экономике) ----
            # Stateful long/flat сигнал с порогом, гистерезисом и min-hold.
            # r̂_t = прогнозируемая доходность относительно прошлой фактической цены.
            signal = np.zeros(n, dtype=int)
            state, bars_held = 0, 0
            for t in range(1, n):
                prev = prices[t - 1]
                pr = (preds[t] / prev - 1.0) if prev != 0 else 0.0
                if state == 1:
                    bars_held += 1
                    if bars_held >= min_hold and pr < exit_threshold:
                        state, bars_held = 0, 0
                else:
                    if pr > enter_threshold:
                        state, bars_held = 1, 1
                signal[t] = state

            # Detect position changes to apply costs
            position = signal.copy()
            trade = np.diff(position, prepend=0)  # +1 = enter, -1 = exit, 0 = hold

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
    """Annualised Sharpe ratio (Rf = 0) on per-period percentage returns.

    Внимание (M4): аннуализация через √periods_per_year предполагает IID-доходности.
    На 5-мин/часовых данных с автокорреляцией и тяжёлыми хвостами это завышает
    Sharpe — для статвывода ориентируйтесь на PSR/DSR, а не на голый Sharpe.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0 or np.std(returns) == 0:
        return float('nan')
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def _sortino(returns: np.ndarray, periods_per_year: int) -> float:
    """Annualised Sortino ratio — штрафует только downside-волатильность."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return float('nan')
    downside = r[r < 0]
    dd = np.std(downside) if downside.size else 0.0
    if dd <= 0:
        return float('nan')
    return float(np.mean(r) / dd * np.sqrt(periods_per_year))


def _calmar(returns: np.ndarray, periods_per_year: int) -> float:
    """Calmar = annualised return / |max drawdown| на % equity-кривой cumprod(1+R)."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return float('nan')
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = np.divide(peak - eq, peak, out=np.zeros_like(eq), where=peak != 0)
    max_dd = float(dd.max())
    if max_dd <= 0:
        return float('nan')
    return float(np.mean(r) * periods_per_year / max_dd)


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
    [model, type, total_return_usd, pnl_usd, total_fees_usd, sharpe, sortino,
    calmar, psr, max_drawdown_usd, turnover]
    """
    rows = []
    for model, sg in strat_df.groupby('model'):
        sg = sg.sort_values('timestamp')
        trades = sg['trade'].abs().sum()

        sg_ret = sg['ret_pct'].values
        rows.append({
            'model': model,
            'type': 'Strategy',
            'total_return_usd': float(sg['equity'].iloc[-1]),
            'pnl_usd': float(sg['pnl_usd'].sum()),
            'total_fees_usd': float(sg['fee_usd'].sum() + sg['slippage_usd'].sum()),
            'sharpe': _sharpe(sg_ret, periods_per_year),
            'sortino': _sortino(sg_ret, periods_per_year),
            'calmar': _calmar(sg_ret, periods_per_year),
            'psr': psr(sg_ret),
            'max_drawdown_usd': _max_drawdown(sg['equity'].values),
            'turnover': int(trades),
        })

        bh = bh_df[bh_df['model'] == model].sort_values('timestamp')
        bh_ret = bh['bh_ret_pct'].values
        rows.append({
            'model': model,
            'type': 'Buy&Hold',
            'total_return_usd': float(bh['bh_equity'].iloc[-1]),
            'pnl_usd': float(bh['bh_pnl'].sum()),
            'total_fees_usd': 0.0,
            'sharpe': _sharpe(bh_ret, periods_per_year),
            'sortino': _sortino(bh_ret, periods_per_year),
            'calmar': _calmar(bh_ret, periods_per_year),
            'psr': psr(bh_ret),
            'max_drawdown_usd': _max_drawdown(bh['bh_equity'].values),
            'turnover': 0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Threshold tuning (turnover control)
# ---------------------------------------------------------------------------

def _strategy_stats(oof_one, fee_rate, slippage_bps, periods_per_year,
                    enter_thr, exit_thr, min_hold):
    """(sharpe, total_return_usd, turnover, total_fees_usd) для одной модели."""
    s = simulate_strategy(oof_one, fee_rate=fee_rate, slippage_bps=slippage_bps,
                          enter_threshold=enter_thr, exit_threshold=exit_thr, min_hold=min_hold)
    sharpe = _sharpe(s['ret_pct'].values, periods_per_year)
    total_return = float(s['equity'].iloc[-1]) if len(s) else 0.0
    turnover = int(s['trade'].abs().sum())
    fees = float(s['fee_usd'].sum() + s['slippage_usd'].sum())
    return sharpe, total_return, turnover, fees


def threshold_sweep(
    oof_df: pd.DataFrame,
    thresholds,
    fee_rate: float = 0.0005,
    slippage_bps: float = 2.0,
    periods_per_year: int = 252,
    min_hold: int = 1,
    hysteresis_gap: float = 0.0,
) -> pd.DataFrame:
    """
    Кривая «порог входа -> (Sharpe, доход, оборот)» для каждой модели.
    enter=thr, exit=thr-hysteresis_gap. Для графика «оборот vs доходность» и для
    демонстрации, как растущий порог режет число сделок (и комиссии).
    """
    rows = []
    for model, grp in oof_df.groupby('model'):
        g = grp.sort_values('timestamp')
        for thr in thresholds:
            sh, ret, to, fees = _strategy_stats(
                g, fee_rate, slippage_bps, periods_per_year, thr, thr - hysteresis_gap, min_hold)
            rows.append({'model': model, 'threshold': float(thr), 'sharpe': sh,
                         'total_return_usd': ret, 'turnover': to, 'total_fees_usd': fees})
    return pd.DataFrame(rows)


def select_threshold_by_sharpe(
    oof_df: pd.DataFrame,
    thresholds,
    fee_rate: float = 0.0005,
    slippage_bps: float = 2.0,
    periods_per_year: int = 252,
    val_frac: float = 0.4,
    min_hold: int = 1,
    hysteresis_gap: float = 0.0,
) -> dict:
    """
    Подбор порога входа по net-of-cost Sharpe на ХРОНОЛОГИЧЕСКИ первой части OOF
    (валидация, доля `val_frac`); оставшаяся тест-часть НЕ используется при выборе —
    значит порог выбран без подглядывания в тестовый период. Возвращает
    {model: best_threshold}; применять к тест-части отдельно.
    """
    best = {}
    for model, grp in oof_df.groupby('model'):
        g = grp.sort_values('timestamp')
        n_val = max(20, int(len(g) * val_frac))
        val = g.iloc[:n_val]
        best_thr, best_sh = float(thresholds[0]), -np.inf
        for thr in thresholds:
            sh, _, _, _ = _strategy_stats(
                val, fee_rate, slippage_bps, periods_per_year, thr, thr - hysteresis_gap, min_hold)
            if np.isfinite(sh) and sh > best_sh:
                best_sh, best_thr = sh, float(thr)
        best[model] = best_thr
    return best
