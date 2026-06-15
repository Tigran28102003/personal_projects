"""
Экономическая оценка торговых стратегий на основе прогнозов модели.

Перенесено практически без изменений из
`Bitcoin-price-prediction/src/models/economic_metrics.py` (дипломная работа
Ушкова Л.И., глава 6.6). Значение `bars_per_year` по умолчанию (365*96,
15-минутные бары) предназначено для исходного проекта Ушкова - при вызове из
текущего ноутбука для Daily/Hourly таймфреймов передавайте `bars_per_year`
явно (365 и 365*24 соответственно).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Базовые метрики
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: np.ndarray,
    bars_per_year: int = 365 * 96,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Аннуализированный коэффициент Шарпа.

    Параметры
    ---------
    returns       : серия доходностей стратегии за бар
    bars_per_year : число баров в году (для Daily=365, Hourly=365*24,
                    15-минутные крипто-бары=365*96)
    risk_free_rate: аннуализированная безрисковая ставка (по умолчанию 0)
    """
    returns = np.asarray(returns, dtype=float)
    excess = returns - risk_free_rate / bars_per_year
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(bars_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Максимальная просадка от пика до впадины equity curve."""
    equity_curve = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return float(dd.min())


def total_return(equity_curve: np.ndarray) -> float:
    """Полная доходность от начала до конца equity curve."""
    equity_curve = np.asarray(equity_curve, dtype=float)
    return float((equity_curve[-1] - equity_curve[0]) / equity_curve[0])


# ---------------------------------------------------------------------------
# Торговая стратегия
# ---------------------------------------------------------------------------

def simple_momentum_strategy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,
    bars_per_year: int = 365 * 96,
) -> dict:
    """
    Простая long-only momentum-стратегия, управляемая прогнозами модели.

    Правило сигнала:
        Если y_pred[t] > y_true[t] (модель предсказывает рост) -> long
        Иначе                                                   -> flat

    Размер позиции: 100% капитала в long, 0% в flat.

    Параметры
    ---------
    y_true           : реализованная серия цен (длина n)
    y_pred           : прогнозная серия цен (длина n, выровненная с y_true)
    initial_capital  : начальный капитал в USD
    transaction_cost : односторонняя комиссия как доля от объёма сделки
    bars_per_year    : число баров в году (для аннуализации Sharpe)

    Возвращает
    ----------
    dict с ключами:
        equity_curve  : np.ndarray - капитал во времени
        bar_returns   : np.ndarray - доходность стратегии за бар
        PnL           : float      - итоговая прибыль/убыток в USD
        total_return  : float      - полная доходность (доля)
        sharpe        : float      - аннуализированный Sharpe ratio
        max_drawdown  : float      - максимальная просадка (доля)
        n_trades      : int        - число изменений позиции
        turnover      : float      - суммарный оборот позиции
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    # Сигнал: 1 = long, 0 = flat
    signal = (y_pred > y_true).astype(float)

    # Фактическая доходность бара: (close[t+1] - close[t]) / close[t]
    price_returns = np.diff(y_true) / y_true[:-1]

    # Доходность стратегии: signal[t] * price_return[t+1]
    strategy_returns = signal[:-1] * price_returns

    # Учёт транзакционных издержек при смене сигнала
    signal_changes = np.diff(signal, prepend=signal[0])
    strategy_returns -= np.abs(signal_changes[:-1]) * transaction_cost

    # Equity curve
    equity = initial_capital * np.cumprod(1.0 + strategy_returns)
    equity = np.insert(equity, 0, initial_capital)

    pnl = equity[-1] - initial_capital
    ret = total_return(equity)
    sharpe = sharpe_ratio(strategy_returns, bars_per_year=bars_per_year)
    mdd = max_drawdown(equity)
    n_trades = int(np.sum(np.abs(signal_changes)))
    turnover = float(np.sum(np.abs(signal_changes)))

    return {
        "equity_curve": equity,
        "bar_returns": strategy_returns,
        "PnL": pnl,
        "total_return": ret,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "n_trades": n_trades,
        "turnover": turnover,
    }


def buy_and_hold(
    y_true: np.ndarray,
    initial_capital: float = 10_000.0,
    bars_per_year: int = 365 * 96,
) -> dict:
    """
    Buy-and-hold бенчмарк: инвестировать всё в момент 0 и держать.
    """
    y_true = np.asarray(y_true, dtype=float)
    price_returns = np.diff(y_true) / y_true[:-1]
    equity = initial_capital * np.cumprod(1.0 + price_returns)
    equity = np.insert(equity, 0, initial_capital)

    return {
        "equity_curve": equity,
        "bar_returns": price_returns,
        "PnL": equity[-1] - initial_capital,
        "total_return": total_return(equity),
        "sharpe": sharpe_ratio(price_returns, bars_per_year=bars_per_year),
        "max_drawdown": max_drawdown(equity),
        "n_trades": 1,
        "turnover": 1.0,
    }


# ---------------------------------------------------------------------------
# Сравнение стратегий
# ---------------------------------------------------------------------------

def compare_strategies(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    initial_capital: float = 10_000.0,
    bars_per_year: int = 365 * 96,
) -> pd.DataFrame:
    """
    Сравнивает стратегии нескольких моделей с buy-and-hold.

    Параметры
    ---------
    y_true       : реализованная серия цен
    predictions  : словарь {имя модели: прогнозная серия цен}
    initial_capital : начальный капитал
    bars_per_year    : число баров в году (для аннуализации Sharpe)

    Возвращает
    ----------
    pd.DataFrame с одной строкой на стратегию.
    """
    rows = []

    bh = buy_and_hold(y_true, initial_capital, bars_per_year=bars_per_year)
    rows.append(
        {
            "Strategy": "Buy & Hold",
            "Total Return (%)": round(bh["total_return"] * 100, 2),
            "Sharpe": round(bh["sharpe"], 3),
            "Max Drawdown (%)": round(bh["max_drawdown"] * 100, 2),
            "Turnover": round(bh["turnover"], 3),
            "Number of Trades": int(bh["n_trades"]),
            "PnL (USD)": round(bh["PnL"], 2),
        }
    )

    for name, y_pred in predictions.items():
        res = simple_momentum_strategy(y_true, y_pred, initial_capital, bars_per_year=bars_per_year)
        rows.append(
            {
                "Strategy": name,
                "Total Return (%)": round(res["total_return"] * 100, 2),
                "Sharpe": round(res["sharpe"], 3),
                "Max Drawdown (%)": round(res["max_drawdown"] * 100, 2),
                "Turnover": round(res["turnover"], 3),
                "Number of Trades": int(res["n_trades"]),
                "PnL (USD)": round(res["PnL"], 2),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

def plot_equity_curves(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    initial_capital: float = 10_000.0,
    bars_per_year: int = 365 * 96,
    title: str = "Strategy Equity Curves",
):
    """Графики equity curves для всех стратегий и buy-and-hold."""
    bh = buy_and_hold(y_true, initial_capital, bars_per_year=bars_per_year)

    plt.figure(figsize=(12, 5))
    plt.plot(bh["equity_curve"], label="Buy & Hold", linewidth=2, color="black")

    for name, y_pred in predictions.items():
        res = simple_momentum_strategy(y_true, y_pred, initial_capital, bars_per_year=bars_per_year)
        plt.plot(res["equity_curve"], label=name, linewidth=1.5)

    plt.title(title)
    plt.xlabel("Bar")
    plt.ylabel("Portfolio Value (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
