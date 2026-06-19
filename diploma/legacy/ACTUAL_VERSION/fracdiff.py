"""Fractional differentiation (EPIC 7, T7.3) — López de Prado, *AFML* гл. 5.

«Третий путь» между ценой (d=0, максимум памяти, нестационарна) и доходностью
(d=1, стационарна, память стёрта): дробное дифференцирование с минимальным
``d*``, при котором ряд проходит ADF-тест на стационарность, сохраняет максимум
памяти при стационарности. Используется как дополнительный признак-память поверх
уже стационарного таргета-доходности.

Реализован Fixed-Width Window FFD: веса ``w_k = −w_{k−1}·(d−k+1)/k`` обрезаются,
когда ``|w_k| < thresh`` (фиксированное окно -> одинаковый лаг для всех t, без
«заглядывания» переменной длины окна). Модуль лёгкий: numpy/pandas, ADF — lazy
import statsmodels.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def ffd_weights(d: float, thresh: float = 1e-5, max_size: int = 10_000) -> np.ndarray:
    """Веса FFD для порядка `d`, обрезанные по порогу `thresh`.

    ``w_0 = 1``; ``w_k = −w_{k−1}·(d − k + 1)/k``. Возвращает массив весов
    ``[w_0, w_1, …]`` (по возрастанию лага), отсортированный для свёртки с прошлым.
    """
    if d < 0:
        raise ValueError(f'd must be >= 0, got {d}')
    w = [1.0]
    k = 1
    while k < max_size:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w, dtype=float)


def frac_diff_ffd(series: pd.Series, d: float, thresh: float = 1e-5) -> pd.Series:
    """Fixed-Width Window fractional differencing ряда `series` порядка `d`.

    Свёртка значения t с фиксированным окном прошлых значений по весам `ffd_weights`.
    Первые ``len(weights)-1`` точек -> NaN (неполное окно). При ``d == 0`` возвращает
    исходный ряд.
    """
    if d == 0:
        return series.astype(float).copy()
    w = ffd_weights(d, thresh=thresh)
    width = len(w) - 1
    vals = series.astype(float).to_numpy()
    out = np.full(len(vals), np.nan, dtype=float)
    # w[0] — вес текущего бара, w[-1] — самого старого в окне; для свёртки разворачиваем.
    w_rev = w[::-1]
    for i in range(width, len(vals)):
        window = vals[i - width: i + 1]
        if np.isnan(window).any():
            continue
        out[i] = float(np.dot(w_rev, window))
    return pd.Series(out, index=series.index, name=series.name)


def _adf_pvalue(series: pd.Series) -> float:
    """ADF p-value (lazy statsmodels). Меньше -> сильнее отвергаем единичный корень."""
    from statsmodels.tsa.stattools import adfuller
    x = series.dropna().to_numpy()
    if x.size < 20:
        return float('nan')
    return float(adfuller(x, maxlag=1, regression='c', autolag=None)[1])


def min_ffd_d(
    series: pd.Series,
    d_grid: Optional[np.ndarray] = None,
    thresh: float = 1e-5,
    adf_pvalue: float = 0.05,
) -> tuple[float, pd.Series]:
    """Минимальный `d*` из сетки, при котором FFD-ряд проходит ADF (стационарен).

    Возвращает ``(d_star, ffd_series)``. Если ни один `d` из сетки не даёт
    стационарности при пороге `adf_pvalue`, берётся максимальный `d` сетки (и
    предупреждение оставляется на стороне вызывающего по p-value). Сетка по
    умолчанию — ``np.linspace(0, 1, 11)``.
    """
    if d_grid is None:
        d_grid = np.linspace(0.0, 1.0, 11)
    best_d, best_series = float(d_grid[-1]), None
    for d in d_grid:
        fd = frac_diff_ffd(series, float(d), thresh=thresh)
        p = _adf_pvalue(fd)
        if np.isfinite(p) and p < adf_pvalue:
            return float(d), fd
        best_series = fd
    return best_d, best_series
