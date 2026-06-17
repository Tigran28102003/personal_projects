"""
Валидация для финансовых рядов с purge + embargo (EPIC 2, T2.1).

Обычный `TimeSeriesSplit`/`KFold` не учитывает, что метки в финансах не IID:
rolling-признаки и многобаровые (horizon>1) метки создают перекрытие между train
и test на стыке, что завышает OOS-оценку. Здесь реализованы два валидатора по
López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018):

* ``PurgedKFold`` (гл. 7) — K-fold с purge перекрывающихся меток и embargo-буфером;
* ``cpcv_splits`` (гл. 12) — Combinatorial Purged CV: тестирует k групп из N
  комбинаторно, давая множество backtest-путей (распределение метрик для DSR/PBO).

Модуль зависит только от ``numpy`` (+ stdlib) и совместим с интерфейсом
sklearn-сплиттера (``split``/``get_n_splits``), чтобы подставляться в
``cross_val_score`` и Optuna-objective'ы.

Соглашение о метках: наблюдение в позиции ``i`` имеет метку, занимающую
``horizon`` баров — интервал ``[i, i + horizon - 1]`` (для одношагового прогноза
``horizon=1`` метка занимает один бар, перекрытий нет, и работает только embargo).
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterator, Optional

import numpy as np


def _n_samples(X) -> int:
    """Число наблюдений в array-like / DataFrame / int."""
    if hasattr(X, "shape"):
        return int(X.shape[0])
    return len(X)


class PurgedKFold:
    """
    K-fold для временных рядов с purge перекрытий меток и embargo (AFML, гл. 7).

    Тестовые блоки — непрерывные и непересекающиеся (объединение покрывает все
    наблюдения, как в обычном KFold, но БЕЗ перемешивания). Из train для каждого
    фолда удаляются:
      * сам тестовый блок;
      * **purge**: наблюдения, чьи метки (``horizon`` баров) перекрываются по
        времени с метками теста — это бары непосредственно перед и после блока;
      * **embargo**: дополнительный буфер из ``embargo`` баров сразу после теста,
        убирающий утечку через автокорреляцию/задержанную реакцию рынка.

    Совместим с sklearn: ``split(X, y=None, groups=None)`` и ``get_n_splits``.

    `n_splits`: число фолдов
    `embargo`: размер embargo-буфера в барах (после тестового блока)
    `horizon`: длина метки в барах (для purge перекрытий; 1 = одношаговый таргет)
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0, horizon: int = 1):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if embargo < 0 or horizon < 1:
            raise ValueError("embargo must be >= 0 and horizon >= 1")
        self.n_splits = n_splits
        self.embargo = int(embargo)
        self.horizon = int(horizon)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = _n_samples(X)
        if n < self.n_splits:
            raise ValueError(f"n_samples={n} < n_splits={self.n_splits}")
        indices = np.arange(n)
        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)
        h = self.horizon

        for i in range(self.n_splits):
            a, b = int(bounds[i]), int(bounds[i + 1])   # тестовый блок [a, b)
            if b <= a:
                continue
            test_idx = indices[a:b]

            train_mask = np.ones(n, dtype=bool)
            # сам тестовый блок
            train_mask[a:b] = False
            # purge слева: метки train-обзора [j, j+h-1] заходят в тест -> j >= a-h+1
            train_mask[max(0, a - h + 1):a] = False
            # purge справа (метка теста заходит вперёд) + embargo
            right_end = min(n, b + (h - 1) + self.embargo)
            train_mask[b:right_end] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx


def cpcv_splits(
    n: int,
    N: int = 6,
    k: int = 2,
    embargo_frac: float = 0.01,
    horizon: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Combinatorial Purged Cross-Validation (AFML, гл. 12).

    Делит ``n`` наблюдений на ``N`` непрерывных групп; тестирует все сочетания из
    ``k`` групп (``C(N, k)`` путей), на каждом сплите применяя purge перекрытий
    меток и embargo вокруг КАЖДОЙ выбранной тест-группы. Даёт множество
    backtest-путей -> распределение OOS-метрик для DSR/PBO.

    `n`: число наблюдений
    `N`: число групп
    `k`: сколько групп идёт в тест на каждом сплите
    `embargo_frac`: доля ``n`` под embargo (но не меньше ``horizon``)
    `horizon`: длина метки в барах (для purge)

    Возвращает генератор пар ``(train_idx, test_idx)`` (отсортированные позиции).
    """
    if not (1 <= k < N):
        raise ValueError("require 1 <= k < N")
    if n < N:
        raise ValueError(f"n={n} < N={N}")

    emb = max(horizon, int(np.ceil(embargo_frac * n)))
    bounds = np.linspace(0, n, N + 1).astype(int)
    groups = [np.arange(int(bounds[i]), int(bounds[i + 1])) for i in range(N)]
    h = horizon

    for test_ids in combinations(range(N), k):
        train_mask = np.ones(n, dtype=bool)
        test_parts = []
        for gi in test_ids:
            g = groups[gi]
            if g.size == 0:
                continue
            a, b = int(g[0]), int(g[-1]) + 1
            test_parts.append(g)
            train_mask[a:b] = False
            # purge + embargo вокруг каждой тест-группы
            train_mask[max(0, a - h + 1):a] = False
            train_mask[b:min(n, b + (h - 1) + emb)] = False
        if not test_parts:
            continue
        test_idx = np.sort(np.concatenate(test_parts))
        train_idx = np.where(train_mask)[0]
        yield train_idx, test_idx


def n_cpcv_paths(N: int = 6, k: int = 2) -> int:
    """Число путей (сплитов) CPCV = C(N, k)."""
    from math import comb
    return comb(N, k)
