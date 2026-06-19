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


# ===========================================================================
# Purged + embargoed WALK-FORWARD (PRIMARY method for selection/tuning) — диплом
# ---------------------------------------------------------------------------
# Forward-chaining: train ВСЕГДА строго раньше test. В отличие от PurgedKFold/CPCV
# здесь нет обучения на пост-тестовых данных => deploy-честная оценка. Перед каждым
# тестовым блоком выкидывается РОВНО `purge` (= HORIZON) train-баров + `embargo`
# (R1: перепургить на бар лучше, чем недопургить — метка бара t тянется до t+H).
# ===========================================================================

def purged_walk_forward_splits(
    n: int,
    n_splits: int = 5,
    purge: int = 24,
    embargo: int = 24,
    min_train_frac: float = 0.5,
    window_mode: str = "expanding",
    train_window: Optional[int] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged+embargoed walk-forward фолды.

    `window_mode`:
      * 'expanding' — train растёт от начала до (начало теста − purge − embargo);
      * 'rolling'   — train фиксированной ширины `train_window`, скользит вперёд.
    Тестовые блоки — последовательные равные куски в хвостовой доле
    ``(1 - min_train_frac)`` ряда. Перед каждым тестом из train удаляются последние
    ``purge + embargo`` баров (purge гасит перекрытие H-баровых меток, embargo —
    автокорреляцию). Возвращает список ``(train_idx, test_idx)`` (позиции).
    """
    if window_mode not in ("expanding", "rolling"):
        raise ValueError("window_mode must be 'expanding' or 'rolling'")
    if window_mode == "rolling" and not train_window:
        raise ValueError("rolling window requires train_window")
    gap = int(purge) + int(embargo)
    min_train = int(np.floor(min_train_frac * n))
    remaining = n - min_train
    test_size = remaining // n_splits
    if test_size < 1:
        raise ValueError(
            f"not enough rows: n={n}, min_train_frac={min_train_frac}, n_splits={n_splits}")

    splits = []
    test_start = min_train
    for _ in range(n_splits):
        test_end = min(test_start + test_size, n)
        if test_end <= test_start:
            break
        train_end = test_start - gap          # purge + embargo before the test block
        if window_mode == "expanding":
            train_beg = 0
        else:
            train_beg = max(0, train_end - int(train_window))
        if train_end - train_beg < 1:
            test_start = test_end
            continue
        train_idx = np.arange(train_beg, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
        test_start = test_end
        if test_start >= n:
            break
    if not splits:
        raise ValueError("no valid walk-forward folds produced")
    return splits


# ---------------------------------------------------------------------------
# Adaptation scaffold (R4/R5): recency weights, label uniqueness, drift detection
# ---------------------------------------------------------------------------

def recency_weights(n: int, halflife: Optional[int]) -> np.ndarray:
    """Экспоненциальные веса по давности: вес бара i = 0.5 ** ((n-1-i)/halflife).

    Самый свежий бар получает вес 1, более старые — экспоненциально меньше. ``halflife``
    в барах; ``None`` -> равные веса (адаптация выключена). Для R5 (sample_weight)."""
    if halflife is None or halflife <= 0:
        return np.ones(n, dtype=float)
    age = np.arange(n)[::-1].astype(float)      # n-1-i: 0 для последнего бара
    return np.power(0.5, age / float(halflife))


def average_uniqueness(n: int, horizon: int) -> np.ndarray:
    """Веса уникальности меток для H-баровых перекрывающихся меток (AFML гл.4, упрощ.).

    Метка бара t занимает [t, t+H-1]; в каждый момент его делят до H меток, поэтому
    вклад невелик. Возвращает per-бар среднюю уникальность ∈ (0,1]: считаем загрузку
    каждого момента числом активных меток и усредняем обратную загрузку по интервалу
    метки. Гасит дублирование информации соседних баров в sample_weight (R5)."""
    h = max(1, int(horizon))
    load = np.zeros(n + h, dtype=float)
    for t in range(n):
        load[t:t + h] += 1.0
    load[load == 0] = 1.0
    u = np.empty(n, dtype=float)
    for t in range(n):
        u[t] = float(np.mean(1.0 / load[t:t + h]))
    return u


def detect_drift(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> dict:
    """Дрейф распределения признака/таргета: PSI + KS между ref и cur выборками.

    PSI (population stability index) и статистика Колмогорова–Смирнова. Используется в
    мониторинге распада эджа (R4): большой PSI/KS => распределение фич/таргета уехало,
    модель пора переобучать. Возвращает {'psi': float, 'ks': float, 'ks_p': float}."""
    from scipy.stats import ks_2samp
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size < 2 or cur.size < 2:
        return {"psi": float("nan"), "ks": float("nan"), "ks_p": float("nan")}
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    eps = 1e-6
    p = np.histogram(ref, bins=edges)[0] / ref.size + eps
    q = np.histogram(cur, bins=edges)[0] / cur.size + eps
    psi = float(np.sum((p - q) * np.log(p / q)))
    ks = ks_2samp(ref, cur)
    return {"psi": psi, "ks": float(ks.statistic), "ks_p": float(ks.pvalue)}


def fold_drift(X, splits, bins: int = 10) -> list:
    """Средний по фичам PSI между train- и test-блоком каждого walk-forward фолда
    (мониторинг распада эджа / нестационарности, W5)."""
    Xv = np.asarray(X, dtype=float)
    out = []
    for tr, te in splits:
        psis = []
        for j in range(Xv.shape[1]):
            d = detect_drift(Xv[tr, j], Xv[te, j], bins=bins)
            if np.isfinite(d["psi"]):
                psis.append(d["psi"])
        out.append(float(np.nanmean(psis)) if psis else float("nan"))
    return out


def drift_triggered_schedule(psi_series, threshold: float = 0.2) -> list:
    """Индексы фолдов, где средний PSI превысил порог -> сигнал «пора переобучать»
    (drift-триггерное переобучение в дополнение к каденсу RETRAIN_EVERY, W5)."""
    return [i for i, p in enumerate(psi_series) if np.isfinite(p) and p > threshold]
