"""
Метрики качества прогноза и экономической оценки стратегии (EPIC 0, T0.2).

Модуль намеренно зависит только от ``numpy`` и ``scipy`` — он импортируется без
``catboost``/``torch``/``sklearn``, чтобы метрики можно было считать в лёгких
скриптах, тестах и в ноутбуке-отчёте без тяжёлого ML-стека.

Все функции рассчитаны на таргет-доходность ``r_t = ln(P_t) - ln(P_{t-1})``:
прогноз ``y_pred`` и факт ``y_true`` — это доходности (а не уровни цены).

Состав:
* ``rank_ic``                — Spearman(r̂, r), монотонная связь прогноза и факта;
* ``ic_ir``                  — mean/std IC по фолдам (стабильность сигнала);
* ``mwda``                   — magnitude-weighted directional accuracy;
* ``return_capture``         — доля захваченного абсолютного движения ∈ [-1, 1];
* ``mcc``                    — Matthews correlation coefficient для знака;
* ``balanced_accuracy``      — (TPR + TNR) / 2, решает проблему base-rate;
* ``net_cost_sharpe``        — Sharpe стратегии после комиссий на оборот;
* ``psr``                    — Probabilistic Sharpe Ratio (с поправкой на γ3/γ4);
* ``deflated_sharpe``        — DSR (поправка на множественное тестирование);
* ``prob_backtest_overfitting`` — PBO через CSCV.

Формулы — см. сопроводительный аудит (§2.3, §3.4): PSR с поправкой на
скошенность/куртозис (Bailey & López de Prado, 2012); DSR корректирует на число
испытаний и не-нормальность (2014); PBO — непараметрический CSCV (Bailey et al.).
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.stats import kurtosis, norm, rankdata, skew, spearmanr

__all__ = [
    "rank_ic",
    "ic_ir",
    "mwda",
    "return_capture",
    "mcc",
    "balanced_accuracy",
    "net_cost_sharpe",
    "psr",
    "deflated_sharpe",
    "prob_backtest_overfitting",
]

_EPS = 1e-12


def _finite_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Приводит к float-массивам и оставляет только попарно конечные значения."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


# ==================== Информационный коэффициент ====================

def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    `Rank Information Coefficient` — ранговая корреляция Спирмена между прогнозом
    и фактической доходностью на одном фолде/срезе.

    IC — стандарт quant-индустрии (Numerai, факторные модели): измеряет монотонную
    связь ``r̂`` с реализованной ``r``, устойчив к выбросам и калибровке прогноза.
    Идеальный монотонный прогноз → IC=1; случайный → IC≈0.

    Возвращает ``nan`` при <2 наблюдениях или вырожденном (константном) входе.

    `y_true`: фактическая доходность на тесте
    `y_pred`: прогноз доходности на тесте
    """
    yt, yp = _finite_pair(y_true, y_pred)
    if yt.size < 2:
        return float("nan")
    if np.ptp(yt) == 0 or np.ptp(yp) == 0:
        return float("nan")
    rho = spearmanr(yt, yp).correlation
    return float(rho)


def ic_ir(ics: Sequence[float]) -> float:
    """
    `IC Information Ratio` — отношение среднего IC к его разбросу по фолдам:
    ``mean(IC) / std(IC)``. Штрафует нестабильный сигнал: даже небольшой, но
    устойчиво положительный IC коммерчески ценнее, чем большой, но скачущий.

    NaN-фолды отбрасываются. При <2 валидных IC возвращает ``nan``.

    `ics`: список значений IC по фолдам
    """
    arr = np.asarray(list(ics), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    sd = float(np.std(arr, ddof=1))
    return float(np.mean(arr) / (sd + _EPS))


# ==================== Магнитудно-взвешенные метрики ====================

def mwda(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    `Magnitude-Weighted Directional Accuracy` —
    ``Σ |r_t| · 1[sign(r̂_t)=sign(r_t)] / Σ |r_t|``.

    В отличие от обычной DA, взвешивает попадания по величине бара, поэтому
    напрямую коррелирует с PnL безразмерного long/flat-бета: крупные движения
    весят больше, шумовые бары около нуля — почти не влияют. Идеальный по знаку
    прогноз → 1; противоположный по знаку → 0.

    `y_true`: фактическая доходность на тесте
    `y_pred`: прогноз доходности на тесте
    """
    yt, yp = _finite_pair(y_true, y_pred)
    w = np.abs(yt)
    denom = w.sum()
    if denom <= _EPS:
        return float("nan")
    hit = (np.sign(yp) == np.sign(yt)).astype(float)
    return float(np.sum(w * hit) / denom)


def return_capture(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    `Return-Capture Ratio` (Mean Directional Value) —
    ``Σ sign(r̂_t) · r_t / Σ |r_t|`` ∈ [-1, 1].

    Это PnL безразмерного long/short-бета, нормированный на полный «теоретический»
    PnL (если бы все знаки угадывались). Идеальный прогноз → 1; инвертированный
    знак → -1; случайный → ≈0.

    `y_true`: фактическая доходность на тесте
    `y_pred`: прогноз доходности на тесте
    """
    yt, yp = _finite_pair(y_true, y_pred)
    denom = np.abs(yt).sum()
    if denom <= _EPS:
        return float("nan")
    return float(np.sum(np.sign(yp) * yt) / denom)


# ==================== Метрики для знаковой задачи ====================

def _binary_confusion(y_true_sign: np.ndarray, y_pred_sign: np.ndarray):
    """Считает (TP, TN, FP, FN), трактуя положительные значения как класс 1."""
    yt = np.asarray(y_true_sign, dtype=float).ravel()
    yp = np.asarray(y_pred_sign, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    pos_t = yt > 0
    pos_p = yp > 0
    tp = float(np.sum(pos_t & pos_p))
    tn = float(np.sum(~pos_t & ~pos_p))
    fp = float(np.sum(~pos_t & pos_p))
    fn = float(np.sum(pos_t & ~pos_p))
    return tp, tn, fp, fn


def mcc(y_true_sign: np.ndarray, y_pred_sign: np.ndarray) -> float:
    """
    `Matthews Correlation Coefficient` —
    ``(TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))`` ∈ [-1, 1].

    Корреляция Пирсона на confusion-матрице: учитывает все четыре ячейки и в
    среднем информативнее accuracy/F1 при дисбалансе. Симметрична относительно
    перестановки факта и прогноза. Если какой-либо множитель в знаменателе равен 0
    (вырожденная разметка — один класс), возвращает 0.0 (конвенция sklearn).

    Вход — знаки/бинарные метки: положительное значение трактуется как класс 1.
    """
    tp, tn, fp, fn = _binary_confusion(y_true_sign, y_pred_sign)
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if den == 0:
        return 0.0
    return float(num / den)


def balanced_accuracy(y_true_sign: np.ndarray, y_pred_sign: np.ndarray) -> float:
    """
    `Balanced Accuracy` — ``(TPR + TNR) / 2`` — среднее recall по двум классам.

    Прямо решает проблему base-rate/дисбаланса: на бычьем рынке простое
    угадывание мажоритарного «up» даёт высокий accuracy, но balanced accuracy
    остаётся ≈0.5. Если в факте присутствует только один класс, усредняется по
    определённым recall (второй класс игнорируется).

    Вход — знаки/бинарные метки: положительное значение трактуется как класс 1.
    """
    tp, tn, fp, fn = _binary_confusion(y_true_sign, y_pred_sign)
    recalls = []
    if (tp + fn) > 0:
        recalls.append(tp / (tp + fn))   # TPR
    if (tn + fp) > 0:
        recalls.append(tn / (tn + fp))   # TNR
    if not recalls:
        return float("nan")
    return float(np.mean(recalls))


# ==================== Экономические метрики ====================

def net_cost_sharpe(
    side: np.ndarray,
    r: np.ndarray,
    fee: float = 0.0014,
    ann: float = 1.0,
) -> float:
    """
    Sharpe стратегии ПОСЛЕ издержек на оборот позиции.

    Доходность стратегии в периоде t: ``R_t = side_t · r_t − fee · |Δside_t|``,
    где издержка начисляется на изменение нотинала позиции (вход/выход/переворот).
    Перед первым баром позиция считается нулевой (вход из 0 → side_0 платный).
    Затем Sharpe = ``√ann · mean(R) / std(R)``.

    `side`: позиция по периодам — знак {-1,0,1} или непрерывный размер ∈ [-cap, cap]
    `r`: фактическая доходность актива по периодам (выровнена с `side`)
    `fee`: round-trip-доля издержек на единичное изменение позиции (по умолчанию 0.14%)
    `ann`: множитель аннуализации дисперсии (periods_per_year); 1.0 — без аннуализации
    """
    pos = np.asarray(side, dtype=float).ravel()
    ret = np.asarray(r, dtype=float).ravel()
    if pos.shape != ret.shape:
        raise ValueError(f"shape mismatch: {pos.shape} vs {ret.shape}")
    mask = np.isfinite(pos) & np.isfinite(ret)
    pos, ret = pos[mask], ret[mask]
    if pos.size == 0:
        return float("nan")
    prev = np.concatenate([[0.0], pos[:-1]])
    cost = fee * np.abs(pos - prev)
    strat = pos * ret - cost
    sd = float(np.std(strat, ddof=1)) if strat.size > 1 else 0.0
    if sd <= _EPS:
        return float("nan")
    return float(np.sqrt(ann) * np.mean(strat) / sd)


def psr(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """
    `Probabilistic Sharpe Ratio` (Bailey & López de Prado, 2012) — вероятность
    того, что истинный Sharpe строго выше порога ``sr_benchmark``, с поправкой на
    скошенность γ3 и куртозис γ4 распределения доходностей и длину выборки T:

        PSR(SR*) = Φ[ (ŜR − SR*)·√(T−1) / √(1 − γ3·ŜR + ((γ4−1)/4)·ŜR²) ]

    где ŜR — выборочный per-period Sharpe (mean/std), γ3 — скошенность, γ4 —
    полный куртозис (для нормального = 3), Φ — CDF стандартного нормального.
    Критично для тяжелохвостых крипто-доходностей: точечная оценка SR от моментов
    не зависит, а вот её доверительные границы — сильно.

    `returns`: per-period доходности стратегии (после издержек)
    `sr_benchmark`: пороговый per-period Sharpe SR* (по умолчанию 0)
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    T = r.size
    if T < 3:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if sd <= _EPS:
        return float("nan")
    sr_hat = float(np.mean(r) / sd)
    g3 = float(skew(r, bias=False))
    g4 = float(kurtosis(r, fisher=False, bias=False))  # полный куртозис (норм=3)
    denom_sq = 1.0 - g3 * sr_hat + ((g4 - 1.0) / 4.0) * sr_hat ** 2
    if denom_sq <= _EPS:
        return float("nan")
    z = (sr_hat - sr_benchmark) * np.sqrt(T - 1) / np.sqrt(denom_sq)
    return float(norm.cdf(z))


def deflated_sharpe(
    sharpes: Sequence[float],
    observed_sr: float,
    T: int,
    skew_: float = 0.0,
    kurt_: float = 3.0,
) -> float:
    """
    `Deflated Sharpe Ratio` (Bailey & López de Prado, 2014) — PSR, в котором порог
    SR* заменён на ожидаемый максимум Sharpe при множественном тестировании, то
    есть DSR корректирует на (1) selection bias по числу испытаний N и (2)
    не-нормальность распределения доходностей.

    Ожидаемый максимум по N независимым испытаниям с дисперсией Sharpe ``V``:

        SR0 = √V · [ (1−γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]

    где γ ≈ 0.5772 — постоянная Эйлера–Маскерони, Z⁻¹ — обратная нормальная CDF.
    Затем DSR = Φ[ (ŜR − SR0)·√(T−1) / √(1 − γ3·ŜR + ((γ4−1)/4)·ŜR²) ].

    `sharpes`: per-period Sharpe-оценки всех испытанных конфигураций (для оценки V и N)
    `observed_sr`: per-period Sharpe выбранной «лучшей» стратегии
    `T`: длина трек-рекорда (число доходностей)
    `skew_`, `kurt_`: моменты доходностей выбранной стратегии (по умолчанию — нормальные)
    """
    arr = np.asarray(list(sharpes), dtype=float)
    arr = arr[np.isfinite(arr)]
    N = arr.size
    if N < 2 or T < 3:
        return float("nan")
    V = float(np.var(arr, ddof=1))
    if V <= _EPS:
        return float("nan")
    gamma = 0.5772156649015329  # постоянная Эйлера–Маскерони
    z1 = norm.ppf(1.0 - 1.0 / N)
    z2 = norm.ppf(1.0 - 1.0 / (N * np.e))
    sr0 = np.sqrt(V) * ((1.0 - gamma) * z1 + gamma * z2)
    denom_sq = 1.0 - skew_ * observed_sr + ((kurt_ - 1.0) / 4.0) * observed_sr ** 2
    if denom_sq <= _EPS:
        return float("nan")
    z = (observed_sr - sr0) * np.sqrt(T - 1) / np.sqrt(denom_sq)
    return float(norm.cdf(z))


def prob_backtest_overfitting(
    perf_matrix: np.ndarray,
    n_splits: int = 16,
) -> float:
    """
    `Probability of Backtest Overfitting` (PBO) через CSCV (Bailey et al.,
    Combinatorially Symmetric Cross-Validation).

    Идея: матрицу производительности конфигураций ``M`` формы (T_наблюдений,
    N_конфигураций) делим по строкам на S непересекающихся блоков; для каждого
    разбиения блоков пополам на IS/OOS выбираем по IS лучшую конфигурацию n*,
    смотрим её относительный ранг в OOS, считаем логит ``λ = ln(ω/(1−ω))``.
    PBO = доля разбиений, где λ ≤ 0 (IS-лучшая оказалась не выше медианы OOS).

    Высокий PBO (→1) сигналит, что лидер бэктеста, скорее всего, переобучен.

    `perf_matrix`: массив формы (T, N) — per-period производительность каждой из N
        конфигураций по T срезам времени (например, доходности стратегии или per-fold IC)
    `n_splits`: число блоков S (чётное); при T < S автоматически уменьшается
    """
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("perf_matrix должен быть 2D: (T_наблюдений, N_конфигураций)")
    T, N = M.shape
    if N < 2 or T < 4:
        return float("nan")

    S = min(n_splits, T)
    if S % 2 == 1:
        S -= 1
    if S < 2:
        return float("nan")

    # непересекающиеся (почти) равные блоки строк
    block_bounds = np.linspace(0, T, S + 1).astype(int)
    blocks = [np.arange(block_bounds[i], block_bounds[i + 1]) for i in range(S)]

    lambdas = []
    half = S // 2
    for is_ids in combinations(range(S), half):
        is_set = set(is_ids)
        is_rows = np.concatenate([blocks[i] for i in is_ids])
        oos_rows = np.concatenate([blocks[i] for i in range(S) if i not in is_set])

        is_perf = np.nanmean(M[is_rows], axis=0)
        oos_perf = np.nanmean(M[oos_rows], axis=0)
        if not np.all(np.isfinite(is_perf)) or not np.all(np.isfinite(oos_perf)):
            continue

        n_star = int(np.argmax(is_perf))
        oos_ranks = rankdata(oos_perf)            # 1..N, выше = лучше
        omega = oos_ranks[n_star] / (N + 1.0)     # относительный ранг ∈ (0, 1)
        omega = min(max(omega, _EPS), 1.0 - _EPS)
        lambdas.append(np.log(omega / (1.0 - omega)))

    if not lambdas:
        return float("nan")
    lam = np.asarray(lambdas)
    return float(np.mean(lam <= 0.0))
