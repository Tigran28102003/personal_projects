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


# ===========================================================================
# 3-class classification layer + calibration + DM + precision-at-coverage + evaluate
# ---------------------------------------------------------------------------
# Класс-метки фиксированы как (-1, 0, +1); колонки proba — [P(-1), P(0), P(+1)].
# Этот блок зависит от sklearn (в отличие от numpy-ядра выше) — он используется в
# ноутбуке-отчёте, где sklearn всё равно загружен.
# ===========================================================================

CLASSES = (-1, 0, 1)


def classification_report3(y_true, y_pred) -> dict:
    """Слой-1 (классификация): accuracy, balanced_accuracy, MCC (главная), Cohen κ,
    precision/recall/f1 по классам + macro + weighted, confusion matrix (counts +
    row-normalized). micro-avg == accuracy (отмечено отдельно)."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                  matthews_corrcoef, cohen_kappa_score,
                                  precision_recall_fscore_support, confusion_matrix)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(CLASSES)
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),       # == micro-F1
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class": {int(c): {"precision": float(p[i]), "recall": float(r[i]),
                               "f1": float(f[i]), "support": int(sup[i])}
                      for i, c in enumerate(labels)},
        "macro": {"precision": float(pm), "recall": float(rm), "f1": float(fm)},
        "weighted": {"precision": float(pw), "recall": float(rw), "f1": float(fw)},
        "recall_up": float(r[labels.index(1)]),
        "recall_down": float(r[labels.index(-1)]),
        "confusion": cm, "confusion_norm": cm_norm, "labels": labels,
    }


def _onehot3(y_true) -> np.ndarray:
    idx = {c: i for i, c in enumerate(CLASSES)}
    oh = np.zeros((len(y_true), 3))
    for j, v in enumerate(np.asarray(y_true).astype(int)):
        oh[j, idx[int(v)]] = 1.0
    return oh


def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    """Мультиклассовый top-1 confidence ECE (R11): берём argmax-класс и его proba,
    бьём по бинам уверенности, |acc − conf| взвешенно по населённости бина."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(y_proba, dtype=float)
    conf = proba.max(axis=1)
    pred_idx = proba.argmax(axis=1)
    pred = np.asarray(CLASSES)[pred_idx]
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_curve(y_true, y_proba, n_bins: int = 10):
    """(bin_conf, bin_acc, bin_count) по top-1 уверенности — для reliability-диаграммы."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(y_proba, dtype=float)
    conf = proba.max(axis=1)
    pred = np.asarray(CLASSES)[proba.argmax(axis=1)]
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    confs, accs, counts = [], [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        confs.append(float(conf[m].mean()))
        accs.append(float(correct[m].mean()))
        counts.append(int(m.sum()))
    return np.array(confs), np.array(accs), np.array(counts)


def proba_metrics3(y_true, y_proba) -> dict:
    """Вероятностные метрики: log_loss, ROC-AUC OvR (macro), PR-AUC по классам,
    multiclass Brier, multiclass ECE. NaN там, где класс отсутствует в y_true."""
    from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(y_proba, dtype=float)
    oh = _onehot3(y_true)
    present = oh.sum(axis=0) > 0
    out = {"log_loss": float("nan"), "roc_auc_ovr": float("nan"),
           "brier": float(np.mean(np.sum((proba - oh) ** 2, axis=1))),
           "ece": expected_calibration_error(y_true, proba),
           "pr_auc": {}}
    try:
        out["log_loss"] = float(log_loss(y_true, proba, labels=list(CLASSES)))
    except Exception:  # noqa: BLE001
        pass
    if present.sum() >= 2:
        try:
            out["roc_auc_ovr"] = float(roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro", labels=list(CLASSES)))
        except Exception:  # noqa: BLE001
            pass
    for i, c in enumerate(CLASSES):
        if present[i]:
            try:
                out["pr_auc"][int(c)] = float(average_precision_score(oh[:, i], proba[:, i]))
            except Exception:  # noqa: BLE001
                out["pr_auc"][int(c)] = float("nan")
    return out


def diebold_mariano(e1, e2, h: int = 1, loss: str = "mse"):
    """Diebold–Mariano с поправкой Harvey–Leybourne–Newbold (малая выборка).

    Сравнивает предсказательную точность двух моделей по их ошибкам ``e1``/``e2`` (или
    готовым лоссам, если ``loss='identity'``). Отрицательная статистика => модель 1
    точнее (её лосс ниже). Учитывает h-шаговую автокорреляцию лосс-дифференциала.
    Возвращает (stat, p_value двусторонний). Порт из v2_volengine.vol_evaluate.
    """
    from scipy.stats import t as student_t
    e1 = np.asarray(e1, dtype=float).ravel()
    e2 = np.asarray(e2, dtype=float).ravel()
    mask = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[mask], e2[mask]
    T = e1.size
    if T < 8:
        return float("nan"), float("nan")
    if loss == "mse":
        d = e1 ** 2 - e2 ** 2
    elif loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:  # identity: e1/e2 уже лоссы
        d = e1 - e2
    d_bar = float(np.mean(d))
    # дисперсия среднего с учётом автокорреляции до лага h-1
    gamma0 = float(np.mean((d - d_bar) ** 2))
    var = gamma0
    for k in range(1, h):
        gk = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        var += 2.0 * gk
    var = var / T
    if var <= 0:
        return float("nan"), float("nan")
    dm = d_bar / np.sqrt(var)
    # HLN small-sample correction
    corr = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm * corr
    p = 2.0 * (1.0 - student_t.cdf(abs(dm_hln), df=T - 1))
    return float(dm_hln), float(p)


def precision_at_coverage(signal_score, fwd_returns, side: str = "both",
                          grid=None) -> "object":
    """Кривая precision-at-coverage для направленного решения.

    Ранжируем бары по уверенности и при покрытии ``c`` торгуем самые уверенные ``c``-долю,
    ставя знак сигнала. precision(c) = доля баров, где ``sign(signal)`` совпал со знаком
    реализованной forward-доходности (экономический hit-rate). ``side``:
      * 'both' — ранжируем по |signal|, ставим sign(signal);
      * '+1'   — только long-кандидаты (signal>0), ранжируем по signal убыв.;
      * '-1'   — только short-кандидаты (signal<0), ранжируем по −signal убыв.
    Возвращает DataFrame [coverage, precision, n]."""
    import pandas as pd
    s = np.asarray(signal_score, dtype=float)
    r = np.asarray(fwd_returns, dtype=float)
    m = np.isfinite(s) & np.isfinite(r)
    s, r = s[m], r[m]
    if side == "+1":
        cand = s > 0
        rank = s.copy()
    elif side == "-1":
        cand = s < 0
        rank = -s.copy()
    else:
        cand = np.ones_like(s, dtype=bool)
        rank = np.abs(s)
    s, r, rank = s[cand], r[cand], rank[cand]
    n = s.size
    if n == 0:
        return pd.DataFrame(columns=["coverage", "precision", "n"])
    order = np.argsort(-rank)
    s, r = s[order], r[order]
    hit = (np.sign(s) == np.sign(r)).astype(float)
    grid = grid if grid is not None else np.linspace(0.05, 1.0, 20)
    rows = []
    for c in grid:
        k = max(1, int(np.ceil(c * n)))
        rows.append({"coverage": float(k / n), "precision": float(hit[:k].mean()), "n": int(k)})
    return pd.DataFrame(rows)


def _baseline_preds(y_true, past_return=None, seed: int = 42):
    """majority / stratified-random / persistence(знак прошлой H-доходности) предсказания."""
    y = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(seed)
    vals, counts = np.unique(y, return_counts=True)
    majority = int(vals[np.argmax(counts)])
    freq = counts / counts.sum()
    out = {
        "majority": np.full_like(y, majority),
        "stratified": rng.choice(vals, size=y.size, p=freq),
    }
    if past_return is not None:
        pr = np.asarray(past_return, dtype=float)
        out["persistence"] = np.sign(pr).astype(int)   # знак последней H-доходности
    return out


def evaluate(y_true, y_pred, y_proba, fwd_returns, signal_score, n_trials: int = 1,
             *, past_return=None, ohlc=None, position=None, bt_cfg=None,
             ic_periods=None, trial_sharpes=None, bets_per_year: float = 365.0,
             pbo_matrix=None) -> dict:
    """Полный модуль метрик (слои 1-6) для одной постановки на одном срезе.

    Слой1 классификация + вероятностные; слой2 baselines + дельты; слой3 сигнал (IC);
    слой4 экономика (бэктест vs B&H, если дан `ohlc`); слой5 решение
    (precision-at-coverage, hit по бакетам уверенности); слой6 экзотика
    (PSR/DSR/DM-vs-naive/PBO). ``n_trials`` — для DSR (R12: ВЕСЬ грид конфигураций).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    proba = np.asarray(y_proba, dtype=float)
    fwd = np.asarray(fwd_returns, dtype=float)
    sig = np.asarray(signal_score, dtype=float)

    res = {"n": int(y_true.size)}
    # ---- Layer 1
    res["classification"] = classification_report3(y_true, y_pred)
    res["proba"] = proba_metrics3(y_true, proba)
    # ---- Layer 2 baselines + deltas (MCC)
    base = _baseline_preds(y_true, past_return=past_return)
    res["baselines"] = {}
    for name, bp in base.items():
        rep = classification_report3(y_true, bp)
        res["baselines"][name] = {"mcc": rep["mcc"], "balanced_accuracy": rep["balanced_accuracy"],
                                  "macro_f1": rep["macro"]["f1"]}
    res["baseline_delta_mcc"] = {
        name: res["classification"]["mcc"] - d["mcc"] for name, d in res["baselines"].items()}
    # ---- Layer 3 signal / IC
    ic_global = rank_ic(fwd, sig)
    ic_layer = {"rank_ic": ic_global, "pearson_ic": float(np.corrcoef(
        sig[np.isfinite(sig) & np.isfinite(fwd)], fwd[np.isfinite(sig) & np.isfinite(fwd)])[0, 1])
        if np.isfinite(sig).sum() > 2 else float("nan")}
    if ic_periods is not None:
        import pandas as pd
        ics = []
        dfp = pd.DataFrame({"fwd": fwd, "sig": sig, "g": np.asarray(ic_periods)})
        for _, grp in dfp.groupby("g"):
            ics.append(rank_ic(grp["fwd"].values, grp["sig"].values))
        ics = [x for x in ics if np.isfinite(x)]
        if ics:
            arr = np.asarray(ics)
            ic_layer.update({"ic_mean": float(arr.mean()), "ic_std": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
                             "ic_ir": ic_ir(arr), "ic_share_pos": float((arr > 0).mean()),
                             "ic_tstat": float(arr.mean() / (arr.std(ddof=1) / np.sqrt(arr.size))) if arr.size > 1 and arr.std(ddof=1) > 0 else float("nan"),
                             "n_periods": int(arr.size)})
    res["ic"] = ic_layer
    # ---- Layer 4 economics
    trade_rets = None
    if ohlc is not None and bt_cfg is not None:
        from . import backtest as bt
        import pandas as pd
        pos = position if position is not None else pd.Series(y_pred, index=ohlc.index[:len(y_pred)])
        sim = bt.simulate_horizon_strategy(pos, ohlc, bt_cfg, bets_per_year=bets_per_year)
        bh = bt.buy_and_hold_metrics(ohlc, pos.index, vol_target=bt_cfg.get("vol_target_bh"))
        res["economics"] = {"strategy": sim["mean"], "buy_hold": bh,
                            "delta_vs_bh": bt.strategy_vs_bh(sim["mean"], bh),
                            "per_phase": sim["per_phase"]}
        trade_rets = sim["trade_returns"]
    # ---- Layer 5 decision
    res["decision"] = {
        "pac_both": precision_at_coverage(sig, fwd, side="both"),
        "pac_long": precision_at_coverage(sig, fwd, side="+1"),
        "pac_short": precision_at_coverage(sig, fwd, side="-1"),
    }
    # confidence-bucket hit rate
    conf = proba.max(axis=1)
    pred = np.asarray(CLASSES)[proba.argmax(axis=1)]
    edges = np.linspace(0, 1, 6)
    buckets = {}
    for b in range(5):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.sum():
            buckets[f"({edges[b]:.1f},{edges[b+1]:.1f}]"] = float((pred[m] == y_true[m]).mean())
    res["decision"]["confidence_buckets"] = buckets
    # hit rate на направленных предсказаниях (±1): доля верного ЗНАКА реализованной доходности
    nz = y_pred != 0
    res["decision"]["hit_rate_pm1"] = (
        float((np.sign(y_pred[nz]) == np.sign(fwd[nz])).mean()) if nz.any() else float("nan"))
    # ---- Layer 6 exotic
    exotic = {}
    src_rets = trade_rets if trade_rets is not None and len(trade_rets) else None
    if src_rets is not None:
        exotic["psr"] = psr(src_rets)
        obs_sr = float(np.mean(src_rets) / np.std(src_rets, ddof=1)) if np.std(src_rets, ddof=1) > 0 else float("nan")
        if trial_sharpes is not None and len(trial_sharpes) >= 2:
            exotic["dsr"] = deflated_sharpe(trial_sharpes, obs_sr, T=len(src_rets))
        exotic["n_trials"] = int(n_trials)
    # DM vs naive (persistence) on directional PnL loss
    if past_return is not None:
        loss_model = -(np.sign(sig) * fwd)
        loss_naive = -(np.sign(np.asarray(past_return, dtype=float)) * fwd)
        dm_stat, dm_p = diebold_mariano(loss_model, loss_naive, h=1, loss="identity")
        exotic["dm_vs_naive"] = {"stat": dm_stat, "p_value": dm_p}
    if pbo_matrix is not None:
        exotic["pbo"] = prob_backtest_overfitting(np.asarray(pbo_matrix))
    res["exotic"] = exotic
    return res
