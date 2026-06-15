"""
Вспомогательные метрики оценки моделей.

`regression_metrics`/`directional_accuracy` перенесены из
`Bitcoin-price-prediction/src/evaluation/metrics.py` (используются `hybrid_model.py`
и `walk_forward.py`). `ks_test_returns`/`prediction_interval_coverage` -
дистрибуционные метрики (KS-тест и покрытие доверительного интервала),
которых не было в портируемом коде Ушкова (см. главу 6.5 его диплома).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)

    n = min(len(y_true), len(y_pred), len(reference))
    if n == 0:
        raise ValueError("inputs must be non-empty")

    y_true = y_true[:n]
    y_pred = y_pred[:n]
    reference = reference[:n]

    pred_up = (y_pred - reference) > 0
    true_up = (y_true - reference) > 0

    tp = int(np.sum(pred_up & true_up))
    fp = int(np.sum(pred_up & ~true_up))
    fn = int(np.sum(~pred_up & true_up))

    da = float(np.mean(pred_up == true_up))
    precision_long = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall_long = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "Directional Accuracy": da,
        "Precision Long": precision_long,
        "Recall Long": recall_long,
    }


def ks_test_returns(empirical_returns: np.ndarray, simulated_returns: np.ndarray) -> dict[str, float]:
    """
    Двухвыборочный тест Колмогорова-Смирнова: сравнивает распределение
    реальных лог-доходностей с распределением симулированных моделью.

    Возвращает `{'ks_stat': ..., 'ks_pvalue': ...}`. Большой `ks_pvalue`
    (>0.05) означает, что распределения статистически неразличимы.
    """
    empirical_returns = np.asarray(empirical_returns, dtype=float).reshape(-1)
    simulated_returns = np.asarray(simulated_returns, dtype=float).reshape(-1)

    result = ks_2samp(empirical_returns, simulated_returns)
    return {"ks_stat": float(result.statistic), "ks_pvalue": float(result.pvalue)}


def prediction_interval_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """
    Доля фактических наблюдений, попавших в предсказательный интервал
    [lower, upper]. Для калиброванного 90%-интервала ожидается ~0.9.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)

    return float(np.mean((y_true >= lower) & (y_true <= upper)))
