"""
Адаптеры SDE / Hybrid SDE-ML моделей к интерфейсу `ForecastResult`/`compute_metrics`,
используемому остальными моделями проекта (`ml_models.py`).

Поддерживаемые модели (после `sde_models.py`/`hybrid_model.py`):
    GBMModel, MertonJumpDiffusionModel, HestonModel, HybridSDEMLModel
Все имеют единый интерфейс `fit(prices, dt)` + `predict_point(prices, horizon, dt)`.
SDE-модели (GBM/Merton/Heston) дополнительно имеют `simulate(S0, n_steps, n_paths, dt)`,
Hybrid - `predict_distribution(prices, horizon, n_paths, dt, quantiles)`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_models import ForecastResult, compute_metrics
from eval_metrics import ks_test_returns, prediction_interval_coverage


def evaluate_sde_model(
    model: Any,
    prices_train: np.ndarray,
    prices_test: np.ndarray,
    train_index: pd.Index | None = None,
    test_index: pd.Index | None = None,
    n_paths: int = 1000,
    dt: float = 1.0,
) -> tuple[ForecastResult, dict]:
    """
    Калибрует `model` на `prices_train` и оценивает one-step-ahead прогноз
    на `prices_train` (in-sample) и `prices_test` (out-of-sample).

    Возвращает `(ForecastResult, dist_metrics)`, где `dist_metrics` содержит
    KS-тест и покрытие 90%-доверительного интервала (для GBM/Merton/Heston -
    через `simulate`, для Hybrid - через `predict_distribution`).
    """
    prices_train = np.asarray(prices_train, dtype=float)
    prices_test = np.asarray(prices_test, dtype=float)
    horizon = 1
    lag = getattr(model, "n_lag_returns", 0)

    model.fit(prices_train, dt=dt)

    # ------------------------------------------------------------------
    # In-sample (train) one-step forecasts
    # ------------------------------------------------------------------
    train_pred = np.asarray(model.predict_point(prices_train, horizon=horizon, dt=dt))
    y_train_true = prices_train[lag + horizon: lag + horizon + len(train_pred)]
    n_train_common = min(len(y_train_true), len(train_pred))
    y_train_true = y_train_true[:n_train_common]
    train_pred = train_pred[:n_train_common]

    # ------------------------------------------------------------------
    # Out-of-sample (test) one-step forecasts: prepend last (lag+1) train
    # points so that the first forecast covers prices_test[0].
    # ------------------------------------------------------------------
    context = prices_train[-(lag + 1):]
    prices_concat = np.concatenate([context, prices_test])

    test_pred = np.asarray(model.predict_point(prices_concat, horizon=horizon, dt=dt))
    n_test_common = min(len(test_pred), len(prices_test))
    test_pred = test_pred[:n_test_common]
    y_test_true = prices_test[:n_test_common]

    metrics = {
        "train": compute_metrics(y_true=y_train_true, y_pred=train_pred, y_train=prices_train),
        "test": compute_metrics(y_true=y_test_true, y_pred=test_pred, y_train=prices_train),
    }

    if test_index is not None:
        pred_index = test_index[:n_test_common]
    else:
        pred_index = pd.RangeIndex(n_test_common)

    result = ForecastResult(
        metrics=metrics,
        y_pred=pd.Series(test_pred, index=pred_index),
        model=model,
    )

    # ------------------------------------------------------------------
    # Distributional metrics (KS-test + prediction-interval coverage)
    # ------------------------------------------------------------------
    dist_metrics: dict = {}

    if hasattr(model, "predict_distribution"):
        # Hybrid SDE-ML: квантили напрямую из predict_distribution
        dist = model.predict_distribution(
            prices_concat, horizon=horizon, n_paths=n_paths, dt=dt, quantiles=(0.05, 0.5, 0.95)
        )
        pi_lower = np.asarray(dist["0.05"])[:n_test_common]
        pi_upper = np.asarray(dist["0.95"])[:n_test_common]
        pi_median = np.asarray(dist["0.5"])[:n_test_common]

        # log-доходности симулированной медианы относительно S_t (приближение для KS-теста)
        s_t = prices_concat[lag: lag + n_test_common]
        sim_returns = np.log(pi_median / s_t)

    elif hasattr(model, "simulate"):
        # GBM / Merton / Heston: симулируем по 1 шагу из каждой точки контекста
        pi_lower = np.empty(n_test_common)
        pi_upper = np.empty(n_test_common)
        sim_returns_list = []

        for i in range(n_test_common):
            S0 = prices_concat[i]
            paths = model.simulate(S0, n_steps=horizon, n_paths=n_paths, dt=dt)
            terminal = paths[:, -1]
            pi_lower[i] = np.percentile(terminal, 5)
            pi_upper[i] = np.percentile(terminal, 95)
            sim_returns_list.append(terminal / S0 - 1.0)

        sim_returns = np.concatenate(sim_returns_list)
    else:
        pi_lower = pi_upper = sim_returns = None

    if pi_lower is not None:
        actual_log_returns = np.diff(np.log(np.concatenate([context[-1:], y_test_true])))
        ks = ks_test_returns(actual_log_returns, sim_returns)
        coverage = prediction_interval_coverage(y_test_true, pi_lower, pi_upper)
        dist_metrics = {
            "ks_stat": ks["ks_stat"],
            "ks_pvalue": ks["ks_pvalue"],
            "pi_coverage_90": coverage,
            "pi_lower": pi_lower,
            "pi_upper": pi_upper,
        }

    return result, dist_metrics
