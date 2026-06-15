"""
Hybrid SDE-ML модель цены BTC.

Перенесено и адаптировано из `Bitcoin-price-prediction/src/models/hybrid_model.py`
(дипломная работа Ушкова Л.И., раздел 4.6).

Идея: классический GBM использует постоянные дрейф mu и волатильность sigma.
Эта модель заменяет постоянный дрейф на изменяющийся во времени mu_t,
предсказываемый Gradient Boosting регрессором на каждом шаге, сохраняя
стохастическую диффузионную составляющую из GBM:

    S_{t+h} = S_t * exp( mu_hat_t * h*dt  +  sigma_t * sqrt(h*dt) * Z )

где
    mu_hat_t = f_ML(лаги лог-доходностей, realised vol, RSI-momentum proxy)
    sigma_t  = rolling realised volatility

Для точечного прогноза (Z = 0):
    S_hat_{t+h} = S_t * exp( mu_hat_t * h * dt )
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform

from eval_metrics import regression_metrics


def _positive_ratio(values: np.ndarray) -> float:
    return float(np.sum(values > 0)) / max(len(values), 1)


class BaseHybridSDEMLModel:
    """
    Hybrid SDE-ML модель с изменяющимся во времени ML-предсказанным дрейфом.

    Параметры
    ---------
    rv_window        : окно для оценки realised volatility (баров)
    n_lag_returns    : число лагов лог-доходностей как признаков ML
    random_state     : сид для воспроизводимости
    """

    def __init__(
        self,
        model_name: str,
        drift_model,
        rv_window: int = 20,
        n_lag_returns: int = 10,
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.rv_window = rv_window
        self.n_lag_returns = n_lag_returns
        self.random_state = random_state
        self._drift_model = drift_model
        self.is_fitted = False
        self.best_params_: dict | None = None
        self.search_results_ = None

    # ------------------------------------------------------------------
    # Построение признаков
    # ------------------------------------------------------------------

    def _compute_rv(self, log_returns: np.ndarray, dt: float) -> np.ndarray:
        return (
            pd.Series(log_returns)
            .rolling(self.rv_window, min_periods=self.rv_window)
            .std()
            .bfill()
            .values
        ) / math.sqrt(dt)

    def _build_features(
        self, log_returns: np.ndarray, rv: np.ndarray
    ) -> np.ndarray:
        """
        Матрица признаков для ML-предсказателя дрейфа.

        Колонки: lag_1 ... lag_n  (лаги лог-доходностей, от самого свежего)
                 realised_vol     (текущая rolling RV)
                 rsi_14           (доля положительных доходностей за 14 баров)
        """
        lag = self.n_lag_returns
        rows = []
        for i in range(lag, len(log_returns)):
            lags = log_returns[i - lag : i][::-1]
            rv_val = float(rv[i]) if i < len(rv) else float(rv[-1])
            window_14 = log_returns[max(0, i - 14) : i]
            rsi = _positive_ratio(window_14)
            rows.append(np.concatenate([lags, [rv_val, rsi]]))
        return np.array(rows, dtype=np.float32)

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------

    def fit(self, prices: np.ndarray, dt: float = 1.0) -> "BaseHybridSDEMLModel":
        """Калибрует hybrid-модель на исторической серии цен."""
        prices = np.asarray(prices, dtype=float)
        log_returns = np.diff(np.log(prices))
        self._dt = dt

        rv = self._compute_rv(log_returns, dt)
        self._rv_last = rv

        lag = self.n_lag_returns
        X = self._build_features(log_returns, rv)   # (n-lag, p)
        y = log_returns[lag:]                        # предсказываем след. лог-доходность

        self._drift_model.fit(X, y)
        self.is_fitted = True
        return self

    def tune(
        self,
        prices: np.ndarray,
        param_distributions: dict,
        dt: float = 1.0,
        n_iter: int = 20,
        cv_splits: int = 3,
        scoring: str = "neg_mean_absolute_error",
        n_jobs: int = -1,
        verbose: int = 1,
    ) -> "BaseHybridSDEMLModel":
        prices = np.asarray(prices, dtype=float)
        log_returns = np.diff(np.log(prices))
        rv = self._compute_rv(log_returns, dt)
        lag = self.n_lag_returns
        X = self._build_features(log_returns, rv)
        y = log_returns[lag:]

        search = RandomizedSearchCV(
            estimator=self._drift_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=TimeSeriesSplit(n_splits=cv_splits),
            random_state=self.random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            refit=True,
        )
        search.fit(X, y)
        self._drift_model = search.best_estimator_
        self.best_params_ = search.best_params_
        self.search_results_ = pd.DataFrame(search.cv_results_).sort_values(
            "rank_test_score"
        )
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Прогноз
    # ------------------------------------------------------------------

    def predict_point(
        self,
        prices: np.ndarray,
        horizon: int = 10,
        dt: float = 1.0,
    ) -> np.ndarray:
        """
        Rolling точечный прогноз (детерминированная траектория при Z = 0).

            S_hat_{t+h} ≈ S_t * exp( mu_hat_t * h * dt )

        Возвращает массив длины len(prices) - n_lag_returns - horizon - 1.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")

        prices = np.asarray(prices, dtype=float)
        log_returns = np.diff(np.log(prices))
        rv = self._compute_rv(log_returns, dt)

        lag = self.n_lag_returns
        X_all = self._build_features(log_returns, rv)
        mu_preds = self._drift_model.predict(X_all)

        forecasts = []
        for i in range(len(mu_preds) - horizon):
            S_t = prices[lag + i]
            mu_t = float(mu_preds[i])
            forecasts.append(S_t * math.exp(mu_t * horizon * dt))

        return np.array(forecasts)

    def predict_distribution(
        self,
        prices: np.ndarray,
        horizon: int = 10,
        n_paths: int = 1000,
        dt: float = 1.0,
        quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    ) -> dict[str, np.ndarray]:
        """
        Дистрибуционный прогноз: на каждом шаге симулируется n_paths путей
        GBM с ML-предсказанным дрейфом и realised volatility, возвращаются
        квантили.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")

        prices = np.asarray(prices, dtype=float)
        log_returns = np.diff(np.log(prices))
        rv = self._compute_rv(log_returns, dt)

        lag = self.n_lag_returns
        X_all = self._build_features(log_returns, rv)
        mu_preds = self._drift_model.predict(X_all)

        rng = np.random.default_rng(self.random_state)
        results: dict[str, list] = {str(q): [] for q in quantiles}

        for i in range(len(mu_preds) - horizon):
            S_t = prices[lag + i]
            mu_t = float(mu_preds[i])
            sigma_t = float(rv[lag + i]) if (lag + i) < len(rv) else float(rv[-1])

            Z = rng.standard_normal((n_paths, horizon))
            step_log = (mu_t - 0.5 * sigma_t ** 2) * dt + sigma_t * math.sqrt(dt) * Z
            S_T = S_t * np.exp(step_log.sum(axis=1))

            for q in quantiles:
                results[str(q)].append(float(np.quantile(S_T, q)))

        return {k: np.array(v) for k, v in results.items()}

    # ------------------------------------------------------------------
    # Оценка
    # ------------------------------------------------------------------

    def evaluate(
        self,
        prices: np.ndarray,
        horizon: int = 10,
        dt: float = 1.0,
    ) -> dict:
        """Точность точечного прогноза на серии цен."""
        y_pred = self.predict_point(prices, horizon, dt)
        lag = self.n_lag_returns
        y_true = prices[lag + horizon : lag + horizon + len(y_pred)]

        return {"model": self.model_name, **regression_metrics(y_true, y_pred)}

    # ------------------------------------------------------------------
    # Важность признаков
    # ------------------------------------------------------------------

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")
        if hasattr(self._drift_model, "feature_importances_"):
            return self._drift_model.feature_importances_
        if hasattr(self._drift_model, "coef_"):
            return np.abs(np.asarray(self._drift_model.coef_)).reshape(-1)
        raise AttributeError("Drift model does not expose importances or coefficients.")

    def feature_importance_df(self) -> pd.DataFrame:
        lag = self.n_lag_returns
        names = [f"lag_{i+1}" for i in range(lag)] + ["realised_vol", "rsi_14"]
        df = pd.DataFrame({"feature": names, "importance": self.feature_importances_})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Графики
    # ------------------------------------------------------------------

    def plot_forecast(
        self,
        prices: np.ndarray,
        horizon: int = 10,
        dt: float = 1.0,
        n_points: int = 200,
    ):
        y_pred = self.predict_point(prices, horizon, dt)
        lag = self.n_lag_returns
        y_true = prices[lag + horizon : lag + horizon + len(y_pred)]

        if n_points:
            y_pred = y_pred[:n_points]
            y_true = y_true[:n_points]

        plt.figure(figsize=(12, 5))
        plt.plot(y_true, label="Actual", linewidth=1.5)
        plt.plot(y_pred, label="Hybrid (point)", linewidth=1.5, linestyle="--")
        plt.title(f"{self.model_name}: Rolling {horizon}-step Forecast")
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class HybridSDEMLModel(BaseHybridSDEMLModel):
    def __init__(
        self,
        rv_window: int = 20,
        n_lag_returns: int = 10,
        gb_n_estimators: int = 200,
        gb_learning_rate: float = 0.05,
        gb_max_depth: int = 3,
        random_state: int = 42,
    ):
        super().__init__(
            model_name="Hybrid SDE-ML (GBM + GradientBoosting drift)",
            drift_model=GradientBoostingRegressor(
                n_estimators=gb_n_estimators,
                learning_rate=gb_learning_rate,
                max_depth=gb_max_depth,
                random_state=random_state,
            ),
            rv_window=rv_window,
            n_lag_returns=n_lag_returns,
            random_state=random_state,
        )

    @staticmethod
    def tuning_space() -> dict:
        return {
            "n_estimators": randint(100, 500),
            "learning_rate": uniform(0.01, 0.19),
            "max_depth": randint(2, 6),
            "subsample": uniform(0.7, 0.3),
            "min_samples_leaf": randint(1, 10),
        }
