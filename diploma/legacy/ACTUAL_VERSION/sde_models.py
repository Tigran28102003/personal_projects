"""
Стохастические модели цены BTC (SDE).

Перенесено и адаптировано из `Bitcoin-price-prediction/src/models/sde_models.py`
(дипломная работа Ушкова Л.И.). Содержит:

* `GBMModel`              - Geometric Brownian Motion (Black-Scholes)
* `MertonJumpDiffusionModel` - Merton (1976) Jump-Diffusion
* `HestonModel`           - Heston (1993) Stochastic Volatility

Все модели калибруются на 1D массиве цен через `fit(prices, dt)` и умеют
симулировать траектории через `simulate(S0, n_steps, n_paths, dt)`.
`predict_point` - алиас над `rolling_forecast` для единообразного интерфейса
с Hybrid SDE-ML моделью (используется в `sde_adapters.py` и `walk_forward.py`).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import gammaln
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class BaseSDEModel(ABC):
    """Базовый класс для стохастических моделей цены."""

    def __init__(self, model_name: str, random_state: int = 42):
        self.model_name = model_name
        self.random_state = random_state
        self.params: dict = {}
        self.is_fitted = False
        self.default_n_paths = 1000

    @abstractmethod
    def fit(self, prices: np.ndarray, dt: float = 1.0) -> "BaseSDEModel":
        """Калибровка параметров модели по исторической серии цен."""

    @abstractmethod
    def simulate(
        self, S0: float, n_steps: int, n_paths: int = 1000, dt: float = 1.0
    ) -> np.ndarray:
        """Возвращает симулированные траектории, shape (n_paths, n_steps + 1)."""

    # ------------------------------------------------------------------
    # Общие методы
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(f"{self.model_name} must be fitted before use.")

    def predict_median(
        self, S0: float, n_steps: int, n_paths: int = 1000, dt: float = 1.0
    ) -> np.ndarray:
        """Медианная траектория среди симулированных (shape: n_steps + 1)."""
        paths = self.simulate(S0, n_steps, n_paths, dt)
        return np.median(paths, axis=0)

    def rolling_forecast(
        self,
        prices: np.ndarray,
        horizon: int = 1,
        n_paths: int = 1000,
        dt: float = 1.0,
    ) -> np.ndarray:
        """One-step-ahead rolling point-прогноз (медиана симулированных путей).

        Для каждого t симулирует `horizon` шагов из prices[t] и берёт медиану
        терминального значения. Возвращает массив длины len(prices) - horizon.
        """
        self._check_fitted()
        forecasts = np.empty(len(prices) - horizon)
        for i in range(len(prices) - horizon):
            paths = self.simulate(prices[i], horizon, n_paths, dt)
            forecasts[i] = np.median(paths[:, -1])
        return forecasts

    def predict_point(
        self,
        prices: np.ndarray,
        horizon: int = 1,
        n_paths: int = 1000,
        dt: float = 1.0,
    ) -> np.ndarray:
        """Алиас над `rolling_forecast` для единого интерфейса с Hybrid SDE-ML."""
        return self.rolling_forecast(prices, horizon=horizon, n_paths=n_paths, dt=dt)

    def evaluate(
        self,
        prices: np.ndarray,
        horizon: int = 1,
        n_paths: int = 1000,
        dt: float = 1.0,
    ) -> dict:
        """Оценка точности rolling-прогноза относительно реальных цен."""
        forecasts = self.rolling_forecast(prices, horizon, n_paths, dt)
        actuals = prices[horizon:]
        mae = mean_absolute_error(actuals, forecasts)
        rmse = np.sqrt(mean_squared_error(actuals, forecasts))
        r2 = r2_score(actuals, forecasts)
        return {"model": self.model_name, "MAE": mae, "RMSE": rmse, "R2": r2}

    def summary(self) -> str:
        """Человеко-читаемое резюме калиброванных параметров."""
        self._check_fitted()
        lines = [f"=== {self.model_name} ==="]
        for k, v in self.params.items():
            lines.append(f"  {k:12s}: {v:.6f}")
        lines.append(f"  {'n_paths':12s}: {self.default_n_paths:d}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Графики
    # ------------------------------------------------------------------

    def plot_simulations(
        self,
        S0: float,
        n_steps: int,
        n_paths: int = 300,
        dt: float = 1.0,
        actual_prices: np.ndarray | None = None,
        title: str | None = None,
        ax=None,
    ):
        """Fan chart симулированных траекторий с опциональным наложением реальных цен."""
        self._check_fitted()
        paths = self.simulate(S0, n_steps, n_paths, dt)
        t = np.arange(n_steps + 1) * dt

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))

        for path in paths[:50]:
            ax.plot(t, path, alpha=0.08, color="steelblue", linewidth=0.6)

        for q_lo, q_hi, alpha, label in [
            (5, 95, 0.15, "5–95 %"),
            (25, 75, 0.25, "25–75 %"),
        ]:
            lo = np.percentile(paths, q_lo, axis=0)
            hi = np.percentile(paths, q_hi, axis=0)
            ax.fill_between(t, lo, hi, alpha=alpha, color="steelblue", label=label)

        ax.plot(t, np.median(paths, axis=0), color="steelblue", linewidth=2, label="Median")

        if actual_prices is not None:
            n_actual = min(len(actual_prices), n_steps + 1)
            ax.plot(
                t[:n_actual],
                actual_prices[:n_actual],
                color="firebrick",
                linewidth=1.5,
                label="Actual",
            )

        ax.set_xlabel("Time steps")
        ax.set_ylabel("Price (USD)")
        ax.set_title(title or f"{self.model_name} — Simulated Paths")
        ax.legend()
        plt.tight_layout()
        return ax

    def plot_forecast(
        self,
        prices: np.ndarray,
        horizon: int = 1,
        n_paths: int = 1000,
        dt: float = 1.0,
        title: str | None = None,
        n_points: int = 200,
    ):
        """Rolling медианный прогноз vs реальные цены."""
        self._check_fitted()
        forecasts = self.rolling_forecast(prices, horizon, n_paths, dt)
        actuals = prices[horizon:]
        if n_points:
            forecasts = forecasts[:n_points]
            actuals = actuals[:n_points]

        plt.figure(figsize=(12, 5))
        plt.plot(actuals, label="Actual", linewidth=1.5)
        plt.plot(forecasts, label="Forecast (median)", linewidth=1.5, linestyle="--")
        plt.title(title or f"{self.model_name}: Rolling {horizon}-step Forecast")
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ======================================================================
# 1. Geometric Brownian Motion
# ======================================================================

class GBMModel(BaseSDEModel):
    """Geometric Brownian Motion (Black-Scholes dynamics).

    SDE:  dS = μ S dt + σ S dW

    Калибровка: MLE по лог-доходностям.
        σ = std(r) / √dt,   μ = mean(r)/dt + ½σ²
    Симуляция: точные логнормальные приращения.
    """

    def __init__(self, random_state: int = 42):
        super().__init__("Geometric Brownian Motion", random_state)
        self.mu: float | None = None
        self.sigma: float | None = None

    def fit(self, prices: np.ndarray, dt: float = 1.0) -> "GBMModel":
        log_returns = np.diff(np.log(prices))
        self.sigma = np.std(log_returns, ddof=1) / math.sqrt(dt)
        self.mu = np.mean(log_returns) / dt + 0.5 * self.sigma ** 2
        self.params = {"mu": self.mu, "sigma": self.sigma}
        self.is_fitted = True
        return self

    def simulate(
        self, S0: float, n_steps: int, n_paths: int = 1000, dt: float = 1.0
    ) -> np.ndarray:
        self._check_fitted()
        rng = np.random.default_rng(self.random_state)
        Z = rng.standard_normal((n_paths, n_steps))
        increments = (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * math.sqrt(dt) * Z
        log_paths = np.cumsum(increments, axis=1)
        return S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))


# ======================================================================
# 2. Merton Jump-Diffusion
# ======================================================================

class MertonJumpDiffusionModel(BaseSDEModel):
    """Merton (1976) Jump-Diffusion model.

    SDE:  dS = (μ - λκ) S dt + σ S dW + S dJ

    J - сложный пуассоновский процесс с нормально распределёнными лог-скачками:
        log(1 + J_k) ~ N(μ_j, σ_j²),   κ = exp(μ_j + ½σ_j²) - 1.

    Калибровка: MLE через Poisson-Gaussian mixture на лог-доходностях.
    Симуляция: точные приращения compound-Poisson.
    """

    def __init__(self, random_state: int = 42, max_jump_terms: int = 20):
        super().__init__("Merton Jump-Diffusion", random_state)
        self.max_jump_terms = max_jump_terms
        self.mu: float | None = None
        self.sigma: float | None = None
        self.lam: float | None = None      # интенсивность скачков (скачков за единицу времени)
        self.mu_j: float | None = None     # средний размер лог-скачка
        self.sigma_j: float | None = None  # std размера лог-скачка

    def _neg_log_likelihood(
        self, params: np.ndarray, log_returns: np.ndarray, dt: float
    ) -> float:
        mu, sigma, lam, mu_j, sigma_j = params
        if sigma <= 0 or lam < 0 or sigma_j <= 0:
            return 1e12

        k = np.arange(self.max_jump_terms)  # (K,)

        # Лог-веса Пуассона: log[ e^{-λdt} (λdt)^k / k! ]
        lam_dt = max(lam * dt, 1e-300)
        log_pw = -lam * dt + k * math.log(lam_dt) - gammaln(k + 1)
        pw = np.exp(log_pw)  # (K,)

        # Средние и дисперсии компонент смеси для каждого k
        drifts = (mu - 0.5 * sigma ** 2) * dt + k * mu_j  # (K,)
        variances = np.maximum(sigma ** 2 * dt + k * sigma_j ** 2, 1e-12)  # (K,)

        # Векторизованный PDF по всем доходностям: shape (n, K)
        r = log_returns[:, None]
        pdfs = norm.pdf(r, loc=drifts[None, :], scale=np.sqrt(variances)[None, :])

        mixture = np.maximum(np.dot(pdfs, pw), 1e-300)  # (n,)
        return -np.sum(np.log(mixture))

    def fit(self, prices: np.ndarray, dt: float = 1.0) -> "MertonJumpDiffusionModel":
        log_returns = np.diff(np.log(prices))

        sigma0 = np.std(log_returns, ddof=1) / math.sqrt(dt)
        mu0 = np.mean(log_returns) / dt + 0.5 * sigma0 ** 2
        x0 = [mu0, sigma0 * 0.8, 1.0, 0.0, 0.05]
        bounds = [
            (None, None),
            (1e-6, None),
            (0.0, None),
            (None, None),
            (1e-6, None),
        ]

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(log_returns, dt),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        self.mu, self.sigma, self.lam, self.mu_j, self.sigma_j = result.x
        kappa = math.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1
        self.params = {
            "mu": self.mu,
            "sigma": self.sigma,
            "lambda": self.lam,
            "mu_j": self.mu_j,
            "sigma_j": self.sigma_j,
            "kappa": kappa,
        }
        self.is_fitted = True
        return self

    def simulate(
        self, S0: float, n_steps: int, n_paths: int = 1000, dt: float = 1.0
    ) -> np.ndarray:
        self._check_fitted()
        rng = np.random.default_rng(self.random_state)

        kappa = math.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1
        drift = (self.mu - self.lam * kappa - 0.5 * self.sigma ** 2) * dt

        Z = rng.standard_normal((n_paths, n_steps))
        diffusion = drift + self.sigma * math.sqrt(dt) * Z

        # Compound Poisson: N ~ Poisson(λ dt), сумма N iid N(μ_j, σ_j²)
        # ~ N(N μ_j,  N σ_j²)
        N = rng.poisson(self.lam * dt, (n_paths, n_steps)).astype(float)
        Z_j = rng.standard_normal((n_paths, n_steps))
        jumps = N * self.mu_j + np.sqrt(N) * self.sigma_j * Z_j

        log_paths = np.cumsum(diffusion + jumps, axis=1)
        return S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))


# ======================================================================
# 3. Heston Stochastic Volatility
# ======================================================================

class HestonModel(BaseSDEModel):
    """Heston (1993) Stochastic Volatility model.

    SDEs:
        dS = μ S dt + √v S dW₁
        dv = κ(θ - v) dt + ξ √v dW₂,   corr(dW₁, dW₂) = ρ

    Параметры
    ---------
    mu    : дрейф процесса цены
    kappa : скорость возврата к среднему дисперсии
    theta : долгосрочная дисперсия (σ_∞²)
    xi    : "vol of vol"
    rho   : корреляция между броуновскими движениями цены и дисперсии
    v0    : начальная дисперсия

    Калибровка: moment-matching на realized variance.
    Симуляция: Euler-Maruyama с full truncation для процесса дисперсии.
    """

    def __init__(self, random_state: int = 42, rv_window: int = 30):
        super().__init__("Heston Stochastic Volatility", random_state)
        self.rv_window = rv_window
        self.calibration_dt: float | None = None
        self.mu: float | None = None
        self.kappa: float | None = None
        self.theta: float | None = None
        self.xi: float | None = None
        self.rho: float | None = None
        self.v0: float | None = None

    def fit(self, prices: np.ndarray, dt: float = 1.0) -> "HestonModel":
        """
        Калибрует модель Хестона напрямую на частоте входных данных.

        Важно: эта реализация не калибруется на дневных данных с последующим
        масштабированием параметров на внутридневную частоту. Вместо этого
        вызывающий код должен передавать серию цен на целевой частоте вместе
        с соответствующим шагом времени, например dt=1/96 для 15-минутных
        баров.
        """
        log_returns = np.diff(np.log(prices))
        n = len(log_returns)
        self.calibration_dt = dt

        # Дрейф
        var_r = np.var(log_returns, ddof=1)
        self.mu = float(np.mean(log_returns) / dt + 0.5 * var_r / dt)

        # Реализованная дисперсия (rolling window квадратов доходностей)
        window = min(self.rv_window, max(n // 10, 5))
        rv = (
            pd.Series(log_returns ** 2)
            .rolling(window, min_periods=window)
            .mean()
            .dropna()
            .values
        ) / dt

        self.v0 = float(var_r / dt)
        self.theta = float(np.mean(rv))

        # Скорость возврата к среднему через AR(1) на реализованной дисперсии
        if len(rv) > 2:
            rv_lag, rv_curr = rv[:-1], rv[1:]
            beta = np.cov(rv_curr, rv_lag)[0, 1] / (np.var(rv_lag) + 1e-12)
            beta = float(np.clip(beta, 1e-6, 1.0 - 1e-6))
            self.kappa = float(max(-math.log(beta) / dt, 0.1))
        else:
            self.kappa = 2.0

        # Vol-of-vol из std приращений реализованной дисперсии
        if len(rv) > 2:
            rv_diff = np.diff(rv)
            self.xi = float(
                np.std(rv_diff, ddof=1) / (math.sqrt(self.theta + 1e-12) * math.sqrt(dt))
            )
            self.xi = max(self.xi, 1e-4)
            # Мягкое условие Феллера: 2κθ > ξ² сохраняет дисперсию положительной
            feller_xi_max = math.sqrt(2 * self.kappa * self.theta)
            if self.xi > feller_xi_max:
                self.xi = max(feller_xi_max * 0.9, 1e-4)
        else:
            self.xi = 0.3

        # Корреляция: взаимная корреляция доходностей и приращений RV
        rv_changes = np.diff(rv)
        r_trim = log_returns[window: window + len(rv_changes)]
        if len(r_trim) > 10:
            self.rho = float(np.clip(np.corrcoef(r_trim, rv_changes)[0, 1], -0.99, 0.99))
        else:
            self.rho = -0.5

        self.params = {
            "mu": self.mu,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
            "v0": self.v0,
            "dt": self.calibration_dt,
        }
        self.is_fitted = True
        return self

    def simulate(
        self, S0: float, n_steps: int, n_paths: int = 1000, dt: float = 1.0
    ) -> np.ndarray:
        """Euler-Maruyama со схемой full truncation для дисперсии."""
        self._check_fitted()
        rng = np.random.default_rng(self.random_state)

        sqrt_rho_sq = math.sqrt(max(1.0 - self.rho ** 2, 0.0))
        sqrt_dt = math.sqrt(dt)

        S = np.full(n_paths, S0, dtype=float)
        v = np.full(n_paths, self.v0, dtype=float)
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = S0

        for t in range(n_steps):
            Z1 = rng.standard_normal(n_paths)
            Z2 = self.rho * Z1 + sqrt_rho_sq * rng.standard_normal(n_paths)

            v_plus = np.maximum(v, 0.0)  # full truncation: ограничение перед использованием
            sqrt_v = np.sqrt(v_plus)

            S = S * np.exp((self.mu - 0.5 * v_plus) * dt + sqrt_v * sqrt_dt * Z1)
            v = v + self.kappa * (self.theta - v_plus) * dt + self.xi * sqrt_v * sqrt_dt * Z2

            paths[:, t + 1] = S

        return paths
