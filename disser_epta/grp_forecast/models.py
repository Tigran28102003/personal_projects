from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ID_COLUMNS = ["region_id", "year"]
DEFAULT_MISSING_DROP_THRESHOLD = 0.4


class BaseForecaster(ABC):
    name: str

    def __init__(self, **params):
        self.name = self.__class__.__name__
        self.params = dict(params)

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        ...

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        ...

    def get_params(self) -> dict:
        return dict(self.params)

    def set_params(self, **params) -> None:
        if "missing_drop_threshold" in params:
            setattr(self, "missing_drop_threshold", float(params.pop("missing_drop_threshold")))
        self.params.update(params)

    def clone(self) -> "BaseForecaster":
        return copy.deepcopy(self)


def _region_ids(X: pd.DataFrame) -> pd.Series:
    if "region_id" in X.columns:
        return X["region_id"].astype(str)
    if "region_id" in X.index.names:
        return pd.Series(X.index.get_level_values("region_id").astype(str), index=X.index)
    return pd.Series("unknown", index=X.index)


def _years(X: pd.DataFrame) -> pd.Series:
    if "year" in X.columns:
        return pd.Series(pd.to_numeric(X["year"], errors="coerce"), index=X.index)
    if "year" in X.index.names:
        return pd.Series(pd.to_numeric(X.index.get_level_values("year"), errors="coerce"), index=X.index)
    return pd.Series(np.nan, index=X.index)


def _numeric_features(X: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    if feature_columns is None:
        drop_cols = [col for col in ID_COLUMNS if col in X.columns]
        X_num = X.drop(columns=drop_cols, errors="ignore")
        X_num = X_num.select_dtypes(include=[np.number, "bool"])
    else:
        X_num = X.reindex(columns=feature_columns, fill_value=0)
    X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X_num.astype(float)


def _fit_numeric_preprocessor(
    X: pd.DataFrame,
    missing_drop_threshold: float = DEFAULT_MISSING_DROP_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    drop_cols = [col for col in ID_COLUMNS if col in X.columns]
    X_num = X.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number, "bool"])
    X_num = X_num.replace([np.inf, -np.inf], np.nan)
    missing_ratios = X_num.isna().mean()
    keep_columns = missing_ratios[missing_ratios <= missing_drop_threshold].index.tolist()
    dropped_columns = [col for col in X_num.columns if col not in keep_columns]
    X_keep = X_num[keep_columns]
    medians = X_keep.median(numeric_only=True).fillna(0.0)
    state = {
        "feature_columns": keep_columns,
        "impute_values": medians,
        "missing_ratios": missing_ratios,
        "dropped_columns": dropped_columns,
        "missing_drop_threshold": float(missing_drop_threshold),
    }
    return X_keep.fillna(medians).astype(float), state


def _transform_numeric_preprocessor(X: pd.DataFrame, state: dict) -> pd.DataFrame:
    columns = state["feature_columns"]
    impute_values = state["impute_values"]
    X_num = X.reindex(columns=columns)
    X_num = X_num.replace([np.inf, -np.inf], np.nan)
    return X_num.fillna(impute_values).fillna(0.0).astype(float)


def _rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    y_arr = np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean((y_arr - y_pred) ** 2)))


def _last_year_validation_split(
    X: pd.DataFrame, inner_val_years: int = 2
) -> tuple[pd.Series, pd.Series]:
    years = _years(X)
    unique_years = sorted(years.dropna().astype(int).unique().tolist())
    if len(unique_years) <= inner_val_years:
        train_mask = pd.Series(True, index=X.index)
        val_mask = pd.Series(False, index=X.index)
        return train_mask, val_mask
    val_years = set(unique_years[-inner_val_years:])
    val_mask = years.astype("Int64").isin(val_years)
    train_mask = ~val_mask
    return train_mask.astype(bool), val_mask.astype(bool)


class NaiveForecaster(BaseForecaster):
    """
    Level mode (default): forecast = log_grp_lag1 (random walk).
    Growth mode (predict_growth=True): forecast = mean(Δlog_grp_train).
    """

    def __init__(self, predict_growth: bool = False):
        super().__init__(predict_growth=predict_growth)
        self.predict_growth: bool = predict_growth
        self.train_mean_: float = 0.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.train_mean_ = float(y_train.mean())

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.predict_growth:
            return np.full(len(X_test), self.train_mean_)
        if "log_grp_lag1" not in X_test.columns:
            raise ValueError("NaiveForecaster requires log_grp_lag1 in X_test.")
        return X_test["log_grp_lag1"].astype(float).to_numpy()


class AR1Forecaster(BaseForecaster):
    """
    Region-wise OLS AR(1).

    Level mode (predict_growth=False, default):
        log_grp_{i,t} = alpha_i + rho_i * log_grp_{i,t-1}
        Regressor: log_grp_lag1 (level).

    Growth mode (predict_growth=True):
        Δlog_grp_{i,t} = alpha_i + rho_i * Δlog_grp_{i,t-1}
        Regressor: grp_growth_lag1 (lagged growth rate).
        This is a proper AR(1) in growth-rate space.

    Regions with fewer than three usable observations fall back to naive
    (mean of y_train for that region, or global mean if unavailable).
    """

    def __init__(self, predict_growth: bool = False):
        super().__init__(predict_growth=predict_growth)
        self.predict_growth: bool = predict_growth
        self.coefs_: dict[str, tuple[float, float]] = {}
        self.global_mean_: float = 0.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        x_col = "grp_growth_lag1" if self.predict_growth else "log_grp_lag1"
        if x_col not in X_train.columns:
            raise ValueError(f"AR1Forecaster requires '{x_col}' in X_train.")
        regions = _region_ids(X_train)
        self.global_mean_ = float(y_train.mean())
        data = pd.DataFrame(
            {
                "region_id": regions,
                "x": X_train[x_col].astype(float),
                "y": y_train.astype(float),
            },
            index=X_train.index,
        ).dropna()
        self.coefs_ = {}
        for region_id, group in data.groupby(data["region_id"], sort=False):
            if len(group) < 3:
                continue
            design = np.column_stack([np.ones(len(group)), group["x"].to_numpy()])
            alpha, rho = np.linalg.lstsq(design, group["y"].to_numpy(), rcond=None)[0]
            self.coefs_[str(region_id)] = (float(alpha), float(rho))
        return None

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        x_col = "grp_growth_lag1" if self.predict_growth else "log_grp_lag1"
        if x_col not in X_test.columns:
            raise ValueError(f"AR1Forecaster requires '{x_col}' in X_test.")
        lag = X_test[x_col].astype(float).to_numpy()
        regions = _region_ids(X_test).to_numpy()
        pred = np.empty(len(X_test), dtype=float)
        for idx, region_id in enumerate(regions):
            alpha_rho = self.coefs_.get(str(region_id))
            if alpha_rho is None:
                pred[idx] = self.global_mean_
            else:
                pred[idx] = alpha_rho[0] + alpha_rho[1] * lag[idx]
        return pred


class RidgeFEForecaster(BaseForecaster):
    """
    Ridge regression with region fixed effects through within-transformation.
    Lambda is selected by inner expanding-window validation over the last
    inner_val_years of the training set.
    """

    def __init__(
        self,
        lambda_grid: list[float] | None = None,
        inner_val_years: int = 2,
        random_state: int = 42,
        missing_drop_threshold: float = DEFAULT_MISSING_DROP_THRESHOLD,
    ):
        super().__init__(
            lambda_grid=lambda_grid or [0.01, 0.1, 1, 10, 100, 1000],
            inner_val_years=inner_val_years,
            random_state=random_state,
            missing_drop_threshold=missing_drop_threshold,
        )
        self.state_: dict | None = None
        self.alpha_: float | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        train_mask, val_mask = _last_year_validation_split(
            X_train, inner_val_years=int(self.params["inner_val_years"])
        )
        lambda_grid = [float(value) for value in self.params["lambda_grid"]]
        best_alpha = lambda_grid[0]
        best_rmse = np.inf
        if val_mask.any() and train_mask.any():
            for alpha in lambda_grid:
                state = self._fit_state(X_train.loc[train_mask], y_train.loc[train_mask], alpha)
                pred = self._predict_with_state(X_train.loc[val_mask], state)
                score = _rmse(y_train.loc[val_mask], pred)
                if score < best_rmse:
                    best_rmse = score
                    best_alpha = alpha
        self.alpha_ = best_alpha
        self.state_ = self._fit_state(X_train, y_train, best_alpha)
        return None

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.state_ is None:
            raise RuntimeError("RidgeFEForecaster must be fitted before predict().")
        return self._predict_with_state(X_test, self.state_)

    def _fit_state(self, X: pd.DataFrame, y: pd.Series, alpha: float) -> dict:
        regions = _region_ids(X)
        X_num, preprocessor = _fit_numeric_preprocessor(
            X,
            missing_drop_threshold=float(self.params.get("missing_drop_threshold", DEFAULT_MISSING_DROP_THRESHOLD)),
        )
        y = y.astype(float)
        region_array = regions.to_numpy()
        x_means = X_num.groupby(region_array).mean()
        y_means = y.groupby(region_array).mean()
        X_centered = X_num - X_num.groupby(region_array).transform("mean")
        y_centered = y - y.groupby(region_array).transform("mean")

        model = Ridge(alpha=alpha, random_state=int(self.params["random_state"]))
        model.fit(X_centered, y_centered)
        return {
            "model": model,
            "preprocessor": preprocessor,
            "feature_columns": X_num.columns.tolist(),
            "x_means": x_means,
            "y_means": y_means,
            "global_x_mean": X_num.mean(),
            "global_y_mean": float(y.mean()),
        }

    def _predict_with_state(self, X: pd.DataFrame, state: dict) -> np.ndarray:
        regions = _region_ids(X)
        X_num = _transform_numeric_preprocessor(X, state["preprocessor"])
        x_means = state["x_means"].reindex(regions.to_numpy())
        x_means.index = X_num.index
        x_means = x_means.fillna(state["global_x_mean"])
        y_means = state["y_means"].reindex(regions.to_numpy()).to_numpy(dtype=float)
        y_means = np.where(np.isnan(y_means), state["global_y_mean"], y_means)
        pred_centered = state["model"].predict(X_num - x_means)
        return y_means + pred_centered


class LGBMForecaster(BaseForecaster):
    def __init__(self, **params):
        self.missing_drop_threshold = float(
            params.pop("missing_drop_threshold", DEFAULT_MISSING_DROP_THRESHOLD)
        )
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 31,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
        }
        defaults.update(params)
        super().__init__(**defaults)
        self.model_ = None
        self.feature_columns_: list[str] = []
        self.preprocessor_: dict | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        from lightgbm import LGBMRegressor

        X_fit, self.preprocessor_ = _fit_numeric_preprocessor(
            X_train,
            missing_drop_threshold=self.missing_drop_threshold,
        )
        self.feature_columns_ = X_fit.columns.tolist()
        self.model_ = LGBMRegressor(**self.params)
        self.model_.fit(X_fit, y_train.astype(float))
        return None

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("LGBMForecaster must be fitted before predict().")
        if self.preprocessor_ is None:
            raise RuntimeError("LGBMForecaster preprocessor is not fitted.")
        X_pred = _transform_numeric_preprocessor(X_test, self.preprocessor_)
        return np.asarray(self.model_.predict(X_pred), dtype=float)

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 20) -> None:
        import optuna
        from lightgbm import LGBMRegressor

        train_mask, val_mask = _last_year_validation_split(X_train, inner_val_years=2)
        if not val_mask.any():
            self.fit(X_train, y_train)
            return None

        X_inner, preprocessor = _fit_numeric_preprocessor(
            X_train.loc[train_mask],
            missing_drop_threshold=self.missing_drop_threshold,
        )
        X_val = _transform_numeric_preprocessor(X_train.loc[val_mask], preprocessor)
        y_inner = y_train.loc[train_mask].astype(float)
        y_val = y_train.loc[val_mask].astype(float)

        # Число деревьев внутри пробы намеренно сокращено (100 вместо 300):
        # это ускоряет каждый trial в ~3x, практически не влияя на ранжирование
        # конфигураций. После поиска финальный fit выполняется с полным n_estimators.
        _TRIAL_N_ESTIMATORS = 100

        def objective(trial: optuna.Trial) -> float:
            params = dict(self.params)
            params.update(
                {
                    "n_estimators": _TRIAL_N_ESTIMATORS,
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                    "n_jobs": -1,
                    "verbose": -1,
                }
            )
            model = LGBMRegressor(**params)
            model.fit(X_inner, y_inner)
            pred = model.predict(X_val)
            return _rmse(y_val, pred)

        sampler = optuna.samplers.TPESampler(seed=int(self.params.get("random_state", 42)))
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        # Применяем лучшие параметры (без n_estimators: финальный fit берёт его из self.params)
        best = {k: v for k, v in study.best_params.items() if k != "n_estimators"}
        self.params.update(best)
        self.params.setdefault("n_jobs", -1)   # параллельное обучение деревьев
        self.fit(X_train, y_train)
        return None


class CatBoostForecaster(BaseForecaster):
    def __init__(self, **params):
        self.missing_drop_threshold = float(
            params.pop("missing_drop_threshold", DEFAULT_MISSING_DROP_THRESHOLD)
        )
        defaults = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3,
            "random_seed": 42,
            "verbose": 0,
            "allow_writing_files": False,
        }
        defaults.update(params)
        super().__init__(**defaults)
        self.model_ = None
        self.feature_columns_: list[str] = []
        self.cat_features_: list[int] = []
        self.preprocessor_: dict | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        from catboost import CatBoostRegressor

        X_fit, self.preprocessor_ = _fit_numeric_preprocessor(
            X_train,
            missing_drop_threshold=self.missing_drop_threshold,
        )
        self.feature_columns_ = X_fit.columns.tolist()
        self.cat_features_ = []
        self.model_ = CatBoostRegressor(**self.params)
        self.model_.fit(X_fit, y_train.astype(float), cat_features=self.cat_features_)
        return None

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("CatBoostForecaster must be fitted before predict().")
        if self.preprocessor_ is None:
            raise RuntimeError("CatBoostForecaster preprocessor is not fitted.")
        X_pred = _transform_numeric_preprocessor(X_test, self.preprocessor_)
        return np.asarray(self.model_.predict(X_pred), dtype=float)

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 20) -> None:
        import optuna
        from catboost import CatBoostRegressor

        train_mask, val_mask = _last_year_validation_split(X_train, inner_val_years=2)
        if not val_mask.any():
            self.fit(X_train, y_train)
            return None

        X_inner, preprocessor = _fit_numeric_preprocessor(
            X_train.loc[train_mask],
            missing_drop_threshold=self.missing_drop_threshold,
        )
        X_val = _transform_numeric_preprocessor(X_train.loc[val_mask], preprocessor)
        cat_features: list[int] = []
        y_inner = y_train.loc[train_mask].astype(float)
        y_val = y_train.loc[val_mask].astype(float)

        # Урезаем число итераций внутри проб (~3x быстрее).
        # Финальный fit после поиска использует полное self.params["iterations"].
        _TRIAL_ITERATIONS = 150

        def objective(trial: optuna.Trial) -> float:
            params = dict(self.params)
            params.update(
                {
                    "iterations": _TRIAL_ITERATIONS,
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20, log=True),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
                    "random_strength": trial.suggest_float("random_strength", 0, 10),
                }
            )
            model = CatBoostRegressor(**params)
            model.fit(X_inner, y_inner, cat_features=cat_features)
            pred = model.predict(X_val)
            return _rmse(y_val, pred)

        sampler = optuna.samplers.TPESampler(seed=int(self.params.get("random_seed", 42)))
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        # Применяем без iterations: финальный fit берёт полное число из self.params
        best = {k: v for k, v in study.best_params.items() if k != "iterations"}
        self.params.update(best)
        self.fit(X_train, y_train)
        return None


class GPBoostForecaster(BaseForecaster):
    def __init__(self, group_col: str = "region_id", **params):
        self.missing_drop_threshold = float(
            params.pop("missing_drop_threshold", DEFAULT_MISSING_DROP_THRESHOLD)
        )
        defaults = {
            "num_boost_round": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 31,
            "verbose_eval": False,
            "random_state": 42,
        }
        defaults.update(params)
        super().__init__(group_col=group_col, **defaults)
        self.group_col = group_col
        self.booster_ = None
        self.gp_model_ = None
        self.feature_columns_: list[str] = []
        self.preprocessor_: dict | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        import gpboost as gpb

        if self.group_col not in X_train.columns:
            raise ValueError(f"GPBoostForecaster requires {self.group_col} in X_train.")
        group_data = X_train[self.group_col].astype(str).to_numpy()
        X_fit, self.preprocessor_ = _fit_numeric_preprocessor(
            X_train,
            missing_drop_threshold=self.missing_drop_threshold,
        )
        self.feature_columns_ = X_fit.columns.tolist()
        self.gp_model_ = gpb.GPModel(group_data=group_data, likelihood="gaussian")
        train_set = gpb.Dataset(X_fit, label=y_train.astype(float).to_numpy())
        params = {
            "objective": "regression_l2",
            "metric": "rmse",
            "learning_rate": self.params["learning_rate"],
            "max_depth": self.params["max_depth"],
            "num_leaves": self.params["num_leaves"],
            "verbose": -1,
            "seed": int(self.params.get("random_state", 42)),
        }
        self.booster_ = gpb.train(
            params=params,
            train_set=train_set,
            gp_model=self.gp_model_,
            num_boost_round=int(self.params["num_boost_round"]),
            verbose_eval=bool(self.params.get("verbose_eval", False)),
        )
        return None

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.booster_ is None or self.gp_model_ is None:
            raise RuntimeError("GPBoostForecaster must be fitted before predict().")
        if self.preprocessor_ is None:
            raise RuntimeError("GPBoostForecaster preprocessor is not fitted.")
        X_pred = _transform_numeric_preprocessor(X_test, self.preprocessor_)
        # Variant 4: use fixed-effects only (ignore_gp_model=True) for stable OOS prediction.
        # GP random effects in new time periods cause large extrapolation errors in level space.
        try:
            pred = self.booster_.predict(data=X_pred, ignore_gp_model=True)
        except Exception:
            try:
                group_data_pred = X_test[self.group_col].astype(str).to_numpy()
                pred = self.booster_.predict(
                    data=X_pred,
                    group_data_pred=group_data_pred,
                    predict_var=False,
                )
            except Exception:
                pred = self.booster_.predict(X_pred)
        return self._extract_pred_mean(pred)

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 20) -> None:
        import optuna

        train_mask, val_mask = _last_year_validation_split(X_train, inner_val_years=2)
        if not val_mask.any():
            self.fit(X_train, y_train)
            return None

        # Урезаем num_boost_round внутри проб: быстрая оценка структуры дерева.
        # Финальный fit после поиска использует полное self.params["num_boost_round"].
        _TRIAL_ROUNDS = 100

        def objective(trial: optuna.Trial) -> float:
            trial_model = GPBoostForecaster(
                group_col=self.group_col,
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 7),
                num_leaves=trial.suggest_int("num_leaves", 15, 63),
                num_boost_round=_TRIAL_ROUNDS,   # фиксировано для скорости
                missing_drop_threshold=self.missing_drop_threshold,
                random_state=int(self.params.get("random_state", 42)),
                verbose_eval=False,
            )
            trial_model.fit(X_train.loc[train_mask], y_train.loc[train_mask])
            pred = trial_model.predict(X_train.loc[val_mask])
            return _rmse(y_train.loc[val_mask], pred)

        sampler = optuna.samplers.TPESampler(seed=int(self.params.get("random_state", 42)))
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        # Финальный fit с полным num_boost_round из self.params (не из проб)
        best = {k: v for k, v in study.best_params.items() if k != "num_boost_round"}
        self.params.update(best)
        self.fit(X_train, y_train)
        return None

    @staticmethod
    def _extract_pred_mean(pred) -> np.ndarray:
        if isinstance(pred, dict):
            for key in ["response_mean", "pred_mean", "mean", "mu"]:
                if key in pred:
                    return np.asarray(pred[key], dtype=float)
            if "fixed_effect" in pred and "random_effect_mean" in pred:
                return np.asarray(pred["fixed_effect"], dtype=float) + np.asarray(
                    pred["random_effect_mean"], dtype=float
                )
            first_key = next(iter(pred))
            return np.asarray(pred[first_key], dtype=float)
        return np.asarray(pred, dtype=float)


class EmbMLPForecaster(BaseForecaster):
    """
    Entity Embedding MLP.

    Categorical inputs go through learned embeddings:
        region_id        → Embedding(n_regions,   8)
        region_type      → Embedding(n_types,      2)
        federal_district → Embedding(n_districts,  2)

    Remaining scaled numeric features are concatenated with the embeddings and
    passed to a fully-connected network:
        concat → Linear(→h1) → BN → ReLU → Dropout
               → Linear(→h2) → BN → ReLU → Dropout
               → Linear(→1)

    Serves as a neural fixed-effect model, comparable to RidgeFE / GPBoost.
    """

    _EMBED_DIMS: dict[str, int] = {
        "region_id": 8,
        "region_type": 2,
        "federal_district": 2,
    }

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.3,
        lr: float = 1e-3,
        max_epochs: int = 200,
        batch_size: int = 128,
        patience: int = 20,
        weight_decay: float = 1e-4,
        num_threads: int = 1,
        random_state: int = 42,
        missing_drop_threshold: float = DEFAULT_MISSING_DROP_THRESHOLD,
    ):
        hs = list(hidden_sizes or [128, 64])
        super().__init__(
            hidden_sizes=hs,
            dropout=dropout,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            weight_decay=weight_decay,
            num_threads=num_threads,
            random_state=random_state,
        )
        self.missing_drop_threshold = missing_drop_threshold
        self.vocab_: dict[str, dict[str, int]] = {}
        self.preprocessor_: dict | None = None
        self.numeric_cols_: list[str] = []
        self._oh_drop_cols_: list[str] = []
        self.net_: object | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError as exc:
            raise ImportError("PyTorch is required. Install with: pip install torch") from exc

        seed = int(self.params.get("random_state", 42))
        torch.set_num_threads(int(self.params.get("num_threads", 1)))
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Detect one-hot columns to replace with embeddings
        self._oh_drop_cols_ = [
            c for col in ("region_type", "federal_district")
            for c in X_train.columns if c.startswith(col + "_")
        ]

        # Build vocabulary for each embedding column from train data
        self.vocab_ = {}
        for col in self._EMBED_DIMS:
            vals = self._get_cat_vals(X_train, col)
            unique = sorted({str(v) for v in vals if pd.notna(v) and str(v) != "unknown"})
            self.vocab_[col] = {"<UNK>": 0, **{v: i + 1 for i, v in enumerate(unique)}}

        # Numeric preprocessing — exclude embedding columns
        X_num_raw = X_train.drop(columns=self._oh_drop_cols_ + ["region_id"], errors="ignore")
        X_num_fit, self.preprocessor_ = _fit_numeric_preprocessor(
            X_num_raw, missing_drop_threshold=self.missing_drop_threshold
        )
        self.numeric_cols_ = X_num_fit.columns.tolist()

        X_cat_np = self._encode_cats(X_train)

        # Inner temporal split for early stopping (last year of train → val)
        train_inner, val_inner = _last_year_validation_split(X_train, inner_val_years=1)

        vocab_sizes = {col: len(v) for col, v in self.vocab_.items()}
        net = self._make_net(
            vocab_sizes=vocab_sizes,
            embed_dims=dict(self._EMBED_DIMS),
            n_numeric=len(self.numeric_cols_),
            hidden_sizes=[int(h) for h in self.params["hidden_sizes"]],
            dropout=float(self.params["dropout"]),
        )

        optimizer = optim.Adam(
            net.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        loss_fn = nn.MSELoss()

        X_num_np = X_num_fit.to_numpy(dtype=float)
        y_np = y_train.astype(float).to_numpy()
        t_idx = np.where(train_inner.to_numpy())[0]
        v_idx = np.where(val_inner.to_numpy())[0]
        if len(v_idx) < 5:
            t_idx, v_idx = np.arange(len(y_np)), np.array([], dtype=int)

        def _tf(arr: np.ndarray) -> "torch.Tensor":
            return torch.tensor(arr, dtype=torch.float32)

        def _tl(arr: np.ndarray) -> "torch.Tensor":
            return torch.tensor(arr, dtype=torch.long)

        x_t, y_t = _tf(X_num_np[t_idx]), _tf(y_np[t_idx])
        c_t = {col: _tl(X_cat_np[col][t_idx]) for col in self._EMBED_DIMS}
        x_v = _tf(X_num_np[v_idx]) if len(v_idx) else None
        y_v = _tf(y_np[v_idx]) if len(v_idx) else None
        c_v = {col: _tl(X_cat_np[col][v_idx]) for col in self._EMBED_DIMS} if len(v_idx) else None

        bs = int(self.params["batch_size"])
        patience_val = int(self.params["patience"])
        max_epochs = int(self.params["max_epochs"])
        best_val, best_state, no_improve = np.inf, None, 0

        for _ in range(max_epochs):
            net.train()
            perm = torch.randperm(len(x_t))
            for start in range(0, len(perm), bs):
                idx = perm[start : start + bs]
                optimizer.zero_grad()
                loss = loss_fn(
                    net(x_t[idx], {col: c_t[col][idx] for col in self._EMBED_DIMS}),
                    y_t[idx],
                )
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            if x_v is not None:
                net.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(net(x_v, c_v), y_v).item())
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience_val:
                        break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.net_ = net

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.net_ is None or self.preprocessor_ is None:
            raise RuntimeError("EmbMLPForecaster must be fitted before predict().")
        import torch

        X_num_raw = X_test.drop(columns=self._oh_drop_cols_ + ["region_id"], errors="ignore")
        X_num = _transform_numeric_preprocessor(X_num_raw, self.preprocessor_)
        X_cat_np = self._encode_cats(X_test)

        x_num = torch.tensor(X_num.to_numpy(dtype=float), dtype=torch.float32)
        x_cat = {col: torch.tensor(X_cat_np[col], dtype=torch.long) for col in self._EMBED_DIMS}

        self.net_.eval()
        with torch.no_grad():
            pred = self.net_(x_num, x_cat)
        return pred.numpy().flatten()

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 20) -> None:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        train_mask, val_mask = _last_year_validation_split(X_train, inner_val_years=2)
        if not val_mask.any():
            self.fit(X_train, y_train)
            return None

        def objective(trial: optuna.Trial) -> float:
            model = EmbMLPForecaster(
                hidden_sizes=[
                    trial.suggest_int("h1", 64, 256),
                    trial.suggest_int("h2", 32, 128),
                ],
                dropout=trial.suggest_float("dropout", 0.1, 0.5),
                lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
                max_epochs=int(self.params["max_epochs"]),
                batch_size=int(self.params["batch_size"]),
                patience=int(self.params["patience"]),
                random_state=int(self.params.get("random_state", 42)),
                missing_drop_threshold=self.missing_drop_threshold,
            )
            model.fit(X_train.loc[train_mask], y_train.loc[train_mask])
            return _rmse(y_train.loc[val_mask], model.predict(X_train.loc[val_mask]))

        sampler = optuna.samplers.TPESampler(seed=int(self.params.get("random_state", 42)))
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        bp = study.best_params
        self.params.update({
            "hidden_sizes": [bp["h1"], bp["h2"]],
            "dropout": bp["dropout"],
            "lr": bp["lr"],
            "weight_decay": bp["weight_decay"],
        })
        self.fit(X_train, y_train)

    def _get_cat_vals(self, X: pd.DataFrame, col: str) -> pd.Series:
        """Return the categorical values for a given embedding column."""
        if col == "region_id":
            return _region_ids(X)
        prefix = col + "_"
        oh_cols = [c for c in X.columns if c.startswith(prefix)]
        if not oh_cols:
            return pd.Series(["unknown"] * len(X), index=X.index)
        # Reconstruct category from one-hot: argmax → strip prefix
        vals = X[oh_cols].idxmax(axis=1).str[len(prefix):]
        vals = vals.copy()
        vals[X[oh_cols].max(axis=1) == 0] = "unknown"
        return vals

    def _encode_cats(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Encode all embedding columns as integer indices using fitted vocabularies."""
        result: dict[str, np.ndarray] = {}
        for col in self._EMBED_DIMS:
            vals = self._get_cat_vals(X, col)
            vocab = self.vocab_.get(col, {})
            result[col] = np.array(
                [vocab.get(str(v) if pd.notna(v) else "<UNK>", 0) for v in vals],
                dtype=np.int64,
            )
        return result

    @staticmethod
    def _make_net(
        vocab_sizes: dict[str, int],
        embed_dims: dict[str, int],
        n_numeric: int,
        hidden_sizes: list[int],
        dropout: float,
    ) -> object:
        """Build and return the PyTorch network (lazy torch import)."""
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.ModuleDict({
                    col: nn.Embedding(vocab_sizes[col], embed_dims[col], padding_idx=0)
                    for col in vocab_sizes
                })
                in_dim = n_numeric + sum(embed_dims.values())
                layers: list[nn.Module] = []
                for h in hidden_sizes:
                    layers += [
                        nn.Linear(in_dim, h),
                        nn.BatchNorm1d(h),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                    in_dim = h
                layers.append(nn.Linear(in_dim, 1))
                self.mlp = nn.Sequential(*layers)

            def forward(self, x_num: "torch.Tensor", x_cats: dict) -> "torch.Tensor":
                parts = [x_num] + [self.embeddings[col](x_cats[col]) for col in x_cats]
                return self.mlp(torch.cat(parts, dim=1)).squeeze(1)

        return _Net()


def get_all_models(config: dict, predict_growth: bool = False) -> list[BaseForecaster]:
    models_config = config.get("models", config)
    missing_drop_threshold = float(
        models_config.get("missing_drop_threshold", DEFAULT_MISSING_DROP_THRESHOLD)
    )
    ridge_cfg = dict(models_config.get("ridge", {}))
    lgbm_cfg = dict(models_config.get("lgbm", {}))
    cat_cfg = dict(models_config.get("catboost", {}))
    gp_cfg = dict(models_config.get("gpboost", {}))
    emb_cfg = dict(models_config.get("emb_mlp", {}))
    for cfg in [ridge_cfg, lgbm_cfg, cat_cfg, gp_cfg, emb_cfg]:
        cfg.setdefault("missing_drop_threshold", missing_drop_threshold)
    models: list[BaseForecaster] = [
        NaiveForecaster(predict_growth=predict_growth),
        AR1Forecaster(predict_growth=predict_growth),
        RidgeFEForecaster(**ridge_cfg),
        LGBMForecaster(**lgbm_cfg),
        CatBoostForecaster(**cat_cfg),
        GPBoostForecaster(**gp_cfg),
    ]
    emb_enabled = bool(emb_cfg.pop("enabled", False))
    if emb_enabled:
        models.append(EmbMLPForecaster(**emb_cfg))
    return models
