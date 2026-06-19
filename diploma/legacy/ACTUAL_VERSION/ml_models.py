import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass
import os
import copy
import random

import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier as SklearnStackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier as CatBoostClassifier_clf
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from scipy.stats import randint, uniform, loguniform
import statsmodels.api as sm

import optuna
from optuna.samplers import TPESampler
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb

from sklearn.exceptions import ConvergenceWarning
import warnings

# T0.1: НЕ глушим предупреждения сплошным filterwarnings('ignore') — раньше это
# скрывало диагностику, важную для финансового пайплайна (деление на ноль в
# метриках, ConvergenceWarning от линейных моделей, потерю точности). Оставляем
# только точечные фильтры по конкретным категориям/сообщениям, которые заведомо
# безвредны и лишь шумят в логах.
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings(
    action="ignore",
    message="X has feature names, but KNeighborsRegressor was fitted without feature names",
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Utility Functions ====================

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    `Symmetric Mean Absolute Percentage Error` -
    метрика качества, симметричное отношение к пере- и недопрогнозу

    В отличие от классического MAPE, знаменатель берётся как средняя величина
    |y_true| и |y_pred|, а не только |y_true|: это убирает асимметричный взрыв
    ошибки при y_true -> 0 (когда модель недопрогнозирует около нулевой цены/
    доходности) и ограничивает вклад каждого наблюдения сверху значением 2.
    При den == 0 (оба значения равны нулю) вклад наблюдения принимается за 0,
    а не NaN/inf.

    `y_true`: тагрет на тесте
    `y_pred`: предсказания на тесте
    """
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2

    return float(np.mean(np.where(den != 0, num / den, 0)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """
    `Mean Absolute Scaled Error` -
    метрика качества, сравнивает результат предсказания с наивным предсказанием.

    в качетсве наивного предсказания - `y[t] = y[t-1]` (на трейн выборке)

    * `MASE` > 1 -> результат хуже наивного предсказания
    * `MASE` < 1 -> результат лучше наивного предсказания

    `y_true`: тагрет на тесте
    `y_pred`: предсказания на тесте
    `y_train`: таргет на трейне
    """
    scale = np.mean(np.abs(np.diff(y_train)))
    # На вырожденном трейне (константный ряд, scale == 0) знаменатель не определён;
    # возвращаем NaN, чтобы не протаскивать inf в Optuna objective/агрегаты.
    if scale == 0 or np.isnan(scale):
        return float('nan')
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    `Directional Accuracy` - доля наблюдений, в которых знак прогнозируемого
    приращения (направление: вверх/вниз) совпал с фактическим.

    Предназначена для таргета-доходности: предсказание считается верным, если
    `(y_pred > 0) == (y_true > 0)`. В отличие от MAE/RMSE/SMAPE, эта метрика не
    вырождается на почти-случайном блуждании цены, где уровень ошибки у всех
    моделей близок к персистентности: она напрямую отвечает на вопрос, есть ли
    у модели направленный навык сверх «вчера = сегодня». DA около 0.5 означает
    отсутствие предсказательной силы.

    `y_true`: фактическая доходность на тесте
    `y_pred`: прогноз доходности на тесте
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean((y_true[mask] > 0) == (y_pred[mask] > 0)))


def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    подсчет метрик на трейн- и тест-выборках

    метрики:
    * `MAE`   - mean absolute error
    * `RMSE`  - root mean squared error
    * `SMAPE` - symmetric mean absolute percentage error (осмысленна в пространстве
                цены; на доходностях знаменатель -> 0 при смене знака, поэтому при
                таргете-доходности её следует читать только на восстановленной цене)
    * `MASE`  - mean absolute scaled error
    * `DA`    - directional accuracy (доля совпадений знака приращения)
    """
    mae_val = mean_absolute_error(y_true, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    smape_val = smape(y_true, y_pred)
    da_val = directional_accuracy(y_true, y_pred)

    if y_train is not None:
        mase_val = mase(y_true, y_pred, y_train)
    else:
        mase_val = None

    return {'mae': mae_val, 'rmse': rmse_val, 'smape': smape_val,
            'mase': mase_val, 'da': da_val}



def run_optuna_study(
        objective,
        direction: str,
        n_trials: int,
        seed: int,
        wandb_project: Optional[str] = None,
        run_name: Optional[str] = None,
        study_name: Optional[str] = None,
) -> optuna.Study:
    """
    Запускает поиск гиперпараметров `Optuna` (`TPESampler`).

    `objective`: функция `trial -> score` (минимизируется/максимизируется согласно `direction`)
    `direction`: `'minimize'` или `'maximize'`
    `n_trials`: число испытаний
    `seed`: random seed для сэмплера
    `wandb_project`: если задан - каждый trial логируется в Weights & Biases
    `run_name`: имя запуска в W&B
    `study_name`: имя исследования Optuna
    """
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler, study_name=study_name)

    callbacks = []
    if wandb_project:
        callbacks.append(WeightsAndBiasesCallback(
            metric_name='score',
            wandb_kwargs={'project': wandb_project, 'name': run_name, 'reinit': True},
            as_multirun=False,
        ))

    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)

    if wandb_project:
        wandb.finish()

    return study


@dataclass
class ForecastResult:
    """
    Класс-контейнер для сбора результатов моделей по итогам обучения
    """
    metrics: Dict[str, Dict[str, float]]
    y_pred: pd.Series
    model: Any

class ForecastBase:
    """
    Класс для логирования
    """
    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n\033[94m[{timestamp}]\033[0m {message}")


class LinearForecaster(ForecastBase, BaseEstimator):
    """
    реализация линейных моделей с регуляризацей

    `test_size`: размер тест. выборки. кол-во наблюдений с конца
    `cv_splits`: кол-во фолдов в кросс-валидации
    `model_type`: доступны три варианта - `Ridge`, `Lasso`, `ElasticNet`
    `n_trials`: число испытаний `Optuna`
    `seed`: random seed для `Optuna`
    `wandb_project`: если задан - логирует тюнинг в Weights & Biases

    * разбивка на трейн и тест происходит внутри самого метода
    * подбор параметров `Optuna` (`TPESampler`)
    * метрика качества - `SMAPE`
    """
    def __init__(
        self,
        test_size: int = 52,
        cv_splits: int = 5,
        model_type: str = 'Ridge',
        n_trials: int = 50,
        seed: int = 42,
        wandb_project: Optional[str] = None,
    ):
        self.test_size = test_size
        self.cv_splits = cv_splits
        self.model_type = model_type
        self.n_trials = n_trials
        self.seed = seed
        self.wandb_project = wandb_project
        self.model_: Optional[BaseEstimator] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LinearForecaster':
        """
        обучение линейных моделей с регуляризацией

        `X`: параметры модели
        `y`: таргет
        """
        allowed = {'Ridge', 'Lasso', 'ElasticNet'}
        if self.model_type not in allowed:
            raise ValueError(f"Выбран недопустимый тип модели. \nДоступные варианты: {allowed}")

        self.log(f"{self.model_type} начала обучаться")

        X_train, X_test = X[:-self.test_size], X[-self.test_size:]
        y_train, y_test = y[:-self.test_size], y[-self.test_size:]

        numeric = X_train.columns.tolist()
        model_classes = {'Ridge': Ridge, 'Lasso': Lasso, 'ElasticNet': ElasticNet}

        def build_model(params: Dict[str, Any]) -> TransformedTargetRegressor:
            pre = ColumnTransformer([
                ('num', Pipeline([
                    ('imputer', SimpleImputer()),
                    ('scaler', RobustScaler())
                ]), numeric)
            ])
            base = model_classes[self.model_type](**params)
            feature_pipe = Pipeline([('pre', pre), ('reg', base)])
            return TransformedTargetRegressor(regressor=feature_pipe, transformer=RobustScaler())

        # TimeSeriesSplit разбивает только X_train/y_train (тестовый хвост уже
        # отрезан выше) и не перемешивает порядок наблюдений, поэтому каждый
        # CV-фолд тестируется на наблюдениях, строго следующих за его трейном.
        # Импьютер/скейлер/трансформер таргета объявлены внутри Pipeline и
        # TransformedTargetRegressor -> cross_val_score переобучает их на
        # трейне каждого фолда отдельно (no look-ahead в препроцессинге).
        cv = TimeSeriesSplit(n_splits=self.cv_splits)
        scorer = make_scorer(smape, greater_is_better=False)

        def objective(trial: optuna.Trial) -> float:
            params = {
                'alpha': trial.suggest_float('alpha', 1e-4, 1e4, log=True),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'tol': trial.suggest_float('tol', 1e-5, 1e-1, log=True),
            }
            if self.model_type == 'Ridge':
                params['solver'] = trial.suggest_categorical(
                    'solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
                )
            else:
                params['max_iter'] = trial.suggest_int('max_iter', 1000, 10000, step=1000)
                if self.model_type == 'ElasticNet':
                    params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

            model = build_model(params)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer, n_jobs=-1)
            return -scores.mean()

        study = run_optuna_study(
            objective,
            direction='minimize',
            n_trials=self.n_trials,
            seed=self.seed,
            wandb_project=self.wandb_project,
            run_name=f'LinearForecaster_{self.model_type}',
            study_name=f'linear_{self.model_type}',
        )
        self.study_ = study

        model = build_model(study.best_params)
        model.fit(X_train, y_train)
        self.model_ = model
        self.log(f"{self.model_type} закончила обучаться")

        self.y_pred_train = model.predict(X_train)
        self.y_pred_test = model.predict(X_test)

        metrics = {
            'train': compute_metrics(y_train=y_train.values,
                                     y_true=y_train.values,
                                     y_pred=self.y_pred_train),
            'test': compute_metrics(y_train=y_train.values,
                                    y_true=y_test.values,
                                    y_pred=self.y_pred_test)
        }
        self.result_ = ForecastResult(
            metrics,
            pd.Series(self.y_pred_test, index=X_test.index),
            model
        )
        return self

    def predict(self) -> ForecastResult:
        """
        предсказания на трейне и на тесте 'Dict[train_pred: pd.Series, test_pred: pd.Series]'
        """
        preds = {
            'train':    self.y_pred_train,
            'test':     self.y_pred_test
        }

        return preds


class StackingForecaster(ForecastBase, BaseEstimator):
    """
    реализация стекинг-моделей.

    * результирующая модель - `Ridge`
    * вторая базовая модель `KNN`, первая - выбранная Вами из доступных.

    * разбивка на трейн и тест происходит внутри самого метода
    * подбор параметров `RandomizedSearchCV`
    * метрика качества - `SMAPE`

    `model_type`: 'RandomForest', 'CatBoost', или 'XGBoost'
    `test_size`:  число последних точек для теста
    `cv_splits`:  число фолдов в кросс-валидации
    `n_trials`:   число испытаний `Optuna`
    `seed`:       random seed
    `wandb_project`: если задан - логирует тюнинг в Weights & Biases
    """
    def __init__(
        self,
        model_type: str = 'RandomForest',
        test_size: int = 260,
        cv_splits: int = 5,
        n_trials: int = 50,
        seed: int = 42,
        wandb_project: Optional[str] = None,
    ):
        allowed = {'RandomForest', 'CatBoost', 'XGBoost', 'LightGBM'}
        if model_type not in allowed:
            raise ValueError(f"тип модели должен быть одним из следующих: {allowed}")

        self.model_type = model_type
        self.test_size = test_size
        self.cv_splits = cv_splits
        self.n_trials = n_trials
        self.seed = seed
        self.wandb_project = wandb_project

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        обучение стеккинг-моделей.

        `X` - датасет с признаками модели
        `y` - временной ряд таргета
        """

        if len(X) <= self.test_size:
            raise ValueError(f"Недостаточно данных ({len(X)}) для test_size={self.test_size}")

        self.log(f"{self.model_type} начала обучаться")


        X_train, X_test = X.iloc[:-self.test_size], X.iloc[-self.test_size:]
        y_train, y_test = y.iloc[:-self.test_size], y.iloc[-self.test_size:]

        self.X_train , self.X_test = X_train, X_test

        numeric_cols = X.select_dtypes('number').columns.tolist()

        def build_model() -> TransformedTargetRegressor:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('continuous', Pipeline([
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler',  RobustScaler()),
                    ]),
                    numeric_cols)
                ],
                remainder='drop'
            )

            base_estimators = {
                'RandomForest': [
                    ("rf",  RandomForestRegressor(random_state=self.seed)),
                    ("knn", KNeighborsRegressor())
                ],
                'CatBoost': [
                    ("cat", CatBoostRegressor(random_seed=self.seed, verbose=0)),
                    ("knn", KNeighborsRegressor())
                ],
                'XGBoost': [
                    ("xgb", XGBRegressor(random_state=self.seed, verbosity=0, n_jobs=-1)),
                    ("knn", KNeighborsRegressor())
                ],
                'LightGBM': [
                    ('lgbm', LGBMRegressor(random_state=self.seed, n_jobs=-1, verbose=-1)),
                    ("knn", KNeighborsRegressor())
                ]
            }

            stack = StackingRegressor(
                estimators      = base_estimators[self.model_type],
                final_estimator = Ridge(),
                cv              = self.cv_splits,
                passthrough     = True,
                n_jobs          = -1
            )

            stack_pipe = Pipeline([
                ('prep',  preprocessor),
                ('stack', stack)
            ])

            return TransformedTargetRegressor(
                regressor   = stack_pipe,
                transformer = RobustScaler()
            )

        # Как и в LinearForecaster: CV выполняется только на X_train/y_train
        # (тест отрезан выше), TimeSeriesSplit сохраняет временной порядок,
        # препроцессинг (impute/scale) переобучается внутри ColumnTransformer
        # на каждом фолде -> утечки информации из будущего нет.
        cv = TimeSeriesSplit(n_splits=self.cv_splits)
        scorer = make_scorer(smape, greater_is_better=False)
        knn_param_keys = build_model().get_params().keys()

        def suggest_knn(trial: optuna.Trial, prefix: str) -> Dict[str, Any]:
            params = {
                f'{prefix}__knn__n_neighbors': trial.suggest_int(f'{prefix}__knn__n_neighbors', 3, 31),
                f'{prefix}__knn__weights': trial.suggest_categorical(f'{prefix}__knn__weights', ['uniform', 'distance']),
            }
            if f'{prefix}__knn__p' in knn_param_keys:
                params[f'{prefix}__knn__p'] = trial.suggest_categorical(f'{prefix}__knn__p', [1, 2])
            return params

        def objective(trial: optuna.Trial) -> float:
            prefix = 'regressor__stack'
            params: Dict[str, Any] = {}

            if self.model_type == 'RandomForest':
                params.update({
                    f'{prefix}__rf__n_estimators': trial.suggest_int(f'{prefix}__rf__n_estimators', 100, 1500),
                    f'{prefix}__rf__max_depth': trial.suggest_int(f'{prefix}__rf__max_depth', 5, 50),
                    f'{prefix}__rf__min_samples_leaf': trial.suggest_int(f'{prefix}__rf__min_samples_leaf', 1, 20),
                    f'{prefix}__rf__min_samples_split': trial.suggest_int(f'{prefix}__rf__min_samples_split', 2, 20),
                    f'{prefix}__rf__min_impurity_decrease': trial.suggest_float(f'{prefix}__rf__min_impurity_decrease', 1e-5, 1e-2, log=True),
                    f'{prefix}__rf__max_samples': trial.suggest_float(f'{prefix}__rf__max_samples', 0.5, 1.0),
                    f'{prefix}__rf__max_features': trial.suggest_float(f'{prefix}__rf__max_features', 0.3, 1.0),
                    f'{prefix}__rf__bootstrap': True,
                })
                params.update(suggest_knn(trial, prefix))
                params[f'{prefix}__final_estimator__alpha'] = trial.suggest_float(f'{prefix}__final_estimator__alpha', 1e-3, 10, log=True)

            elif self.model_type == 'CatBoost':
                bootstrap_type = trial.suggest_categorical(f'{prefix}__cat__bootstrap_type', ['Bayesian', 'Bernoulli'])
                params.update({
                    f'{prefix}__cat__learning_rate': trial.suggest_float(f'{prefix}__cat__learning_rate', 0.01, 0.3),
                    f'{prefix}__cat__depth': trial.suggest_int(f'{prefix}__cat__depth', 4, 11),
                    f'{prefix}__cat__l2_leaf_reg': trial.suggest_float(f'{prefix}__cat__l2_leaf_reg', 1, 20),
                    f'{prefix}__cat__iterations': trial.suggest_int(f'{prefix}__cat__iterations', 100, 1500),
                    f'{prefix}__cat__border_count': trial.suggest_int(f'{prefix}__cat__border_count', 32, 255),
                    f'{prefix}__cat__random_strength': trial.suggest_float(f'{prefix}__cat__random_strength', 0, 10),
                    f'{prefix}__cat__bootstrap_type': bootstrap_type,
                })
                if bootstrap_type == 'Bayesian':
                    params[f'{prefix}__cat__bagging_temperature'] = trial.suggest_float(f'{prefix}__cat__bagging_temperature', 0, 10)
                else:
                    params[f'{prefix}__cat__subsample'] = trial.suggest_float(f'{prefix}__cat__subsample', 0.5, 1.0)
                params.update(suggest_knn(trial, prefix))
                params[f'{prefix}__final_estimator__alpha'] = trial.suggest_float(f'{prefix}__final_estimator__alpha', 1e-4, 1.0)

            elif self.model_type == 'XGBoost':
                params.update({
                    f'{prefix}__xgb__eta': trial.suggest_float(f'{prefix}__xgb__eta', 0.01, 0.3),
                    f'{prefix}__xgb__max_depth': trial.suggest_int(f'{prefix}__xgb__max_depth', 3, 10),
                    f'{prefix}__xgb__subsample': trial.suggest_float(f'{prefix}__xgb__subsample', 0.5, 1.0),
                    f'{prefix}__xgb__colsample_bytree': trial.suggest_float(f'{prefix}__xgb__colsample_bytree', 0.5, 1.0),
                    f'{prefix}__xgb__lambda': trial.suggest_float(f'{prefix}__xgb__lambda', 1e-3, 10, log=True),
                    f'{prefix}__xgb__alpha': trial.suggest_float(f'{prefix}__xgb__alpha', 1e-3, 10, log=True),
                    f'{prefix}__xgb__n_estimators': trial.suggest_int(f'{prefix}__xgb__n_estimators', 100, 1500),
                    f'{prefix}__xgb__gamma': trial.suggest_float(f'{prefix}__xgb__gamma', 0, 5),
                    f'{prefix}__xgb__min_child_weight': trial.suggest_int(f'{prefix}__xgb__min_child_weight', 1, 10),
                })
                params.update(suggest_knn(trial, prefix))
                params[f'{prefix}__final_estimator__alpha'] = trial.suggest_float(f'{prefix}__final_estimator__alpha', 1e-4, 1.0)

            elif self.model_type == 'LightGBM':
                params.update({
                    f'{prefix}__lgbm__learning_rate': trial.suggest_float(f'{prefix}__lgbm__learning_rate', 0.01, 0.3),
                    f'{prefix}__lgbm__num_leaves': trial.suggest_int(f'{prefix}__lgbm__num_leaves', 20, 149),
                    f'{prefix}__lgbm__min_child_samples': trial.suggest_int(f'{prefix}__lgbm__min_child_samples', 5, 99),
                    f'{prefix}__lgbm__feature_fraction': trial.suggest_float(f'{prefix}__lgbm__feature_fraction', 0.5, 1.0),
                    f'{prefix}__lgbm__bagging_fraction': trial.suggest_float(f'{prefix}__lgbm__bagging_fraction', 0.5, 1.0),
                    f'{prefix}__lgbm__bagging_freq': trial.suggest_int(f'{prefix}__lgbm__bagging_freq', 1, 10),
                    f'{prefix}__lgbm__min_split_gain': trial.suggest_float(f'{prefix}__lgbm__min_split_gain', 0, 1),
                    f'{prefix}__lgbm__reg_alpha': trial.suggest_float(f'{prefix}__lgbm__reg_alpha', 1e-3, 10, log=True),
                    f'{prefix}__lgbm__reg_lambda': trial.suggest_float(f'{prefix}__lgbm__reg_lambda', 1e-3, 10, log=True),
                    f'{prefix}__lgbm__n_estimators': trial.suggest_int(f'{prefix}__lgbm__n_estimators', 100, 1500),
                })
                params.update(suggest_knn(trial, prefix))
                params[f'{prefix}__final_estimator__alpha'] = trial.suggest_float(f'{prefix}__final_estimator__alpha', 1e-4, 1.0)

            model = build_model()
            model.set_params(**params)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer, n_jobs=4)
            return -scores.mean()

        study = run_optuna_study(
            objective,
            direction='minimize',
            n_trials=self.n_trials,
            seed=self.seed,
            wandb_project=self.wandb_project,
            run_name=f'StackingForecaster_{self.model_type}',
            study_name=f'stacking_{self.model_type}',
        )
        self.study_ = study

        model = build_model()
        model.set_params(**study.best_params)
        model.fit(X_train, y_train)
        self.model_ = model

        self.log(f"{self.model_type} закончила обучаться")

        self.y_pred_train = model.predict(X_train)
        self.y_pred_test = model.predict(X_test)

        metrics = {
            'train': compute_metrics(
                        y_train=y_train.values,
                        y_true=y_train.values,
                        y_pred=self.y_pred_train
                    ),
            'test': compute_metrics(
                        y_train=y_train.values,
                        y_true=y_test.values,
                        y_pred=self.y_pred_test
                    )
        }
        self.result_ = ForecastResult(
            metrics,
            pd.Series(self.y_pred_test, index=X_test.index),
            model
        )

        return self


    def predict(self) -> Dict[str, pd.Series]:
        """
        предсказания модели на трейн и тест выборках
        """
        preds = {
            'train':    self.y_pred_train,
            'test':     self.y_pred_test
        }

        return preds



def set_global_seed(seed: int):
    """Фиксируем сиды для воспроизводимости"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # PyTorch's intra-op thread pool conflicts with the OpenMP runtimes
    # bundled by xgboost/lightgbm/catboost (also imported in this module),
    # which can segfault on macOS unless torch is restricted to 1 thread.
    torch.set_num_threads(1)
    random.seed(seed)


def smape_loss_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    `Symmetric Mean Absolute Percentage Error` -
    метрика качества, симметричное отношение к пере- и недопрогнозу

    посчитана с использованием PyTorch

    `eps` добавлен непосредственно в знаменатель (а не как post-hoc `clamp`/`max`
    после деления): это гарантирует, что функция остаётся гладкой и
    дифференцируемой в точке y_true = y_pred = 0, где обычный SMAPE имеет 0/0.
    При |y_true| + |y_pred| >> eps вклад eps пренебрежимо мал и не искажает метрику.

    `y_true`: тагрет на тесте
    `y_pred`: предсказания на тесте
    """
    eps = 1e-8
    denom = (torch.abs(y_true) + torch.abs(y_pred) + eps) / 2.0
    diff = torch.abs(y_true - y_pred)
    return torch.mean(diff / denom)


def _sample_crypto_hparams(trial, arch: str, window_size: int) -> dict:
    """
    Сэмплирует гиперпараметры `CryptoNet` для архитектуры `arch`
    (диапазоны зеркалят бывший `CryptoHyperModel`/`CryptoHyperModelCLF`).
    """
    hp = {}

    if arch == 'MLP':
        hp['mlp_layers'] = trial.suggest_int('mlp_layers', 1, 3)
        for i in range(hp['mlp_layers']):
            hp[f'mlp_units_{i}'] = trial.suggest_int(f'mlp_units_{i}', 32, 192, step=32)
            hp[f'mlp_drop_{i}'] = trial.suggest_float(f'mlp_drop_{i}', 0.1, 0.5, step=0.1)

    elif arch == 'LSTM':
        hp['lstm_units'] = trial.suggest_int('lstm_units', 32, 192, step=32)
        hp['lstm_drop'] = trial.suggest_float('lstm_drop', 0.1, 0.5, step=0.1)

    elif arch == 'StackedLSTM':
        hp['stacked_layers'] = trial.suggest_int('stacked_layers', 2, 3)
        for i in range(hp['stacked_layers']):
            hp[f'stack_units_{i}'] = trial.suggest_int(f'stack_units_{i}', 32, 192, step=32)
            hp[f'stack_drop_{i}'] = trial.suggest_float(f'stack_drop_{i}', 0.1, 0.5, step=0.1)

    elif arch == 'CNN_LSTM':
        max_kernel = max(2, min(5, window_size - 1))
        hp['cnn_kernel'] = trial.suggest_int('cnn_kernel', 2, max_kernel)
        conv_out = window_size - hp['cnn_kernel'] + 1
        max_pool = min(4, conv_out) if conv_out >= 2 else 1
        hp['pool_size'] = trial.suggest_int('pool_size', 1, max_pool)
        hp['cnn_filters'] = trial.suggest_int('cnn_filters', 16, 96, step=16)
        hp['cnn_drop'] = trial.suggest_float('cnn_drop', 0.1, 0.5, step=0.1)
        hp['cnn_lstm_units'] = trial.suggest_int('cnn_lstm_units', 32, 192, step=32)

    else:  # GRU
        hp['gru_units'] = trial.suggest_int('gru_units', 32, 192, step=32)
        hp['gru_drop'] = trial.suggest_float('gru_drop', 0.1, 0.5, step=0.1)

    hp['ex_layers'] = trial.suggest_int('ex_layers', 1, 2)
    for i in range(hp['ex_layers']):
        hp[f'ex_units_{i}'] = trial.suggest_int(f'ex_units_{i}', 16, 96, step=16)
        hp[f'ex_drop_{i}'] = trial.suggest_float(f'ex_drop_{i}', 0.1, 0.5, step=0.1)

    # Оптимизация и регуляризация (общие для всех arch). LR-диапазон сужен к области,
    # где RNN обучаются стабильно; weight_decay добавлен против переобучения на коротких
    # фолдах (особенно hourly/5min, где у сети много параметров на мало данных).
    hp['learning_rate'] = trial.suggest_float('learning_rate', 3e-4, 5e-3, log=True)
    hp['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    return hp


class CryptoNet(nn.Module):
    """
    Двухветочная нейросеть для прогнозирования BTC: ветка последовательности
    цены (`x_seq`, shape `(B, window_size, 1)`) + ветка экзогенных признаков
    (`x_exog`, shape `(B, exog_dim)`). Поддерживает архитектуры MLP, LSTM,
    StackedLSTM, CNN_LSTM, GRU - аналог бывшего `CryptoHyperModel`/
    `CryptoHyperModelCLF` на Keras, но конфигурируется словарём
    гиперпараметров `hp`, сэмплированным через `_sample_crypto_hparams`.

    При `output_activation='sigmoid'` (классификация) `forward` всё равно
    возвращает сырой логит - сигмоида применяется отдельно при инференсе,
    при обучении используется `BCEWithLogitsLoss`.
    """
    def __init__(self, window_size: int, exog_dim: int, arch: str, hp: dict, output_activation: Optional[str] = None):
        super().__init__()
        self.arch = arch
        self.output_activation = output_activation

        if arch == 'MLP':
            layers = []
            in_dim = window_size
            for i in range(hp['mlp_layers']):
                units = hp[f'mlp_units_{i}']
                layers += [nn.Linear(in_dim, units), nn.ReLU(), nn.Dropout(hp[f'mlp_drop_{i}']), nn.BatchNorm1d(units)]
                in_dim = units
            self.seq_branch = nn.Sequential(*layers)
            seq_out_dim = in_dim

        elif arch == 'LSTM':
            units = hp['lstm_units']
            self.seq_rnn = nn.LSTM(1, units, batch_first=True)
            self.seq_dropout = nn.Dropout(hp['lstm_drop'])
            seq_out_dim = units

        elif arch == 'StackedLSTM':
            n_layers = hp['stacked_layers']
            self.seq_rnns = nn.ModuleList()
            self.seq_dropouts = nn.ModuleList()
            in_size = 1
            for i in range(n_layers):
                units = hp[f'stack_units_{i}']
                self.seq_rnns.append(nn.LSTM(in_size, units, batch_first=True))
                self.seq_dropouts.append(nn.Dropout(hp[f'stack_drop_{i}']))
                in_size = units
            seq_out_dim = in_size

        elif arch == 'CNN_LSTM':
            filters = hp['cnn_filters']
            self.conv = nn.Conv1d(1, filters, kernel_size=hp['cnn_kernel'])
            self.pool = nn.MaxPool1d(kernel_size=hp['pool_size'])
            self.conv_dropout = nn.Dropout(hp['cnn_drop'])
            units = hp['cnn_lstm_units']
            self.seq_rnn = nn.LSTM(filters, units, batch_first=True)
            seq_out_dim = units

        else:  # GRU
            units = hp['gru_units']
            self.seq_rnn = nn.GRU(1, units, batch_first=True)
            self.seq_dropout = nn.Dropout(hp['gru_drop'])
            seq_out_dim = units

        ex_layers = []
        in_dim = exog_dim
        for i in range(hp['ex_layers']):
            units = hp[f'ex_units_{i}']
            ex_layers += [nn.Linear(in_dim, units), nn.ReLU(), nn.Dropout(hp[f'ex_drop_{i}'])]
            in_dim = units
        self.exog_branch = nn.Sequential(*ex_layers)
        exog_out_dim = in_dim

        self.head = nn.Linear(seq_out_dim + exog_out_dim, 1)

    def forward(self, x_seq: torch.Tensor, x_exog: torch.Tensor) -> torch.Tensor:
        if self.arch == 'MLP':
            x = self.seq_branch(x_seq.flatten(1))

        elif self.arch == 'LSTM':
            out, _ = self.seq_rnn(x_seq)
            x = self.seq_dropout(out[:, -1, :])

        elif self.arch == 'StackedLSTM':
            x = x_seq
            for rnn, drop in zip(self.seq_rnns, self.seq_dropouts):
                x, _ = rnn(x)
                x = drop(x)
            x = x[:, -1, :]

        elif self.arch == 'CNN_LSTM':
            c = torch.relu(self.conv(x_seq.permute(0, 2, 1)))
            c = self.conv_dropout(self.pool(c))
            out, _ = self.seq_rnn(c.permute(0, 2, 1))
            x = out[:, -1, :]

        else:  # GRU
            out, _ = self.seq_rnn(x_seq)
            x = self.seq_dropout(out[:, -1, :])

        ex = self.exog_branch(x_exog)
        return self.head(torch.cat([x, ex], dim=1))


def _make_loader(X_seq: np.ndarray, X_ex: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool,
                 weights: Optional[np.ndarray] = None) -> DataLoader:
    tensors = [
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(X_ex, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    ]
    if weights is not None:
        # Опциональные веса наблюдений (для взвешенной BCE в CryptoNetClassifier).
        tensors.append(torch.tensor(weights, dtype=torch.float32))
    dataset = TensorDataset(*tensors)
    drop_last = shuffle and len(dataset) > batch_size
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def _train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                  lr: float, epochs: int, patience: int = 5, task: str = 'reg', device: str = DEVICE,
                  weight_decay: float = 0.0, grad_clip: float = 1.0):
    """
    Обучает `CryptoNet`: AdamW (+weight_decay) + `ReduceLROnPlateau` + grad-clipping +
    ручной early stopping.

    `task='reg'`: минимизирует Huber/SmoothL1 loss, early stopping/scheduler по val MAE.
        Huber выбран ВМЕСТО SMAPE: на таргете-доходности (центр ~0, частые значения у
        нуля) знаменатель SMAPE |y|+|ŷ| вырождается, лосс доминируется шумовыми мелкими
        доходностями, и обучение скатывается к предсказанию ~константы у нуля (отсюда
        был DA < 0.5 на hourly). Huber устойчив к тяжёлым хвостам доходностей и корректно
        ведёт себя в нуле; бустинги минимизируют L2 по той же причине.
    `task='clf'`: `BCEWithLogitsLoss`, early stopping/scheduler по `val_auc` (максимизация).

    `weight_decay` - L2-регуляризация в AdamW; `grad_clip` - max-norm обрезка градиента
    (стабилизирует обучение RNN). Возвращает `(model, best_val_metric)` - `model` с весами
    лучшей по валидации эпохи (для reg метрика = val MAE, меньше -> лучше).
    """
    model.to(device)
    loss_fn = nn.SmoothL1Loss() if task == 'reg' else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min' if task == 'reg' else 'max', factor=0.5, patience=3
    )

    best_val = float('inf') if task == 'reg' else -float('inf')
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        for xb_seq, xb_ex, yb in train_loader:
            xb_seq, xb_ex, yb = xb_seq.to(device), xb_ex.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb_seq, xb_ex).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb_seq, xb_ex, yb in val_loader:
                xb_seq, xb_ex, yb = xb_seq.to(device), xb_ex.to(device), yb.to(device)
                val_preds.append(model(xb_seq, xb_ex).squeeze(-1))
                val_true.append(yb)
        val_preds = torch.cat(val_preds)
        val_true = torch.cat(val_true)

        if task == 'reg':
            val_metric = torch.mean(torch.abs(val_true - val_preds)).item()  # val MAE
            improved = val_metric < best_val
        else:
            val_proba = torch.sigmoid(val_preds).cpu().numpy()
            val_metric = roc_auc_score(val_true.cpu().numpy(), val_proba)
            improved = val_metric > best_val

        scheduler.step(val_metric)

        if improved:
            best_val = val_metric
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val


class CryptoNetRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn-совместимая (`fit`/`predict`) обёртка над `CryptoNet` для
    `arch in {'LSTM', 'GRU', 'StackedLSTM', 'CNN_LSTM'}`, дающая боостинг- и
    нейросетевым моделям общий интерфейс для одного walk-forward цикла и
    единого `compute_metrics`.

    `X` - DataFrame с `target_col` (исходная, нескейленная цена) и
    `feature_cols` (экзогенные признаки), упорядоченный по времени без
    пропусков. Окна строятся внутри `fit`/`predict`: `X_seq[i]` - это
    `window_size` последовательных значений `target_col`, `X_ex[i]` -
    экзогенные признаки в момент прогноза, целевая переменная - `target_col`
    в той же точке (то есть `X.index[window_size:]`).

    Каждый вызов `fit` создаёт новую `CryptoNet` и новые `RobustScaler`
    (`scaler_y_`/`scaler_x_`), обученные только на тренировочной части `X`
    (до `val_frac`-хвоста). Это принципиально для walk-forward: ни веса
    модели, ни скейлеры, ни скрытое состояние RNN не переносятся между
    независимыми окнами/фолдами.

    Для `predict` на тестовом фолде в `X` нужно передать `window_size`
    дополнительных строк истории перед первой целевой датой - иначе для неё
    не из чего собрать `X_seq[0]`.
    """

    def __init__(self, arch: str = 'GRU', target_col: str = 'BTC',
                 feature_cols: Optional[list] = None, window_size: int = 10,
                 hp: Optional[dict] = None, lr: float = 1e-3, epochs: int = 20,
                 batch_size: int = 32, patience: int = 5, val_frac: float = 0.2,
                 seed: int = 42, device: str = DEVICE, weight_decay: float = 0.0,
                 optuna_val_frac: float = 0.0):
        self.arch = arch
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.hp = hp
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_frac = val_frac
        # optuna_val_frac > 0 -> fit карвит ДОПОЛНИТЕЛЬНЫЙ held-out срез в самом
        # конце окна (схема train / es-val / optuna-val), и best_val_score_ =
        # метрика на нём, а НЕ на es-val. Это разносит выбор гиперпараметров
        # (Optuna) и early stopping на разные срезы (фикс C3 / T2.4). При 0.0
        # (по умолчанию, для финального production-фита) поведение прежнее: 80/20.
        self.optuna_val_frac = optuna_val_frac
        self.seed = seed
        self.device = device

    def _make_windows(self, X: pd.DataFrame):
        y_raw = X[self.target_col].to_numpy(dtype=float)
        exog_raw = X[self.feature_cols].to_numpy(dtype=float)

        X_seq, X_ex, y_seq = [], [], []
        for i in range(len(y_raw) - self.window_size):
            X_seq.append(y_raw[i:i + self.window_size].reshape(self.window_size, 1))
            X_ex.append(exog_raw[i + self.window_size])
            y_seq.append(y_raw[i + self.window_size])

        index = X.index[self.window_size:]
        return np.array(X_seq), np.array(X_ex), np.array(y_seq), index

    def fit(self, X: pd.DataFrame, y=None) -> 'CryptoNetRegressor':
        if not self.feature_cols:
            raise ValueError('feature_cols must be a non-empty list of column names')
        if self.hp is None:
            raise ValueError('hp must be set (see _sample_crypto_hparams)')

        set_global_seed(self.seed)
        X_seq, X_ex, y_seq, _ = self._make_windows(X)

        n = len(y_seq)
        val_n = max(1, int(self.val_frac * n))            # es-val (early stopping)
        opt_n = int(self.optuna_val_frac * n) if self.optuna_val_frac > 0 else 0
        train_n = n - val_n - opt_n
        if train_n <= 0:
            raise ValueError(f'not enough rows after windowing for fit: n={n}, window_size={self.window_size}')
        es_end = train_n + val_n                          # граница es-val; [es_end:] = optuna-val

        # Скейлеры обучаются только на тренировочной части окна (без val/opt-хвостов),
        # чтобы ни early-stopping, ни Optuna-objective не подсматривали статистику
        # будущих данных.
        self.scaler_y_ = RobustScaler().fit(y_seq[:train_n].reshape(-1, 1))
        self.scaler_x_ = RobustScaler().fit(X_ex[:train_n])

        def scale_seq(arr):
            return self.scaler_y_.transform(arr.reshape(-1, 1)).reshape(arr.shape)

        X_seq_s = scale_seq(X_seq)
        X_ex_s = self.scaler_x_.transform(X_ex)
        y_s = self.scaler_y_.transform(y_seq.reshape(-1, 1)).flatten()

        train_loader = _make_loader(X_seq_s[:train_n], X_ex_s[:train_n], y_s[:train_n], self.batch_size, shuffle=True)
        # early-stopping строго на es-val [train_n:es_end] — БЕЗ optuna-val хвоста.
        val_loader = _make_loader(X_seq_s[train_n:es_end], X_ex_s[train_n:es_end], y_s[train_n:es_end], self.batch_size, shuffle=False)

        self.exog_dim_ = X_ex.shape[1]
        # Свежая CryptoNet на каждый fit: веса/оптимизатор/скрытое состояние RNN
        # не наследуются между walk-forward фолдами.
        model = CryptoNet(self.window_size, self.exog_dim_, self.arch, self.hp)
        self.model_, self.best_es_val_score_ = _train_model(
            model, train_loader, val_loader,
            lr=self.lr, epochs=self.epochs, patience=self.patience, task='reg', device=self.device,
            weight_decay=self.weight_decay,
        )
        # best_val_score_ — метрика для выбора гиперпараметров Optuna. Если выделен
        # отдельный optuna-val срез (opt_n>0) — считаем val MAE на нём (в скейленном
        # пространстве, как в _train_model); иначе fallback на es-val метрику.
        if opt_n > 0:
            self.model_.eval()
            with torch.no_grad():
                pred_opt = self.model_(
                    torch.tensor(X_seq_s[es_end:], dtype=torch.float32).to(self.device),
                    torch.tensor(X_ex_s[es_end:], dtype=torch.float32).to(self.device),
                ).squeeze(-1).cpu().numpy()
            self.best_val_score_ = float(np.mean(np.abs(y_s[es_end:] - pred_opt)))
        else:
            self.best_val_score_ = self.best_es_val_score_
        # Алиас оставлен для обратной совместимости с кодом, читавшим best_val_smape_.
        self.best_val_smape_ = self.best_val_score_
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        check_is_fitted(self, 'model_')
        X_seq, X_ex, _, index = self._make_windows(X)
        if len(index) == 0:
            return pd.Series(dtype=float)

        X_seq_s = self.scaler_y_.transform(X_seq.reshape(-1, 1)).reshape(X_seq.shape)
        X_ex_s = self.scaler_x_.transform(X_ex)

        self.model_.eval()
        with torch.no_grad():
            preds_scaled = self.model_(
                torch.tensor(X_seq_s, dtype=torch.float32).to(self.device),
                torch.tensor(X_ex_s, dtype=torch.float32).to(self.device),
            ).cpu().numpy().flatten()

        preds = self.scaler_y_.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        return pd.Series(preds, index=index, name=f'{self.arch} (sequence NN)')


def focal_mw_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal + magnitude-weighted BCE для классификатора знака доходности (T3.3).

    Объединяет фокальную модуляцию (Lin et al., 2017, *Focal Loss*) с уже
    используемым взвешиванием по величине движения. `weight` — пер-семпл вес `|r|`
    (нормирован к среднему трейна + пол `weight_floor`), `gamma` — фокусирующий
    параметр: множитель ``(1 − p_t)^γ`` даунвейтит «лёгкие» (уверенно верные) бары
    и переносит вес на трудные разворотные/крупные. При ``gamma == 0`` точно
    сводится к прежней взвешенной BCE.

    Все три тензора одномерны и одинаковой длины; `targets` ∈ {0, 1}.
    """
    log_pt = -nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction='none'
    )                                          # = log(p_t), численно стабильно
    focal = (1.0 - log_pt.exp()).pow(gamma)    # (1 − p_t)^γ
    return -(weight * focal * log_pt).mean()


class CryptoNetClassifier(BaseEstimator, ClassifierMixin):
    """
    Классификатор ЗНАКА следующей доходности (вверх/вниз) на базе CryptoNet.

    Зачем отдельно от CryptoNetRegressor: регрессия минимизирует ошибку ВЕЛИЧИНЫ
    (MSE/Huber), а её оптимум на почти непредсказуемом ряду с нулевым средним -
    предсказывать ~0, что даёт Directional Accuracy ≈ 0.5. Здесь цель ставится
    напрямую как бинарная `y = 1[r_t > 0]`, а лосс - FOCAL + magnitude-weighted BCE
    (`focal_mw_bce`, T3.3): вес `|r_t|` (нормирован к среднему по трейну + небольшой
    пол `weight_floor`) делает крупные движения важнее, а фокальный множитель
    ``(1−p_t)^γ`` (`gamma`, по умолчанию 2.0) дополнительно даунвейтит «лёгкие» бары
    и фокусирует обучение на трудных разворотных. При `gamma=0` сводится к прежней
    взвешенной BCE. Так обучение выравнивается с целевой метрикой DA.

    Окна как в регрессоре: `x_seq` - window_size прошлых доходностей, `x_ex` -
    экзогенные признаки на момент прогноза. `predict_proba` -> P(up);
    `decision_return` -> псевдо-доходность `p - 0.5` (знак = направленная ставка),
    что позволяет встроить классификатор в общий OOF-конвейер и экономический бэктест.
    """

    def __init__(self, arch: str = 'GRU', target_col: str = 'BTC_logret',
                 feature_cols: Optional[list] = None, window_size: int = 10,
                 hp: Optional[dict] = None, lr: float = 1e-3, epochs: int = 20,
                 batch_size: int = 32, patience: int = 8, val_frac: float = 0.2,
                 seed: int = 42, device: str = DEVICE, weight_decay: float = 0.0,
                 weight_power: float = 1.0, weight_floor: float = 0.05, grad_clip: float = 1.0,
                 gamma: float = 2.0, optuna_val_frac: float = 0.0):
        self.arch = arch
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.hp = hp
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_frac = val_frac
        # См. CryptoNetRegressor: optuna_val_frac>0 выделяет отдельный held-out
        # срез для Optuna-objective, разнося его с es-val (фикс C3 / T2.4).
        self.optuna_val_frac = optuna_val_frac
        self.seed = seed
        self.device = device
        self.weight_decay = weight_decay
        self.weight_power = weight_power
        self.weight_floor = weight_floor
        self.grad_clip = grad_clip
        self.gamma = gamma   # фокусирующий параметр focal-loss (T3.3)

    def _make_windows(self, X: pd.DataFrame):
        ret = X[self.target_col].to_numpy(dtype=float)
        exog = X[self.feature_cols].to_numpy(dtype=float)
        X_seq, X_ex, y_cls, mag = [], [], [], []
        for i in range(len(ret) - self.window_size):
            X_seq.append(ret[i:i + self.window_size].reshape(self.window_size, 1))
            X_ex.append(exog[i + self.window_size])
            r = ret[i + self.window_size]
            y_cls.append(1.0 if r > 0 else 0.0)
            mag.append(abs(r))
        index = X.index[self.window_size:]
        return np.array(X_seq), np.array(X_ex), np.array(y_cls), np.array(mag), index

    def fit(self, X: pd.DataFrame, y=None) -> 'CryptoNetClassifier':
        if not self.feature_cols:
            raise ValueError('feature_cols must be a non-empty list of column names')
        if self.hp is None:
            raise ValueError('hp must be set (see _sample_crypto_hparams)')

        set_global_seed(self.seed)
        X_seq, X_ex, y_cls, mag, _ = self._make_windows(X)
        n = len(y_cls)
        val_n = max(1, int(self.val_frac * n))            # es-val (early stopping)
        opt_n = int(self.optuna_val_frac * n) if self.optuna_val_frac > 0 else 0
        train_n = n - val_n - opt_n
        if train_n <= 0:
            raise ValueError(f'not enough rows after windowing: n={n}, window_size={self.window_size}')
        es_end = train_n + val_n                          # граница es-val; [es_end:] = optuna-val

        # Скейлеры обучаются только на трейн-части окна (без val/opt-хвостов).
        self.scaler_seq_ = RobustScaler().fit(X_seq[:train_n].reshape(-1, 1))
        self.scaler_x_ = RobustScaler().fit(X_ex[:train_n])
        X_seq_s = self.scaler_seq_.transform(X_seq.reshape(-1, 1)).reshape(X_seq.shape)
        X_ex_s = self.scaler_x_.transform(X_ex)

        # Веса наблюдений = |r|^power, нормированные к среднему по трейну, + пол:
        # крупные движения весомее, но околонулевые наблюдения тоже немного участвуют.
        w = np.power(mag, self.weight_power)
        scale = w[:train_n].mean() or 1.0
        w = w / scale + self.weight_floor

        train_loader = _make_loader(X_seq_s[:train_n], X_ex_s[:train_n], y_cls[:train_n],
                                    self.batch_size, shuffle=True, weights=w[:train_n])
        # early-stopping строго на es-val [train_n:es_end] — без optuna-val хвоста.
        val_loader = _make_loader(X_seq_s[train_n:es_end], X_ex_s[train_n:es_end], y_cls[train_n:es_end],
                                  self.batch_size, shuffle=False, weights=w[train_n:es_end])

        self.exog_dim_ = X_ex.shape[1]
        model = CryptoNet(self.window_size, self.exog_dim_, self.arch, self.hp,
                          output_activation='sigmoid').to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        # Обучение — на focal+magnitude-weighted BCE (T3.3). `bce` ниже остаётся
        # только для МОНИТОРА на val (early stopping / Optuna): стабильная
        # взвешенная BCE как сигнал отбора, не зависящий от gamma.
        bce = nn.BCEWithLogitsLoss(reduction='none')

        best_val = float('inf')
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
        for _ in range(self.epochs):
            model.train()
            for xb_seq, xb_ex, yb, wb in train_loader:
                xb_seq, xb_ex, yb, wb = (xb_seq.to(self.device), xb_ex.to(self.device),
                                         yb.to(self.device), wb.to(self.device))
                optimizer.zero_grad()
                logit = model(xb_seq, xb_ex).squeeze(-1)
                loss = focal_mw_bce(logit, yb, wb, gamma=self.gamma)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()

            model.eval()
            num, den = 0.0, 0.0
            with torch.no_grad():
                for xb_seq, xb_ex, yb, wb in val_loader:
                    xb_seq, xb_ex, yb, wb = (xb_seq.to(self.device), xb_ex.to(self.device),
                                             yb.to(self.device), wb.to(self.device))
                    logit = model(xb_seq, xb_ex).squeeze(-1)
                    num += (bce(logit, yb) * wb).sum().item()
                    den += wb.sum().item()
            val_metric = num / den if den > 0 else float('inf')
            scheduler.step(val_metric)

            if val_metric < best_val:
                best_val = val_metric
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        model.load_state_dict(best_state)
        self.model_ = model
        self.best_es_val_score_ = best_val  # es-val weighted BCE (меньше -> лучше)
        # best_val_score_ для Optuna: на отдельном optuna-val срезе, если выделен
        # (разносит выбор гиперпараметров с early-stopping, T2.4); иначе es-val.
        if opt_n > 0:
            opt_val_loader = _make_loader(X_seq_s[es_end:], X_ex_s[es_end:], y_cls[es_end:],
                                          self.batch_size, shuffle=False, weights=w[es_end:])
            model.eval()
            num, den = 0.0, 0.0
            with torch.no_grad():
                for xb_seq, xb_ex, yb, wb in opt_val_loader:
                    xb_seq, xb_ex, yb, wb = (xb_seq.to(self.device), xb_ex.to(self.device),
                                             yb.to(self.device), wb.to(self.device))
                    logit = model(xb_seq, xb_ex).squeeze(-1)
                    num += (bce(logit, yb) * wb).sum().item()
                    den += wb.sum().item()
            self.best_val_score_ = num / den if den > 0 else float('inf')
        else:
            self.best_val_score_ = best_val
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        check_is_fitted(self, 'model_')
        X_seq, X_ex, _, _, index = self._make_windows(X)
        if len(index) == 0:
            return pd.Series(dtype=float)
        X_seq_s = self.scaler_seq_.transform(X_seq.reshape(-1, 1)).reshape(X_seq.shape)
        X_ex_s = self.scaler_x_.transform(X_ex)
        self.model_.eval()
        with torch.no_grad():
            logit = self.model_(
                torch.tensor(X_seq_s, dtype=torch.float32).to(self.device),
                torch.tensor(X_ex_s, dtype=torch.float32).to(self.device),
            ).squeeze(-1)
            p = torch.sigmoid(logit).cpu().numpy()
        return pd.Series(p, index=index, name=f'{self.arch} (sign clf)')

    def decision_return(self, X: pd.DataFrame) -> pd.Series:
        """Псевдо-доходность `p - 0.5`: знак = направленная ставка (для OOF/бэктеста)."""
        return self.predict_proba(X) - 0.5

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return (self.predict_proba(X) > 0.5).astype(int)


class NeuralForecaster:
    """
    Нейросетевой прогнозировщик цены BTC (PyTorch + Optuna).

    Параллельно обучает 5 архитектур (MLP, LSTM, StackedLSTM, CNN_LSTM, GRU),
    подбор гиперпараметров каждой - через `run_optuna_study` (TPESampler),
    минимизируя `val_smape`.
    """
    def __init__(
        self,
        window_size=10,
        test_size=260,
        max_trials=10,
        epochs=50,
        batch_size=32,
        seed=42,
        wandb_project: Optional[str] = None,
    ):
        self.window_size = window_size
        self.test_size = test_size
        self.max_trials = max_trials
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.wandb_project = wandb_project
        self.architectures = ['MLP', 'LSTM', 'StackedLSTM', 'CNN_LSTM', 'GRU']
        self.models_ = {}
        self.results_ = {}
        set_global_seed(self.seed)


    def fit(self, df: pd.DataFrame, target_col: str):
        """

        """

        y_raw = df[target_col].values
        exog_raw = df.drop(columns=[target_col]).values


        X_seq_raw, X_ex_raw, y_seq_raw = [], [], []

        for i in range(len(y_raw) - self.window_size):
            X_seq_raw.append(y_raw[i: i + self.window_size].reshape(self.window_size, 1))
            X_ex_raw.append(exog_raw[i + self.window_size])
            y_seq_raw.append(y_raw[i + self.window_size])

        X_seq_raw = np.array(X_seq_raw)
        X_ex_raw = np.array(X_ex_raw)
        y_seq_raw = np.array(y_seq_raw)


        N = len(y_seq_raw)
        tn = self.test_size
        rem = N - tn

        if rem <= 0:
            warnings.warn(f"Недостаточно данных для разбиения на трейн и валидацию: total={N}, test={tn}")
            return self

        vn = int(0.2 * rem)
        trn = rem - vn

        if trn <= 0:
            warnings.warn(f"Недостаточно данных для разбиения на трейн и валидацию: remaining={rem}, val={vn}")
            return self

        X_seq_tr_raw = X_seq_raw[:trn]
        X_ex_tr_raw = X_ex_raw[:trn]
        y_tr_raw = y_seq_raw[:trn]

        self.scaler_y = RobustScaler().fit(y_tr_raw.reshape(-1, 1))
        self.scaler_x = RobustScaler().fit(X_ex_tr_raw)

        flat_seq = X_seq_raw.reshape(-1, 1)
        flat_seq_scaled = self.scaler_y.transform(flat_seq)
        X_seq = flat_seq_scaled.reshape(X_seq_raw.shape)

        X_ex = self.scaler_x.transform(X_ex_raw)
        y_seq = self.scaler_y.transform(y_seq_raw.reshape(-1, 1)).flatten()

        Xt, Xe, yt = X_seq[:trn], X_ex[:trn], y_seq[:trn]
        Xv, Xev, yv = X_seq[trn:trn + vn], X_ex[trn:trn + vn], y_seq[trn:trn + vn]
        Xte, Xe_te, yte = X_seq[-tn:], X_ex[-tn:], y_seq[-tn:]

        # MASE must use the training target in the original scale.
        y_train_inv = y_tr_raw

        exog_dim = X_ex.shape[1]
        train_loader = _make_loader(Xt, Xe, yt, self.batch_size, shuffle=True)
        val_loader = _make_loader(Xv, Xev, yv, self.batch_size, shuffle=False)
        test_loader = _make_loader(Xte, Xe_te, yte, self.batch_size, shuffle=False)

        for arch in self.architectures:

            def objective(trial):
                hp = _sample_crypto_hparams(trial, arch, self.window_size)
                lr = hp.pop('learning_rate')
                set_global_seed(self.seed)
                model = CryptoNet(self.window_size, exog_dim, arch, hp)
                _, val_smape = _train_model(model, train_loader, val_loader, lr, self.epochs, patience=5, task='reg', device=DEVICE)
                return val_smape

            study = run_optuna_study(
                objective,
                direction='minimize',
                n_trials=self.max_trials,
                seed=self.seed,
                wandb_project=self.wandb_project,
                run_name=f'NeuralForecaster_{arch}',
                study_name=f'neural_{arch}',
            )

            best_params = dict(study.best_params)
            lr = best_params.pop('learning_rate')
            set_global_seed(self.seed)
            model = CryptoNet(self.window_size, exog_dim, arch, best_params)
            model, _ = _train_model(model, train_loader, val_loader, lr, self.epochs, patience=5, task='reg', device=DEVICE)
            self.models_[arch] = model

            model.eval()
            preds = []
            with torch.no_grad():
                for xb_seq, xb_ex, _ in test_loader:
                    preds.append(model(xb_seq, xb_ex).squeeze(-1))
            pred_scaled = torch.cat(preds).numpy()
            pred_inv = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

            n_pred = len(pred_inv)
            true_inv = self.scaler_y.inverse_transform(yte[:n_pred].reshape(-1, 1)).flatten()

            metrics = compute_metrics(
                y_true=true_inv,
                y_pred=pred_inv,
                y_train=y_train_inv
            )
            self.results_[arch] = ForecastResult(
                {'test': metrics},
                pd.Series(pred_inv, index=df.index[-tn:][:n_pred]),
                model
            )

            if self.wandb_project:
                wandb.init(project=self.wandb_project, name=f'NeuralForecaster_{arch}_final', reinit=True)
                wandb.log({f'{arch}/test_{k}': v for k, v in metrics.items()})
                wandb.finish()

        return self

    def predict(self):
        return self.results_


# ==================== ARIMA Forecaster ====================================
class ARIMAForecaster(ForecastBase):
    def __init__(self,
                 order=(1, 0, 0),
                 seasonal_order=(0, 0, 0, 0),
                 test_size: Optional[int] = None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.test_size = test_size
        self.scaler = RobustScaler()
        self.model_ = None
        self.result_ = None

    def fit(self, df: pd.DataFrame, target_col: str) -> 'ARIMAForecaster':
        y_all = df[[target_col]].values

        # Time-based train/test split. Fit the scaler only on train to avoid leakage.
        if self.test_size:
            y_train_raw = y_all[:-self.test_size]
            y_test_raw = y_all[-self.test_size:]
            idx_train = df.index[:-self.test_size]
            idx_test = df.index[-self.test_size:]
        else:
            y_train_raw = y_all
            y_test_raw = None
            idx_train = df.index
            idx_test = None

        y_train_scaled = self.scaler.fit_transform(y_train_raw).flatten()
        y_test_scaled = (
            self.scaler.transform(y_test_raw).flatten()
            if y_test_raw is not None else None
        )

        self.model_ = sm.tsa.SARIMAX(
            y_train_scaled,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        train_pred_scaled = np.asarray(self.model_.fittedvalues)
        self.y_pred_train = self.scaler.inverse_transform(
            train_pred_scaled.reshape(-1, 1)
        ).flatten()
        self.y_train = y_train_raw.flatten()
        self.train_index_ = idx_train

        if y_test_scaled is not None:
            y_pred_scaled = np.asarray(
                self.model_.get_forecast(steps=len(y_test_scaled)).predicted_mean
            )
            y_pred = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = y_test_raw.flatten()
            y_train_inv = y_train_raw.flatten()

            metrics = {
                'train': compute_metrics(
                    y_true=self.y_train,
                    y_pred=self.y_pred_train,
                    y_train=self.y_train
                ),
                'test': compute_metrics(
                    y_true=y_true,
                    y_pred=y_pred,
                    y_train=y_train_inv
                )
            }
            self.y_pred_test = y_pred
            self.test_index_ = idx_test
            self.result_ = ForecastResult(
                metrics,
                pd.Series(y_pred, index=idx_test),
                self.model_
            )
        return self

    def predict(self) -> ForecastResult:

        if self.result_ is not None:
            return self.result_

        y_pred_scaled = np.asarray(self.model_.fittedvalues)
        y_pred = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        return ForecastResult({}, pd.Series(y_pred), self.model_)


# ==================== Classification Utilities ====================

@dataclass
class ClassificationResult:
    """
    Класс-контейнер для результатов классификационных моделей.
    """
    metrics: Dict[str, Dict[str, float]]
    y_pred:  pd.Series
    y_proba: pd.Series
    model:   Any


def compute_clf_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Метрики бинарной классификации:
    * accuracy, f1, precision, recall, roc_auc
    """
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'roc_auc':   float(roc_auc_score(y_true, y_proba)) if y_proba is not None else None,
    }


# ==================== Linear Classifier ====================

class LinearClassifier(ForecastBase, BaseEstimator):
    """
    Логистическая регрессия с регуляризацией для классификации направления BTC.

    Таргет: 1 если BTC[t] > BTC[t-1], иначе 0.

    `test_size`: число последних точек для теста
    `cv_splits`: число фолдов в кросс-валидации
    `n_trials`:  число испытаний `Optuna`
    `seed`:      random seed
    `wandb_project`: если задан - логирует тюнинг в Weights & Biases
    """
    def __init__(
        self,
        test_size: int = 52,
        cv_splits: int = 5,
        n_trials: int = 50,
        seed: int = 42,
        wandb_project: Optional[str] = None,
    ):
        self.test_size  = test_size
        self.cv_splits  = cv_splits
        self.n_trials = n_trials
        self.seed = seed
        self.wandb_project = wandb_project
        self.model_: Optional[BaseEstimator] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LinearClassifier':
        self.log("LogisticRegression начала обучаться")

        X_train, X_test = X.iloc[:-self.test_size], X.iloc[-self.test_size:]
        y_train, y_test = y.iloc[:-self.test_size], y.iloc[-self.test_size:]

        def build_model(params: Dict[str, Any]) -> Pipeline:
            return Pipeline([
                ('imputer', SimpleImputer()),
                ('scaler',  RobustScaler()),
                ('clf',     LogisticRegression(max_iter=2000, class_weight='balanced', **params))
            ])

        cv = TimeSeriesSplit(n_splits=self.cv_splits)

        def objective(trial: optuna.Trial) -> float:
            solver = trial.suggest_categorical('solver', ['lbfgs', 'saga', 'liblinear'])
            params: Dict[str, Any] = {
                'C': trial.suggest_float('C', 1e-4, 1e4, log=True),
                'solver': solver,
            }
            if solver == 'saga':
                penalty = trial.suggest_categorical('penalty', ['l2', 'elasticnet'])
                params['penalty'] = penalty
                if penalty == 'elasticnet':
                    params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
            else:
                params['penalty'] = 'l2'

            model = build_model(params)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            return scores.mean()

        study = run_optuna_study(
            objective,
            direction='maximize',
            n_trials=self.n_trials,
            seed=self.seed,
            wandb_project=self.wandb_project,
            run_name='LinearClassifier',
            study_name='linear_classifier',
        )
        self.study_ = study

        best_params = dict(study.best_params)
        best_params.pop('penalty', None)
        best_params.pop('l1_ratio', None)
        clf_params: Dict[str, Any] = {'C': best_params['C'], 'solver': best_params['solver']}
        if study.best_params['solver'] == 'saga':
            clf_params['penalty'] = study.best_params['penalty']
            if study.best_params['penalty'] == 'elasticnet':
                clf_params['l1_ratio'] = study.best_params['l1_ratio']
        else:
            clf_params['penalty'] = 'l2'

        model = build_model(clf_params)
        model.fit(X_train, y_train)
        self.model_ = model

        self.log("LogisticRegression закончила обучаться")

        self.y_pred_train  = model.predict(X_train)
        self.y_proba_train = model.predict_proba(X_train)[:, 1]
        self.y_pred_test   = model.predict(X_test)
        self.y_proba_test  = model.predict_proba(X_test)[:, 1]

        metrics = {
            'train': compute_clf_metrics(y_train.values, self.y_pred_train, self.y_proba_train),
            'test':  compute_clf_metrics(y_test.values,  self.y_pred_test,  self.y_proba_test),
        }
        self.result_ = ClassificationResult(
            metrics,
            pd.Series(self.y_pred_test,  index=X_test.index),
            pd.Series(self.y_proba_test, index=X_test.index),
            model
        )
        return self

    def predict(self) -> Dict[str, np.ndarray]:
        return {'train': self.y_pred_train, 'test': self.y_pred_test}


# ==================== Stacking Classifier ====================

class StackingClassifierModel(ForecastBase, BaseEstimator):
    """
    Стекинг-классификатор для бинарной классификации направления BTC.

    Базовые модели: выбранная (RF/CatBoost/XGBoost/LightGBM) + KNN.
    Мета-модель: LogisticRegression.

    `model_type`: 'RandomForest', 'CatBoost', 'XGBoost' или 'LightGBM'
    `test_size`:  число последних точек для теста
    `cv_splits`:  число фолдов в кросс-валидации
    `n_trials`:   число испытаний `Optuna`
    `seed`:       random seed
    `wandb_project`: если задан - логирует тюнинг в Weights & Biases
    """
    def __init__(
        self,
        model_type: str = 'RandomForest',
        test_size: int = 260,
        cv_splits: int = 5,
        n_trials: int = 50,
        seed: int = 42,
        wandb_project: Optional[str] = None,
    ):
        allowed = {'RandomForest', 'CatBoost', 'XGBoost', 'LightGBM'}
        if model_type not in allowed:
            raise ValueError(f"тип модели должен быть одним из следующих: {allowed}")

        self.model_type = model_type
        self.test_size  = test_size
        self.cv_splits  = cv_splits
        self.n_trials   = n_trials
        self.seed       = seed
        self.wandb_project = wandb_project

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'StackingClassifierModel':
        if len(X) <= self.test_size:
            raise ValueError(f"Недостаточно данных ({len(X)}) для test_size={self.test_size}")

        self.log(f"{self.model_type} (classifier) начала обучаться")

        X_train, X_test = X.iloc[:-self.test_size], X.iloc[-self.test_size:]
        y_train, y_test = y.iloc[:-self.test_size], y.iloc[-self.test_size:]

        numeric_cols = X.select_dtypes('number').columns.tolist()

        def build_model() -> Pipeline:
            preprocessor = ColumnTransformer(
                transformers=[('continuous', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler',  RobustScaler()),
                ]), numeric_cols)],
                remainder='drop'
            )


            _GPU = torch.cuda.is_available()


            base_estimators = {
                'RandomForest': [
                    ("rf",  RandomForestClassifier(random_state=self.seed, class_weight='balanced')),
                    ("knn", KNeighborsClassifier())
                ],
                'CatBoost': [
                    ("cat", CatBoostRegressor(random_seed=self.seed, verbose=0,
                                            task_type='GPU' if _GPU else 'CPU')),
                ],
                'XGBoost': [
                    ("xgb", XGBRegressor(random_state=self.seed, verbosity=0, n_jobs=-1,
                                        device='cuda' if _GPU else 'cpu')),
                ],
                'LightGBM': [
                    ('lgbm', LGBMClassifier(random_state=self.seed, n_jobs=-1,
                                            class_weight='balanced', verbose=-1)),
                    ("knn", KNeighborsClassifier())
                ]
            }

            stack = SklearnStackingClassifier(
                estimators      = base_estimators[self.model_type],
                final_estimator = LogisticRegression(class_weight='balanced', max_iter=1000),
                cv              = self.cv_splits,
                passthrough     = True,
                n_jobs          = -1
            )

            return Pipeline([
                ('prep',  preprocessor),
                ('stack', stack)
            ])

        cv = TimeSeriesSplit(n_splits=self.cv_splits)
        knn_param_keys = build_model().get_params().keys()

        def suggest_knn(trial: optuna.Trial) -> Dict[str, Any]:
            params = {
                'stack__knn__n_neighbors': trial.suggest_int('stack__knn__n_neighbors', 3, 31),
                'stack__knn__weights': trial.suggest_categorical('stack__knn__weights', ['uniform', 'distance']),
            }
            if 'stack__knn__p' in knn_param_keys:
                params['stack__knn__p'] = trial.suggest_categorical('stack__knn__p', [1, 2])
            return params

        def objective(trial: optuna.Trial) -> float:
            params: Dict[str, Any] = {}

            if self.model_type == 'RandomForest':
                params.update({
                    'stack__rf__n_estimators': trial.suggest_int('stack__rf__n_estimators', 100, 1500),
                    'stack__rf__max_depth': trial.suggest_int('stack__rf__max_depth', 5, 50),
                    'stack__rf__min_samples_leaf': trial.suggest_int('stack__rf__min_samples_leaf', 1, 20),
                    'stack__rf__min_samples_split': trial.suggest_int('stack__rf__min_samples_split', 2, 20),
                    'stack__rf__max_features': trial.suggest_float('stack__rf__max_features', 0.3, 1.0),
                })
                params.update(suggest_knn(trial))
                params['stack__final_estimator__C'] = trial.suggest_float('stack__final_estimator__C', 1e-3, 10, log=True)

            elif self.model_type == 'CatBoost':
                bootstrap_type = trial.suggest_categorical('stack__cat__bootstrap_type', ['Bayesian', 'Bernoulli'])
                params.update({
                    'stack__cat__learning_rate': trial.suggest_float('stack__cat__learning_rate', 0.01, 0.3),
                    'stack__cat__depth': trial.suggest_int('stack__cat__depth', 4, 11),
                    'stack__cat__l2_leaf_reg': trial.suggest_float('stack__cat__l2_leaf_reg', 1, 20),
                    'stack__cat__iterations': trial.suggest_int('stack__cat__iterations', 100, 1500),
                    'stack__cat__border_count': trial.suggest_int('stack__cat__border_count', 32, 255),
                    'stack__cat__random_strength': trial.suggest_float('stack__cat__random_strength', 0, 10),
                    'stack__cat__bootstrap_type': bootstrap_type,
                })
                if bootstrap_type == 'Bayesian':
                    params['stack__cat__bagging_temperature'] = trial.suggest_float('stack__cat__bagging_temperature', 0, 10)
                else:
                    params['stack__cat__subsample'] = trial.suggest_float('stack__cat__subsample', 0.5, 1.0)
                params.update(suggest_knn(trial))
                params['stack__final_estimator__C'] = trial.suggest_float('stack__final_estimator__C', 1e-4, 1.0)

            elif self.model_type == 'XGBoost':
                params.update({
                    'stack__xgb__eta': trial.suggest_float('stack__xgb__eta', 0.01, 0.3),
                    'stack__xgb__max_depth': trial.suggest_int('stack__xgb__max_depth', 3, 10),
                    'stack__xgb__subsample': trial.suggest_float('stack__xgb__subsample', 0.5, 1.0),
                    'stack__xgb__colsample_bytree': trial.suggest_float('stack__xgb__colsample_bytree', 0.5, 1.0),
                    'stack__xgb__n_estimators': trial.suggest_int('stack__xgb__n_estimators', 100, 1500),
                    'stack__xgb__gamma': trial.suggest_float('stack__xgb__gamma', 0, 5),
                    'stack__xgb__min_child_weight': trial.suggest_int('stack__xgb__min_child_weight', 1, 10),
                })
                params.update(suggest_knn(trial))
                params['stack__final_estimator__C'] = trial.suggest_float('stack__final_estimator__C', 1e-4, 1.0)

            elif self.model_type == 'LightGBM':
                params.update({
                    'stack__lgbm__learning_rate': trial.suggest_float('stack__lgbm__learning_rate', 0.01, 0.3),
                    'stack__lgbm__num_leaves': trial.suggest_int('stack__lgbm__num_leaves', 20, 149),
                    'stack__lgbm__min_child_samples': trial.suggest_int('stack__lgbm__min_child_samples', 5, 99),
                    'stack__lgbm__feature_fraction': trial.suggest_float('stack__lgbm__feature_fraction', 0.5, 1.0),
                    'stack__lgbm__bagging_fraction': trial.suggest_float('stack__lgbm__bagging_fraction', 0.5, 1.0),
                    'stack__lgbm__bagging_freq': trial.suggest_int('stack__lgbm__bagging_freq', 1, 10),
                    'stack__lgbm__min_split_gain': trial.suggest_float('stack__lgbm__min_split_gain', 0, 1),
                    'stack__lgbm__n_estimators': trial.suggest_int('stack__lgbm__n_estimators', 100, 1500),
                })
                params.update(suggest_knn(trial))
                params['stack__final_estimator__C'] = trial.suggest_float('stack__final_estimator__C', 1e-4, 1.0)

            model = build_model()
            model.set_params(**params)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=4)
            return scores.mean()

        study = run_optuna_study(
            objective,
            direction='maximize',
            n_trials=self.n_trials,
            seed=self.seed,
            wandb_project=self.wandb_project,
            run_name=f'StackingClassifier_{self.model_type}',
            study_name=f'stacking_clf_{self.model_type}',
        )
        self.study_ = study

        model = build_model()
        model.set_params(**study.best_params)
        model.fit(X_train, y_train)
        self.model_ = model

        self.log(f"{self.model_type} (classifier) закончила обучаться")

        self.y_pred_train  = model.predict(X_train)
        self.y_proba_train = model.predict_proba(X_train)[:, 1]
        self.y_pred_test   = model.predict(X_test)
        self.y_proba_test  = model.predict_proba(X_test)[:, 1]

        metrics = {
            'train': compute_clf_metrics(y_train.values, self.y_pred_train, self.y_proba_train),
            'test':  compute_clf_metrics(y_test.values,  self.y_pred_test,  self.y_proba_test),
        }
        self.result_ = ClassificationResult(
            metrics,
            pd.Series(self.y_pred_test,  index=X_test.index),
            pd.Series(self.y_proba_test, index=X_test.index),
            model
        )
        return self

    def predict(self) -> Dict[str, np.ndarray]:
        return {'train': self.y_pred_train, 'test': self.y_pred_test}


# ==================== Neural Classifier ====================

class NeuralClassifier:
    """
    Нейронный классификатор направления цены BTC (PyTorch + Optuna).

    Параллельно обучает 5 архитектур (MLP, LSTM, StackedLSTM, CNN_LSTM, GRU),
    подбор гиперпараметров каждой - через `run_optuna_study` (TPESampler),
    максимизируя `val_auc`.

    `window_size`: длина последовательности для входа LSTM/CNN
    `test_size`:   размер тест. выборки
    """
    def __init__(
        self,
        window_size: int = 10,
        test_size:   int = 260,
        max_trials:  int = 10,
        epochs:      int = 50,
        batch_size:  int = 32,
        seed:        int = 42,
        wandb_project: Optional[str] = None,
    ):
        self.window_size   = window_size
        self.test_size     = test_size
        self.max_trials    = max_trials
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.seed          = seed
        self.wandb_project = wandb_project
        self.architectures = ['MLP', 'LSTM', 'StackedLSTM', 'CNN_LSTM', 'GRU']
        self.models_       = {}
        self.results_      = {}
        set_global_seed(self.seed)

    def fit(self, df: pd.DataFrame, target_col: str) -> 'NeuralClassifier':
        """
        `df`:         датасет с признаками и бинарным таргетом (0/1)
        `target_col`: имя бинарной целевой колонки
        """
        y_raw    = df[target_col].values.astype(np.float32)
        exog_raw = df.drop(columns=[target_col]).values

        X_seq_raw, X_ex_raw, y_seq_raw = [], [], []
        for i in range(len(y_raw) - self.window_size):
            X_seq_raw.append(y_raw[i: i + self.window_size].reshape(self.window_size, 1))
            X_ex_raw.append(exog_raw[i + self.window_size])
            y_seq_raw.append(y_raw[i + self.window_size])

        X_seq_raw = np.array(X_seq_raw)
        X_ex_raw  = np.array(X_ex_raw)
        y_seq_raw = np.array(y_seq_raw)

        N, tn = len(y_seq_raw), self.test_size
        rem   = N - tn

        if rem <= 0:
            warnings.warn(f"Недостаточно данных: total={N}, test={tn}")
            return self

        vn, trn = int(0.2 * rem), rem - int(0.2 * rem)

        if trn <= 0:
            warnings.warn(f"Недостаточно данных для трейн/вал: remaining={rem}")
            return self

        # Масштабируем экзогенные признаки и последовательность BTC (не таргет)
        self.scaler_x   = RobustScaler().fit(X_ex_raw[:trn])
        self.scaler_seq = RobustScaler().fit(X_seq_raw[:trn].reshape(-1, 1))

        X_ex  = self.scaler_x.transform(X_ex_raw)
        flat  = self.scaler_seq.transform(X_seq_raw.reshape(-1, 1))
        X_seq = flat.reshape(X_seq_raw.shape)

        Xt,  Xe,   yt  = X_seq[:trn],       X_ex[:trn],       y_seq_raw[:trn]
        Xv,  Xev,  yv  = X_seq[trn:trn+vn], X_ex[trn:trn+vn], y_seq_raw[trn:trn+vn]
        Xte, Xe_te, yte = X_seq[-tn:],      X_ex[-tn:],        y_seq_raw[-tn:]

        exog_dim = X_ex.shape[1]
        train_loader = _make_loader(Xt, Xe, yt, self.batch_size, shuffle=True)
        val_loader = _make_loader(Xv, Xev, yv, self.batch_size, shuffle=False)
        test_loader = _make_loader(Xte, Xe_te, yte, self.batch_size, shuffle=False)

        for arch in self.architectures:

            def objective(trial):
                hp = _sample_crypto_hparams(trial, arch, self.window_size)
                lr = hp.pop('learning_rate')
                set_global_seed(self.seed)
                model = CryptoNet(self.window_size, exog_dim, arch, hp, output_activation='sigmoid')
                _, val_auc = _train_model(model, train_loader, val_loader, lr, self.epochs, patience=5, task='clf')
                return val_auc

            study = run_optuna_study(
                objective,
                direction='maximize',
                n_trials=self.max_trials,
                seed=self.seed,
                wandb_project=self.wandb_project,
                run_name=f'NeuralClassifier_{arch}',
                study_name=f'neural_clf_{arch}',
            )

            best_params = dict(study.best_params)
            lr = best_params.pop('learning_rate')
            set_global_seed(self.seed)
            model = CryptoNet(self.window_size, exog_dim, arch, best_params, output_activation='sigmoid')
            model, _ = _train_model(model, train_loader, val_loader, lr, self.epochs, patience=5, task='clf')
            self.models_[arch] = model

            model.eval()
            logits = []
            with torch.no_grad():
                for xb_seq, xb_ex, _ in test_loader:
                    logits.append(model(xb_seq, xb_ex).squeeze(-1))
            proba = torch.sigmoid(torch.cat(logits)).numpy()
            pred  = (proba > 0.5).astype(int)

            n_pred = len(proba)
            metrics = compute_clf_metrics(
                y_true=yte[:n_pred].astype(int),
                y_pred=pred,
                y_proba=proba
            )
            self.results_[arch] = ClassificationResult(
                {'test': metrics},
                pd.Series(pred,  index=df.index[-tn:][:n_pred]),
                pd.Series(proba, index=df.index[-tn:][:n_pred]),
                model
            )

            if self.wandb_project:
                wandb.init(project=self.wandb_project, name=f'NeuralClassifier_{arch}_final', reinit=True)
                wandb.log({f'{arch}/test_{k}': v for k, v in metrics.items()})
                wandb.finish()

        return self

    def predict(self) -> Dict[str, ClassificationResult]:
        return self.results_
