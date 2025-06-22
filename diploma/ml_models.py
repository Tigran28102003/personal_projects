import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass
import os
import shutil
import random

import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import kerastuner as kt
import tensorflow as tf
from tensorflow.keras import Input, Model, callbacks, optimizers
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, concatenate, BatchNormalization
)

from scipy.stats import randint, uniform, loguniform
import statsmodels.api as sm

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore')
warnings.filterwarnings(
    action="ignore",
    message="X has feature names, but KNeighborsRegressor was fitted without feature names"
)

# ==================== Utility Functions ====================

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    `Symmetric Mean Absolute Percentage Error` -
    метрика качества, симметричное отношение к пере- и недопрогнозу

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
    return float(
                np.mean(np.abs(y_true - y_pred)) / # ср. абс. ошибка предсказаний на тесте
                np.mean(np.abs(np.diff(y_train)))  # ср. абс. ошибка наивного предсказания на трейне
            )


def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    подсчет метрик на трейн- и тест-выборках

    метрики:
    * `MAE` - mean absolute error
    * `RMSE` - root mean squared error
    * `SMAPE` - symmetric mean absolute percentage error
    * `MASE` - mean absolute scaled error
    """
    mae_val = mean_absolute_error(y_true, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    smape_val = smape(y_true, y_pred)

    if y_train is not None:
        mase_val = mase(y_true, y_pred, y_train)
    else:
        mase_val = None

    return {'mae': mae_val, 'rmse': rmse_val, 'smape': smape_val, 'mase': mase_val}



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

    * разбивка на трейн и тест происходит внутри самого метода
    * подбор параметров `GridSearchCV`
    * метрика качества - `SMAPE`
    """
    def __init__(self, test_size: int = 52, cv_splits: int = 5, model_type: str = 'Ridge'):
        self.test_size = test_size
        self.cv_splits = cv_splits
        self.model_type = model_type
        self.model_: Optional[GridSearchCV] = None

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
        pre = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer()),
                ('scaler', RobustScaler())
            ]), numeric)
        ])

        model_map = {
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet()
        }
        base = model_map.get(self.model_type, Ridge())
        feature_pipe = Pipeline([('pre', pre), ('reg', base)])

        ttr = TransformedTargetRegressor(
            regressor=feature_pipe,
            transformer=RobustScaler()
        )

        grid = {
            'regressor__reg__alpha': np.logspace(-4, 4, 20),
            'regressor__reg__fit_intercept': [True, False],
            'regressor__reg__tol': [1e-2, 1e-3, 1e-4]
        }
        if self.model_type == 'ElasticNet':
            grid.update({
                'regressor__reg__l1_ratio': np.linspace(0,1,20),
                'regressor__reg__max_iter': [1000, 5000]
            })

        cv = TimeSeriesSplit(n_splits=self.cv_splits)
        scorer = make_scorer(smape, greater_is_better=False)
        gs = GridSearchCV(
            estimator=ttr,
            param_grid=grid,
            cv=cv,
            scoring=scorer,
            n_jobs=-1
        )
        gs.fit(X_train, y_train)
        self.model_ = gs
        self.log(f"{self.model_type} закончила обучаться")

        self.y_pred_train = gs.predict(X_train)
        self.y_pred_test = gs.predict(X_test)

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
            gs
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
    `n_iter`:     число итераций Randomized Search
    `seed`:       random seed
    """
    def __init__(
        self,
        model_type: str = 'RandomForest',
        test_size: int = 260,
        cv_splits: int = 5,
        n_iter: int = 20,
        seed: int = 42
    ):
        allowed = {'RandomForest', 'CatBoost', 'XGBoost', 'LightGBM'}
        if model_type not in allowed:
            raise ValueError(f"тип модели должен быть одним из следующих: {allowed}")

        self.model_type = model_type
        self.test_size = test_size
        self.cv_splits = cv_splits
        self.n_iter = n_iter
        self.seed = seed

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

        preprocessor = ColumnTransformer(
            transformers=[
                ('continuous', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler',  RobustScaler()),
                ]),
                X.select_dtypes('number').columns.tolist())
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
                ('lgbm', LGBMRegressor(random_state=self.seed, n_jobs=-1)),
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

        model = TransformedTargetRegressor(
            regressor   = stack_pipe,
            transformer = RobustScaler()
        )

        param_dist = {
            'RandomForest': {
                # Random Forest
                'regressor__stack__rf__n_estimators':      randint(100, 500),
                'regressor__stack__rf__max_depth':         randint(5, 50),
                'regressor__stack__rf__min_samples_leaf':  randint(1, 20),
                'regressor__stack__rf__min_impurity_decrease': loguniform(1e-5, 1e-2),
                'regressor__stack__rf__bootstrap':         [True],
                'regressor__stack__rf__max_samples':       uniform(0.5, 0.5),

                # KNN
                'regressor__stack__knn__n_neighbors': randint(3, 15),
                'regressor__stack__knn__weights':    ['uniform', 'distance'],

                # Ridge
                'regressor__stack__final_estimator__alpha':    loguniform(1e-3, 10),
            },
            'CatBoost': {
                # CatBoostRegressor
                'regressor__stack__cat__learning_rate': uniform(0.01, 0.29),      # [0.01, 0.3)
                'regressor__stack__cat__depth': randint(4, 12),                   # целые от 4 до 11
                'regressor__stack__cat__l2_leaf_reg': uniform(1, 19),             # [1, 20)
                'regressor__stack__cat__iterations': randint(100, 1000),          # [100, 999]
                'regressor__stack__cat__border_count': randint(32, 256),          # [32, 255]

                # KNeighborsRegressor
                'regressor__stack__knn__n_neighbors': randint(3, 31),             # [3, 30]
                'regressor__stack__knn__weights': ['uniform', 'distance'],
                'regressor__stack__knn__p': [1, 2],

                # Ridge
                'regressor__stack__final_estimator__alpha': uniform(1e-4, 1.0),    # [1e-4, 1.0001)
            },
            'XGBoost': {
                # XGBRegressor
                'regressor__stack__xgb__eta': uniform(0.01, 0.29),                 # [0.01, 0.3)
                'regressor__stack__xgb__max_depth': randint(3, 11),                # [3, 10]
                'regressor__stack__xgb__subsample': uniform(0.5, 0.5),             # [0.5, 1.0)
                'regressor__stack__xgb__colsample_bytree': uniform(0.5, 0.5),      # [0.5, 1.0)
                'regressor__stack__xgb__lambda': loguniform(1e-3, 10),             # L2-регуляризация
                'regressor__stack__xgb__alpha': loguniform(1e-3, 10),              # L1-регуляризация
                'regressor__stack__xgb__n_estimators': randint(100, 1000),         # [100, 999]

                # KNeighborsRegressor
                'regressor__stack__knn__n_neighbors': randint(3, 31),
                'regressor__stack__knn__weights': ['uniform', 'distance'],
                'regressor__stack__knn__p': [1, 2],

                # Ridge
                'regressor__stack__final_estimator__alpha': uniform(1e-4, 1.0),
            },
            'LightGBM': {
                # LGBMRegressor
                'regressor__stack__lgbm__learning_rate': uniform(0.01, 0.29),      # [0.01, 0.3)
                'regressor__stack__lgbm__num_leaves': randint(20, 150),            # [20, 149]
                'regressor__stack__lgbm__min_child_samples': randint(5, 100),      # [5, 99]
                'regressor__stack__lgbm__subsample': uniform(0.5, 0.5),            # [0.5, 1.0)
                'regressor__stack__lgbm__colsample_bytree': uniform(0.5, 0.5),     # [0.5, 1.0)
                'regressor__stack__lgbm__reg_alpha': loguniform(1e-3, 10),
                'regressor__stack__lgbm__reg_lambda': loguniform(1e-3, 10),
                'regressor__stack__lgbm__n_estimators': randint(100, 1000),

                # KNeighborsRegressor
                'regressor__stack__knn__n_neighbors': randint(3, 31),
                'regressor__stack__knn__weights': ['uniform', 'distance'],
                'regressor__stack__knn__p': [1, 2],

                # Ridge
                'regressor__stack__final_estimator__alpha': uniform(1e-4, 1.0),
            }
        }

        rs_stack = RandomizedSearchCV(
            estimator           = model,
            param_distributions = param_dist[self.model_type],
            n_iter              = self.n_iter,
            cv                  = TimeSeriesSplit(n_splits=self.cv_splits),
            scoring             = make_scorer(smape, greater_is_better=False),
            random_state        = self.seed,
            n_jobs              = 4
        )

        rs_stack.fit(X_train, y_train)
        self.model_ = rs_stack

        self.log(f"{self.model_type} закончила обучаться")

        # для сравнения проверяем метрики качества на трейн и тест данных
        self.y_pred_train = rs_stack.predict(X_train)
        self.y_pred_test = rs_stack.predict(X_test)

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
            rs_stack
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
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


def smape_loss(y_true: np.array, y_pred: np.array):
    """
    `Symmetric Mean Absolute Percentage Error` -
    метрика качества, симметричное отношение к пере- и недопрогнозу

    посчитана с использованием библиотеки tensorflow

    `y_true`: тагрет на тесте
    `y_pred`: предсказания на тесте
    """
    eps = tf.keras.backend.epsilon()
    denom = (tf.abs(y_true) + tf.abs(y_pred) + eps) / 2.0
    diff = tf.abs(y_true - y_pred)
    smape = tf.reduce_mean(diff / denom)

    return smape

class CryptoHyperModel(kt.HyperModel):
    """

    """
    def __init__(self, window_size: int, exog_dim: int, fixed_arch: str = None):
        self.window_size = window_size
        self.exog_dim = exog_dim
        self.fixed_arch = fixed_arch


    def build(self, hp):
        arch = self.fixed_arch or hp.Choice('architecture', ['MLP', 'LSTM', 'StackedLSTM', 'CNN_LSTM', 'GRU'])
        inp_seq = Input(shape=(self.window_size, 1), name='seq_input')
        inp_ex = Input(shape=(self.exog_dim,), name='exog_input')

        if arch == 'MLP':

            x = Flatten()(inp_seq)
            for i in range(hp.Int('mlp_layers', 1, 3)):
                x = Dense(hp.Int(f'mlp_units_{i}', 32, 128, 32), activation='relu')(x)
                x = Dropout(hp.Float(f'mlp_drop_{i}', 0.1, 0.5, 0.1))(x)
                x = BatchNormalization()(x)

        elif arch == 'LSTM':

            x = LSTM(hp.Int('lstm_units', 32, 128, 32))(inp_seq)
            x = Dropout(hp.Float('lstm_drop', 0.1, 0.5, 0.1))(x)

        elif arch == 'StackedLSTM':

            x = inp_seq
            layers = hp.Int('stacked_layers', 2, 3)

            for i in range(layers):
                return_sequences = (i < layers - 1)
                x = LSTM(
                    hp.Int(f'stack_units_{i}', 32, 128, 32),
                    return_sequences=return_sequences
                )(x)
                x = Dropout(hp.Float(f'stack_drop_{i}', 0.1, 0.5, 0.1))(x)

        elif arch == 'CNN_LSTM':

            x = Conv1D(
                filters=hp.Int('cnn_filters', 16, 64, 16),
                kernel_size=hp.Int('cnn_kernel', 2, 5),
                activation='relu'
            )(inp_seq)
            x = MaxPooling1D(pool_size=hp.Int('pool_size', 2, 4))(x)
            x = Dropout(hp.Float('cnn_drop', 0.1, 0.5, 0.1))(x)
            x = LSTM(hp.Int('cnn_lstm_units', 32, 128, 32))(x)

        else:
            x = GRU(hp.Int('gru_units', 32, 128, 32))(inp_seq)
            x = Dropout(hp.Float('gru_drop', 0.1, 0.5, 0.1))(x)

        y_ex = inp_ex
        for i in range(hp.Int('ex_layers', 1, 2)):
            y_ex = Dense(hp.Int(f'ex_units_{i}', 16, 64, 16), activation='relu')(y_ex)
            y_ex = Dropout(hp.Float(f'ex_drop_{i}', 0.1, 0.5, 0.1))(y_ex)

        merged = concatenate([x, y_ex])
        out = Dense(1, name='output')(merged)
        model = Model([inp_seq, inp_ex], out)

        model.compile(
            optimizer=optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
            loss=smape_loss,
            metrics=[smape_loss, 'mae', 'mse']
        )
        return model


class NeuralForecaster:
    """

    """
    def __init__(
        self,
        window_size=10,
        test_size=260,
        max_trials=10,
        epochs=50,
        batch_size=32,
        seed=42
    ):
        self.window_size = window_size
        self.test_size = test_size
        self.max_trials = max_trials
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
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

        # Для обратного преобразования y_train (MASE)
        y_train_full = y_seq_raw[:trn + self.window_size]
        y_train_inv = self.scaler_y.inverse_transform(np.array(y_train_full).reshape(-1, 1)).flatten()

        for arch in self.architectures:

            project_name = f'tuner_{arch}'
            dirpath = 'keras_tuner_results'
            arch_path = os.path.join(dirpath, project_name)

            if os.path.exists(arch_path):
                shutil.rmtree(arch_path)

            tuner = kt.Hyperband(
                CryptoHyperModel(self.window_size, X_ex.shape[1], fixed_arch=arch),
                objective='val_loss',
                max_epochs=self.epochs,
                factor=3,
                max_consecutive_failed_trials=10,
                seed=self.seed,
                directory=dirpath,
                project_name=project_name
            )

            es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            mc = callbacks.ModelCheckpoint(
                filepath=os.path.join(dirpath, f'{project_name}_best.h5'),
                monitor='val_loss', save_best_only=True
            )
            rlrop = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

            tuner.search(
                [Xt, Xe], yt,
                validation_data=([Xv, Xev], yv),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[es, mc, rlrop]
            )

            model = tuner.get_best_models(num_models=1)[0]
            self.models_[arch] = model

            pred_scaled = model.predict([Xte, Xe_te]).flatten()
            pred_inv = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            true_inv = self.scaler_y.inverse_transform(yte.reshape(-1, 1)).flatten()

            metrics = compute_metrics(
                y_true=true_inv,
                y_pred=pred_inv,
                y_train=y_train_inv
            )
            self.results_[arch] = ForecastResult(
                {'test': metrics},
                pd.Series(pred_inv, index=df.index[-tn:]),
                model
            )

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
        y_scaled = self.scaler.fit_transform(y_all).flatten()

        # Разбиение на train/test
        if self.test_size:
            y_train_scaled = y_scaled[:-self.test_size]
            y_test_scaled = y_scaled[-self.test_size:]
            idx_test = df.index[-self.test_size:]
        else:
            y_train_scaled = y_scaled
            y_test_scaled = None
            idx_test = None

        self.model_ = sm.tsa.SARIMAX(
            y_train_scaled,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        if y_test_scaled is not None:
            y_pred_scaled = self.model_.predict(start=0, end=len(y_test_scaled) - 1)
            y_pred = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = self.scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
            y_train_inv = self.scaler.inverse_transform(
                y_train_scaled.reshape(-1, 1)
            ).flatten()

            metrics = compute_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_train=y_train_inv
            )
            self.result_ = ForecastResult(
                {'test': metrics},
                pd.Series(y_pred, index=idx_test),
                self.model_
            )
        return self

    def predict(self) -> ForecastResult:

        if self.result_ is not None:
            return self.result_

        y_pred_scaled = self.model_.predict()
        y_pred = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        return ForecastResult({}, pd.Series(y_pred), self.model_)
