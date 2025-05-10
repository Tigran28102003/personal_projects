import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass
import os
import shutil

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
    LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, concatenate
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
            raise ValueError(f"model_type must be one of {allowed}")
        
        self.model_type = model_type
        self.test_size = test_size
        self.cv_splits = cv_splits
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        обучение стеккинг-моделей. 
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
              
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        splits = list(
            TimeSeriesSplit(n_splits=self.cv_splits*2).split(X_train, y_train)  
        ) # берем в два раза больше фолдов, т.к. данные достаточно волатильны

        stack = StackingRegressor(
            estimators       = base_estimators[self.model_type],
            final_estimator  = Ridge(),
            cv               = splits,
            passthrough      = True,
            n_jobs           = -1
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
            cv                  = tscv,
            scoring             = make_scorer(smape, greater_is_better=False),
            random_state        = self.seed,
            n_jobs              = -1
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
    
# ==================== Crypto HyperModel ====================
# class CryptoHyperModel(kt.HyperModel):
#     """
#     HyperModel for tuning neural network architectures for time series forecasting.

#     :param window_size: length of past sequence input.
#     :param exog_dim: number of exogenous features.
#     """
#     def __init__(self, window_size: int, exog_dim: int):
#         self.window_size = window_size
#         self.exog_dim = exog_dim

#     def build(self, hp):
#         # Choice of architecture
#         arch = hp.Choice('architecture', ['MLP','LSTM','StackedLSTM','CNN_LSTM','GRU'])
#         inp_seq = Input(shape=(self.window_size, 1), name='seq_input')
#         inp_ex = Input(shape=(self.exog_dim,), name='exog_input')

#         # Build sequence branch
#         if arch == 'MLP':
#             x = Flatten()(inp_seq)
#             for i in range(hp.Int('mlp_layers', 1, 3)):
#                 x = Dense(hp.Int(f'mlp_units_{i}', 32, 128, step=32), activation='relu')(x)
#                 x = Dropout(hp.Float(f'mlp_drop_{i}', 0.1, 0.5, step=0.1))(x)
#         elif arch == 'LSTM':
#             x = LSTM(hp.Int('lstm_units', 32, 128, step=32))(inp_seq)
#             x = Dropout(hp.Float('lstm_drop', 0.1, 0.5, step=0.1))(x)
#         elif arch == 'StackedLSTM':
#             x = inp_seq
#             layers = hp.Int('stacked_layers', 2, 3)
#             for i in range(layers):
#                 return_seq = i < layers - 1
#                 x = LSTM(hp.Int(f'stack_units_{i}', 32, 128, step=32), return_sequences=return_seq)(x)
#                 x = Dropout(hp.Float(f'stack_drop_{i}', 0.1, 0.5, step=0.1))(x)
#         elif arch == 'CNN_LSTM':
#             x = Conv1D(filters=hp.Int('cnn_filters', 16, 64, step=16),
#                        kernel_size=hp.Int('cnn_kernel', 2, 5), activation='relu')(inp_seq)
#             x = MaxPooling1D(pool_size=hp.Int('pool_size', 2, 4))(x)
#             x = Dropout(hp.Float('cnn_drop', 0.1, 0.5, step=0.1))(x)
#             x = LSTM(hp.Int('cnn_lstm_units', 32, 128, step=32))(x)
#         else:
#             x = GRU(hp.Int('gru_units', 32, 128, step=32))(inp_seq)
#             x = Dropout(hp.Float('gru_drop', 0.1, 0.5, step=0.1))(x)

#         # Exogenous branch
#         y_ex = inp_ex
#         for i in range(hp.Int('ex_layers', 1, 2)):
#             y_ex = Dense(hp.Int(f'ex_units_{i}', 16, 64, step=16), activation='relu')(y_ex)
#             y_ex = Dropout(hp.Float(f'ex_drop_{i}', 0.1, 0.5, step=0.1))(y_ex)

#         # Concatenate and output
#         concat = concatenate([x, y_ex])
#         out = Dense(1, name='output')(concat)
#         model = Model([inp_seq, inp_ex], out)
#         model.compile(
#             optimizer=optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
#             loss='mean_absolute_error'
#         )
#         return model

# # ==================== Neural Forecaster ====================

# class NeuralForecaster(ForecastBase):
#     """
#     Neural network forecaster combining sequence and exogenous features.

#     :param window_size: look-back window length for creating input sequences.
#     :param test_size: fixed number of final samples to reserve for testing.
#     :param max_trials: number of HyperModel trials for keras-tuner search.
#     :param epochs: maximum training epochs per trial.
#     :param batch_size: number of samples per gradient update.
#     :param seed: random seed for reproducibility across runs.
#     """
#     def __init__(
#         self,
#         window_size: int = 10,
#         test_size: int = 260,
#         max_trials: int = 10,
#         epochs: int = 50,
#         batch_size: int = 32,
#         seed: int = 42
#     ):
#         # Store initialization parameters
#         self.window_size = window_size
#         self.test_size = test_size
#         self.max_trials = max_trials
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.seed = seed
#         # Placeholders for model and target scaler
#         self.model_: Optional[Model] = None
#         self.scaler_y: Optional[RobustScaler] = None

#     def fit(self, df: pd.DataFrame, target_col: str) -> 'NeuralForecaster':
#         """
#         Prepare data, tune hyperparameters, and train the neural network.

#         Steps:
#         1. Scale target and exogenous features with RobustScaler to reduce effect of outliers.
#         2. Build rolling window sequences of length `window_size` for the target.
#         3. Split into train/validation/test based on fixed test_size and 20% of remaining for validation.
#         4. Use keras-tuner RandomSearch to find best architecture/hyperparameters of CryptoHyperModel.
#         5. Train with EarlyStopping on validation loss.
#         6. Evaluate on test set and store metrics and predictions.

#         :param df: DataFrame containing target and exogenous features.
#         :param target_col: name of the column in df to forecast.
#         :return: self with trained model and result_ populated.
#         """
#         self.log("Starting neural model training")

#         # 1. Scale target and exogenous features
#         scaler_y = RobustScaler()
#         y = scaler_y.fit_transform(df[[target_col]]).flatten()
#         scaler_x = RobustScaler()
#         exog = scaler_x.fit_transform(df.drop(columns=[target_col]))

#         # 2. Build sequences: each sample consists of `window_size` past targets + current exogenous
#         X_seq, X_ex, y_seq = [], [], []
#         for i in range(len(y) - self.window_size):
#             # Sequence of past target values
#             X_seq.append(y[i : i + self.window_size].reshape(self.window_size, 1))
#             # Exogenous features at time i + window_size
#             X_ex.append(exog[i + self.window_size])
#             # True target at time i + window_size
#             y_seq.append(y[i + self.window_size])
#         X_seq, X_ex, y_seq = map(np.array, (X_seq, X_ex, y_seq))

#         # 3. Split into train/validation/test
#         N = len(y_seq)
#         test_n = self.test_size
#         # Reserve last `test_size` samples for test
#         remaining = N - test_n
#         # Use 20% of remaining for validation
#         val_n = int(0.2 * remaining)
#         train_n = remaining - val_n

#         # Training set
#         Xt, Xe, yt = X_seq[:train_n], X_ex[:train_n], y_seq[:train_n]
#         # Validation set
#         Xv, Xev, yv = (
#             X_seq[train_n : train_n + val_n],
#             X_ex[train_n : train_n + val_n],
#             y_seq[train_n : train_n + val_n]
#         )
#         # Test set
#         Xte, Xe_te, yte = X_seq[-test_n:], X_ex[-test_n:], y_seq[-test_n:]

#         # 4. Hyperparameter tuning with keras-tuner
#         tuner = kt.RandomSearch(
#             CryptoHyperModel(self.window_size, X_ex.shape[1]),
#             objective='val_loss',
#             max_trials=self.max_trials,
#             seed=self.seed
#         )
#         # Early stopping to prevent overfitting
#         es = callbacks.EarlyStopping(
#             monitor='val_loss', patience=5, restore_best_weights=True
#         )
#         tuner.search(
#             [Xt, Xe], yt,
#             validation_data=([Xv, Xev], yv),
#             epochs=self.epochs,
#             batch_size=self.batch_size,
#             callbacks=[es]
#         )

#         # 5. Retrieve best model and store
#         model = tuner.get_best_models(num_models=1)[0]
#         self.model_ = model
#         self.scaler_y = scaler_y

#         # 6. Evaluate on test set
#         pred_scaled = model.predict([Xte, Xe_te]).flatten()
#         # Inverse scaling of predictions and true values
#         pred_inv = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
#         y_true_inv = scaler_y.inverse_transform(yte.reshape(-1, 1)).flatten()
#         idx = df.index[-test_n:]

#         # Compute metrics on test set
#         metrics_test = compute_metrics(y_true_inv, y_true_inv, pred_inv)

#         # Store results
#         self.result_ = ForecastResult(
#             metrics={'test': metrics_test},
#             y_pred=pd.Series(pred_inv, index=idx),
#             model=model
#         )
#         self.log("Finished neural model training")
#         return self
# ==================== Crypto HyperModel ====================
# ==================== Crypto HyperModel ====================


# # Custom SMAPE loss
# def smape_loss(y_true, y_pred):
#     eps = tf.keras.backend.epsilon()
#     num = tf.abs(y_pred - y_true)
#     den = tf.abs(y_true) + tf.abs(y_pred) + eps
#     return tf.reduce_mean(2.0 * num / den)

class CryptoHyperModel(kt.HyperModel):
    def __init__(self, window_size: int, exog_dim: int, fixed_arch: str = None):
        """
        
        """
        self.window_size = window_size
        self.exog_dim = exog_dim
        self.fixed_arch = fixed_arch

    def build(self, hp):
        """
        
        """
        arch = self.fixed_arch or hp.Choice('architecture', ['MLP', 'LSTM', 'StackedLSTM', 'CNN_LSTM', 'GRU'])
        inp_seq = Input(shape=(self.window_size, 1), name='seq_input')
        inp_ex = Input(shape=(self.exog_dim,), name='exog_input')

        # Sequence branch
        if arch == 'MLP':
            x = Flatten()(inp_seq)
            for i in range(hp.Int('mlp_layers', 1, 3)):
                x = Dense(hp.Int(f'mlp_units_{i}', 32, 128, 32), activation='relu')(x)
                x = Dropout(hp.Float(f'mlp_drop_{i}', 0.1, 0.5, 0.1))(x)
        elif arch == 'LSTM':
            x = LSTM(hp.Int('lstm_units', 32, 128, 32))(inp_seq)
            x = Dropout(hp.Float('lstm_drop', 0.1, 0.5, 0.1))(x)
        elif arch == 'StackedLSTM':
            x = inp_seq
            layers = hp.Int('stacked_layers', 2, 3)
            for i in range(layers):
                return_seq = i < layers - 1
                x = LSTM(
                    hp.Int(f'stack_units_{i}', 32, 128, 32),
                    return_sequences=return_seq
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

        # Exogenous branch
        y_ex = inp_ex
        for i in range(hp.Int('ex_layers', 1, 2)):
            y_ex = Dense(hp.Int(f'ex_units_{i}', 16, 64, 16), activation='relu')(y_ex)
            y_ex = Dropout(hp.Float(f'ex_drop_{i}', 0.1, 0.5, 0.1))(y_ex)

        # Merge and compile
        merged = concatenate([x, y_ex])
        out = Dense(1, name='output')(merged)
        model = Model([inp_seq, inp_ex], out)
        model.compile(
            optimizer=optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
            loss=smape_loss,
            metrics=[smape_loss]
        )
        return model

class NeuralForecaster:
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

    def fit(self, df: pd.DataFrame, target_col: str):
        """

        """        
        scaler_y = RobustScaler()
        y = scaler_y.fit_transform(df[[target_col]]).flatten()
        scaler_x = RobustScaler()
        exog = scaler_x.fit_transform(df.drop(columns=[target_col]))

        # Prepare sequences
        X_seq, X_ex, y_seq = [], [], []
        for i in range(len(y) - self.window_size):
            X_seq.append(y[i : i + self.window_size].reshape(self.window_size, 1))
            X_ex.append(exog[i + self.window_size])
            y_seq.append(y[i + self.window_size])
        X_seq, X_ex, y_seq = map(np.array, (X_seq, X_ex, y_seq))

        # Fixed train/val/test split sizes
        N = len(y_seq)
        tn = self.test_size
        rem = N - tn
        if rem <= 0:
            raise ValueError(
                f"Not enough data for test split: total={N}, test={tn}"
            )
        # Use 20% of remaining for validation
        vn = int(0.2 * rem)
        trn = rem - vn
        if trn <= 0:
            raise ValueError(
                f"Not enough data for train/val splits: remaining after test={rem}, val={vn}"
            )

        Xt, Xe, yt = X_seq[:trn], X_ex[:trn], y_seq[:trn]
        Xv, Xev, yv = X_seq[trn : trn + vn], X_ex[trn : trn + vn], y_seq[trn : trn + vn]
        Xte, Xe_te, yte = X_seq[-tn :], X_ex[-tn :], y_seq[-tn :]

        # Hyperparameter tuning per architecture
        for arch in self.architectures:
            project_name = f'tuner_{arch}'
            dirpath = 'keras_tuner_results'
            arch_path = os.path.join(dirpath, project_name)
            if os.path.exists(arch_path):
                shutil.rmtree(arch_path)

            tuner = kt.RandomSearch(
                CryptoHyperModel(self.window_size, X_ex.shape[1], fixed_arch=arch),
                objective='val_loss',
                max_trials=self.max_trials,
                seed=self.seed,
                directory=dirpath,
                project_name=project_name
            )

            es = callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True)
            tuner.search(
                [Xt, Xe], yt,
                validation_data=([Xv, Xev], yv),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[es]
            )

            model = tuner.get_best_models(num_models=1)[0]
            self.models_[arch] = model

            pred = model.predict([Xte, Xe_te]).flatten()
            pred_inv = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()
            true_inv = scaler_y.inverse_transform(yte.reshape(-1, 1)).flatten()

            # Pass y_train from training portion if needed for MASE
            y_train_inv = scaler_y.inverse_transform(y[:trn + self.window_size].reshape(-1,1)).flatten()
            metrics = compute_metrics(y_true=true_inv, y_pred=pred_inv, y_train=y_train_inv)
            self.results_[arch] = ForecastResult(
                {'test': metrics},
                pd.Series(pred_inv, index=df.index[-tn :]),
                model
            )

        self.scaler_y = scaler_y
        return self

    def predict(self):
        return self.results_

# ==================== ARIMA Forecaster ====================================
class ARIMAForecaster(ForecastBase):
    """
    :param order: ARIMA(p,d,q)
    :param seasonal_order: SARIMA seasonal tuple
    :param test_size: optional test size
    """
    def __init__(self,
                 order=(1,0,0),
                 seasonal_order=(0,0,0,0),
                 test_size: Optional[int]=None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.test_size = test_size
        # сохраняем скейлер для повторного использования
        self.scaler = RobustScaler()
        self.model_: Optional[Any] = None
        self.result_: Optional[ForecastResult] = None

    def fit(self, df: pd.DataFrame, target_col: str) -> 'ARIMAForecaster':
        self.log("Starting ARIMA training")
        # масштабируем y
        y_all = df[[target_col]].values
        y_scaled = self.scaler.fit_transform(y_all).flatten()

        if self.test_size:
            y_train = y_scaled[:-self.test_size]
            y_test  = y_scaled[-self.test_size:]
            index_test = df.index[-self.test_size:]
        else:
            y_train = y_scaled
            y_test  = None
            index_test = None

        # строим и обучаем SARIMAX на масштабированных данных
        self.model_ = sm.tsa.SARIMAX(
            y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        self.log("Finished ARIMA training")

        # если есть отложенная часть — делаем предсказание и обратно инвертируем
        if y_test is not None:
            start, end = 0, len(y_test)-1
            y_pred_scaled = self.model_.predict(start=start, end=end)
            y_pred = self.scaler.inverse_transform(
                y_pred_scaled.reshape(-1,1)
            ).flatten()

            metrics = {
                'test': compute_metrics(
                    y_train=y_train,
                    y_true=y_test,
                    y_pred=y_pred_scaled
                )
            }
            self.result_ = ForecastResult(
                metrics,
                pd.Series(y_pred, index=index_test),
                self.model_
            )
        return self

    def predict(self) -> ForecastResult:
        # если уже есть результат (с инвертированными предсказаниями) — возвращаем его
        if self.result_ is not None:
            return self.result_

        # иначе предсказываем на всём доступном диапазоне и обратно инвертируем
        # (можно настроить start/end по-другому)
        y_pred_scaled = self.model_.predict()
        y_pred = self.scaler.inverse_transform(
            y_pred_scaled.reshape(-1,1)
        ).flatten()

        return ForecastResult({}, pd.Series(y_pred), self.model_)


# """
# Библиотека для прогнозирования временных рядов BTC:
# - LinearForecaster: линейные модели (Ridge, Lasso, ElasticNet)
# - StackingForecaster: деревья + стекинг
# - NeuralForecaster: нейронные сети (MLP, LSTM, GRU, CNN-LSTM)
# - ARIMAForecaster: ARIMA/SARIMAX
# Общие вспомогательные функции: feature engineering, метрики, визуализация.
# """
# import pandas as pd
# import numpy as np

# from datetime import datetime

# import matplotlib.pyplot as plt

# from sklearn import preprocessing
# from sklearn.pipeline import Pipeline
# from sklearn.base import BaseEstimator
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
# from sklearn.linear_model import Ridge, Lasso, ElasticNet
# from sklearn.preprocessing import StandardScaler, RobustScaler
# from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer

# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
# from sklearn.ensemble import StackingRegressor, RandomForestRegressor, ExtraTreesRegressor

# import statsmodels.api as sm

# from tensorflow.keras import Input, Model, callbacks, optimizers
# from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, concatenate
# import kerastuner as kt

# import warnings
# warnings.filterwarnings('ignore')

# # ==================== Вспомогательные функции ====================

# def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     """
    
#     """
#     num = np.abs(y_true - y_pred)
#     den = (np.abs(y_true) + np.abs(y_pred)) / 2

#     return float(np.mean(np.where(den != 0, num/den, 0)))
# smape_scorer = make_scorer(smape, greater_is_better=False)


# def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
#     """
    
#     """
#     scale = np.mean(np.abs(np.diff(y_train)))
    
#     return float(np.mean(np.abs(y_true - y_pred)) / scale)

# # Визуализация прогнозов

# def plot_forecast(y_true: pd.Series, preds: dict, title: str):
#     """
    
#     """
#     plt.figure(figsize=(12,6))
#     plt.plot(y_true, label='Actual', linewidth=2)
    
#     for name, y_pred in preds.items():
#         plt.plot(y_pred, label=name, alpha=0.8)
    
#     plt.title(title)
#     plt.xlabel('Date')
#     plt.ylabel(y_true.name)
    
#     plt.grid(True)
#     plt.tight_layout()
#     plt.legend()
    
#     plt.show()

# # ==================== Базовый класс с логированием ====================
# class ForecastBase:
#     def log(self, message: str) -> None:
#         """
#             Выводит сообщение в консоль с временной меткой.
#         """
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#         print(f"\n\033[94m[{timestamp}]\033[0m {message}")

# # ==================== Классы ====================

# class LinearForecaster(ForecastBase, BaseEstimator):
#     def __init__(self, test_size:int=52, cv_splits:int=5, model_type:str='Ridge'):
#         """
#             класс для предсказания линейных моделей

#             из доступных для обучения моделей представлены `Ridge`, `Lasso`, `ElasticNet`
#             подбор гиперпараметров производится с помощью поиска по сетке (GridSearch)

#             :param test_size - размер теста, кол-во отобранных с конца наблюдений для теста
#             :param cv_splits - кол-во батчей на кросс-валидации (используется TimeSeriesSplit)
#             :param model_type - модель, что необходимо обучить, доступны три опции - `Ridge`, `Lasso`, `ElasticNet`. в случая указания иной модели будет обучена дефолтная модель (`Ridge`)
#         """
#         self.test_size = test_size
#         self.cv_splits = cv_splits
#         self.model_type = model_type
#         self.model_ = None

#     def fit(self, X_train: np.array, y_train: np.array):
#         """
#             обучение линейной модели
#         """    
#         self.log(f"Начало обучения {self.model_type}-регрессии")

#         self.X_train, self.y_train = X_train, y_train

#         numeric = X_train.columns.tolist()
        
#         pre = ColumnTransformer([
#                 ('num', Pipeline([
#                     ('imp', SimpleImputer()),
#                     ('sc', StandardScaler())
#                 ]), numeric)]
#             )
        
#         model_map = {
#             'Ridge':        Ridge(max_iter=10**3), 
#             'Lasso':        Lasso(max_iter=10**3), 
#             'ElasticNet':   ElasticNet(max_iter=10**3)
#         }
#         reg = model_map.get(self.model_type, Ridge())
#         pipe = Pipeline([
#                     ('pre', pre), 
#                     ('reg', reg)
#                 ])

#         param_grid = {
#             'ElasticNet': {
#                 'reg__alpha': np.logspace(-4,4,20),
#                 'reg__l1_ratio': np.linspace(0,1,20)
#             }
#         }
#         cv = TimeSeriesSplit(n_splits=self.cv_splits)

#         gs = GridSearchCV(
#                 estimator=pipe, 
#                 param_grid=param_grid.get(self.model_type, {'reg__alpha': np.logspace(-4,4,20)}), 
#                 cv=cv, 
#                 scoring=smape_scorer, 
#                 n_jobs=-1,
#                 verbose=0
#             )
#         gs.fit(X_train, y_train)
        
#         self.model_ = gs
#         self.log(f"Конец обучения {self.model_type}-регрессии")
        
#         return self

#     def predict(self, X_test: np.array, y_test: np.array):
#         """

#         """

#         preds = pd.Series(self.model_.predict(X_test), index=X_test.index)
        
#         y_pred_train = self.model_.predict(self.X_train)
#         y_pred_test = self.model_.predict(X_test)

#         p = self.X_train.shape[1]
#         n_train, n_test = len(self.y_train), len(y_test)

#         # Метрики
#         rmse_tr = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
#         rmse_te = np.sqrt(mean_squared_error(y_test, y_pred_test))
#         smape_tr = smape(self.y_train.values, y_pred_train)
#         smape_te = smape(y_test.values, y_pred_test)
#         mase_te = mase(y_test.values, y_pred_test, self.y_train.values)

#         # R2 и скорректированный R2
#         r2_tr = self.model_.score(self.X_train, self.y_train)
#         r2_te = self.model_.score(X_test, y_test)
#         adj_r2_tr = 1 - (1 - r2_tr) * (n_train - 1) / (n_train - p - 1)
#         adj_r2_te = 1 - (1 - r2_te) * (n_test - 1) / (n_test - p - 1)

#         results = pd.Series(data={
#             'model': self.model_type,
#             'best_params': self.model_.best_params_,
#             'rmse_train': rmse_tr,
#             'adj_r2_train': adj_r2_tr,
#             'smape_train': smape_tr,
#             'rmse_test': rmse_te,
#             'adj_r2_test': adj_r2_te,
#             'smape_test': smape_te,
#             'mase_test': mase_te
#         }, name=self.model_type)

#         return results, preds, y_test


# class StackingForecaster(ForecastBase, BaseEstimator):
#     def __init__(
#             self, 
#             test_size:int=260, 
#             cv_splits:int=5, 
#             n_iter:int=20,
#             seed:int=42
#         ):
#         """
#             класс для предсказания ML-моделей

#             за одну итерацию  

#             :param test_size - размер теста, кол-во отобранных с конца наблюдений для теста
#             :param cv_splits - кол-во батчей на кросс-валидации (используется TimeSeriesSplit)
#             :param n_iter - кол-во итераций при RandomSearch
#         """
#         self.test_size = test_size
#         self.cv_splits = cv_splits
#         self.n_iter = n_iter
#         self.models_ = {}
#         self.seed = seed

#     def fit(self, X_train: np.array, y_train: np.array):
#         """
        
#         """

#         df = df.copy()

#         numeric = X_train.columns.tolist()
#         pre = ColumnTransformer([
#             ('num', Pipeline([
#                 ('imp', SimpleImputer()),
#                 ('sc', RobustScaler())]), 
#             numeric)
#         ])
#         # Конфиг стекингов
#         stacks_cfg = {
#             'Stack_RF_ET': {
#                 'estimators': [
#                     ('rf', RandomForestRegressor(random_state=self.seed, n_jobs=-1)),
#                     ('et', ExtraTreesRegressor(random_state=self.seed, n_jobs=-1))
#                 ],
#                 'params': {
#                     'reg__est__rf__n_estimators': [100, 200, 300, 400, 500],
#                     'reg__est__rf__max_depth': [None, 10, 20, 30, 40],
                    
#                     'reg__est__et__n_estimators': range(100, 601, 100),
#                     'reg__est__et__max_depth': [None, 10, 20, 30, 40],
#                 }
#             },
#             'Stack_Cat_ET': {
#                 'estimators': [
#                     ('cat', CatBoostRegressor(random_seed=self.seed, verbose=0)),
#                     ('et', ExtraTreesRegressor(random_state=self.seed, n_jobs=-1))
#                 ],
#                 'params': {
#                     'reg__est__cat__iterations': [100, 200, 300, 400, 500],
#                     'reg__est__cat__learning_rate': [0.01, 0.05, 0.08, 0.09, 0.1],
#                     'reg__est__cat__depth': range(3, 11),
                    
#                     'reg__est__et__n_estimators': range(100, 601, 100),
#                     'reg__est__et__max_depth': [None, 10, 20, 30, 40],
#                 }
#             },
#             'Stack_XGB_ET': {
#                 'estimators': [
#                     ('xgb', XGBRegressor(random_state=self.seed, verbosity=0)),
#                     ('et', ExtraTreesRegressor(random_state=self.seed, n_jobs=-1))
#                 ],
#                 'params': {
#                     'reg__est__xgb__n_estimators': [100, 200, 300, 400, 500],
#                     'reg__est__xgb__learning_rate': [0.01, 0.05, 0.08, 0.09, 0.1],
#                     'reg__est__xgb__max_depth': range(1, 11),
                    
#                     'reg__est__et__n_estimators': [100, 300],
#                     'reg__est__et__max_depth': [None, 10]
#                 }
#             }
#         }

#         tscv = TimeSeriesSplit(n_splits=self.cv_splits)
#         records = []
#         preds = {}
#         importances = {}
#         naive_scale = np.mean(np.abs(np.diff(y_train)))

#         for name, cfg in stacks_cfg.items():

#             self.log(f"Начало обучения {self.model_type}-регрессии")

#             stack = StackingRegressor(
#                 estimators=cfg['estimators'],
#                 final_estimator=ElasticNet(random_state=self.seed),
#                 passthrough=True
#             )
#             # Pipeline
#             pipe = Pipeline([
#                 ('pre', pre),
#                 ('reg', TransformedTargetRegressor(
#                     regressor=Pipeline([('stack', stack)]),
#                     transformer=RobustScaler()
#                 ))
#             ])
#             # Random search
#             param_dist = {}
#             for p, vals in cfg['params'].items():
#                 # Переход к пути: regressor__regressor__stack__...
#                 param_dist[p.replace('reg__est', 'regressor__regressor__stack')] = vals

#             rs = RandomizedSearchCV(
#                 pipe,
#                 param_dist,
#                 n_iter=self.n_iter,
#                 cv=tscv,
#                 scoring=smape_scorer,
#                 random_state=self.seed,
#                 n_jobs=-1
#             )
            
#             rs.fit(X_train, y_train)
#             self.models_['stack'] = rs

#             self.log(f"Конец обучения {self.model_type}-регрессии")

#         return self

#     def predict(self, X_test: pd.DataFrame):
#         """
        
#         """
#         preds = pd.Series(self.models_['stack'].predict(X_test), index=X_test.index)
        
#         return preds

# # ==================== Модуль нейросетей ====================

# def prepare_nn_data(
#     df: pd.DataFrame,
#     target_col: str,
#     window_size: int = 10,
# ):
#     """
    
#     """
#     df_feat = df.copy()
#     scaler_y, scaler_x = RobustScaler(), RobustScaler()

#     y_scaled = scaler_y.fit_transform(df_feat[[target_col]]).flatten()
    
#     exog_cols = df_feat.columns[df_feat.columns != target_col]
#     X_exog = scaler_x.fit_transform(df_feat[exog_cols])
    
#     X_seq, X_ex, y = [], [], []

#     for i in range(len(df_feat) - window_size):
#         X_seq.append(y_scaled[i:i+window_size].reshape(window_size,1))
#         X_ex.append(X_exog[i+window_size])
#         y.append(y_scaled[i+window_size])

#     X_seq, X_ex = np.array(X_seq), np.array(X_ex)

#     y = np.array(y)
#     split1, split2 = int(len(y)*0.6), int(len(y)*0.8)

#     X_trT, X_valT, X_testT = X_seq[:split1], X_seq[split1:split2], X_seq[split2:]
#     X_trE, X_valE, X_testE = X_ex[:split1], X_ex[split1:split2], X_ex[split2:]
#     y_tr, y_val, y_test = y[:split1], y[split1:split2], y[split2:]
    
#     return (X_trT, X_trE, y_tr), (X_valT, X_valE, y_val), (X_testT, X_testE, y_test), scaler_y


# class CryptoHyperModel(kt.HyperModel):
#     def __init__(self, window_size, exog_dim):
#         """
        
#         """
#         self.window_size = window_size
#         self.exog_dim = exog_dim

#     def build(self, hp):
#         """

#         """
#         arch = hp.Choice('architecture', ['MLP','LSTM','StackedLSTM','CNN_LSTM','GRU'])
#         inp_seq = Input(shape=(self.window_size,1), name='seq_input')
#         inp_ex = Input(shape=(self.exog_dim,), name='exog_input')
        
#         if arch == 'MLP':
#             x = Flatten()(inp_seq)
            
#             for i in range(hp.Int('mlp_layers', 1, 3)):
#                 x = Dense(hp.Int(f'mlp_units_{i}', 32, 128, step=32), activation='relu')(x)
#                 x = Dropout(hp.Float(f'mlp_drop_{i}', 0.1, 0.5, step=0.1))(x)

#         elif arch == 'LSTM':
#             x = LSTM(hp.Int('lstm_units', 32, 128, step=32))(inp_seq)
#             x = Dropout(hp.Float('lstm_drop', 0.1, 0.5, step=0.1))(x)

#         elif arch == 'StackedLSTM':
#             x = inp_seq

#             for i in range(hp.Int('stacked_layers', 2, 3)):
#                 return_seq = i < hp.get('stacked_layers')-1
#                 x = LSTM(hp.Int(f'stack_units_{i}', 32, 128, step=32), return_sequences=return_seq)(x)
#                 x = Dropout(hp.Float(f'stack_drop_{i}', 0.1, 0.5, step=0.1))(x)

#         elif arch == 'CNN_LSTM':
#             x = Conv1D(filters=hp.Int('cnn_filters',16,64,step=16), kernel_size=hp.Int('cnn_kernel',2,5), activation='relu')(inp_seq)
#             x = MaxPooling1D(pool_size=hp.Int('pool_size',2,4))(x)
#             x = Dropout(hp.Float('cnn_drop',0.1,0.5,step=0.1))(x)
#             x = LSTM(hp.Int('cnn_lstm_units',32,128,step=32))(x)

#         else:
#             x = GRU(hp.Int('gru_units',32,128,step=32))(inp_seq)
#             x = Dropout(hp.Float('gru_drop',0.1,0.5,step=0.1))(x)

#         y_ex = inp_ex
        
#         for i in range(hp.Int('ex_layers',1,2)):
#             y_ex = Dense(hp.Int(f'ex_units_{i}',16,64,step=16), activation='relu')(y_ex)
#             y_ex = Dropout(hp.Float(f'ex_drop_{i}',0.1,0.5,step=0.1))(y_ex)

#         concat = concatenate([x,y_ex])
#         out = Dense(1, name='output')(concat)
#         model = Model([inp_seq, inp_ex], out)
#         model.compile(
#             optimizer=optimizers.Adam(hp.Float('learning_rate',1e-4,1e-2,sampling='log')),
#             loss='mean_absolute_error'
#         )

#         return model

# class NeuralForecaster(ForecastBase):
#     def __init__(
#             self, 
#             window_size:int=10, 
#             test_size:int=260, 
#             max_trials:int=10, 
#             epochs:int=50, 
#             batch_size:int=32, 
#             seed:int=42
#         ):
#         """

#         """
#         self.window_size = window_size
#         self.test_size = test_size
#         self.max_trials = max_trials
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.seed = seed
#         self.model_ = None
#         self.scaler_ = None

#     def fit(self, df: pd.DataFrame, date_col:str, target_col:str='BTC'):
#         """

#         """
        
#         self.log(f"Начало обучения нейронной сети")

#         (X_trT, X_trE, y_tr), (X_valT, X_valE, y_val), (_,_,_), scaler = prepare_nn_data(
#             df, date_col, target_col, self.window_size, self.test_size
#         )

#         exog_dim = X_trE.shape[1]
#         tuner = kt.RandomSearch(
#             CryptoHyperModel(self.window_size, exog_dim),
#             objective='val_loss',
#             max_trials=self.max_trials,
#             seed=self.seed,
#             directory='tuner_dir'
#         )
#         es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#         tuner.search(
#             [X_trT, X_trE], y_tr,
#             validation_data=([X_valT, X_valE], y_val),
#             epochs=self.epochs,
#             batch_size=self.batch_size,
#             callbacks=[es]
#         )
#         best_model = tuner.get_best_models(num_models=1)[0]
#         self.model_ = best_model
#         self.scaler_ = scaler
        
#         self.log(f"Конец обучения нейронной сети")

#         return self

#     def predict(self, df: pd.DataFrame):
#         """
        
#         """
#         df = df.copy()
#         (_,_,_), (_,_,_), (X_testT, X_testE, y_test), _ = prepare_nn_data(
#             df, df.columns[0], df.columns[-1], self.window_size, self.test_size
#         )
#         pred_scaled = self.model_.predict([X_testT, X_testE])
        
#         pred = self.scaler_.inverse_transform(pred_scaled).flatten()
#         preds = pd.Series(pred, index=df.index[-self.test_size:])
        
#         return preds

# class ARIMAForecaster(ForecastBase):
#     def __init__(self, order=(1,0,0), seasonal_order=(0,0,0,0)):
#         """
        
#         """
#         self.order = order
#         self.seasonal_order = seasonal_order
#         self.model_ = None

#     def fit(self, df: pd.DataFrame, target_col:str='BTC'):
#         """
        
#         """
#         self.log(f"Начало обучения ARIMA-модели")

#         df = df.copy()

#         self.model_ = sm.tsa.statespace.SARIMAX(
#             df[target_col], order=self.order, seasonal_order=self.seasonal_order,
#             enforce_stationarity=False, enforce_invertibility=False
#         ).fit(disp=False)
        
#         self.log(f"Конец обучения ARIMA-модели")

#         return self

#     def predict(self, start=None, end=None):
#         """

#         """
#         return self.model_.predict(start=start, end=end)


