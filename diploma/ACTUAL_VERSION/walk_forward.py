"""
Walk-forward validation for BTC forecasting pipeline.

Two split strategies (chosen by frequency):
- `expanding_window_splits` - Daily: growing train set, fixed-length test blocks.
- `rolling_window_splits`   - Hourly/5-min: fixed-width sliding train window.

Window sizes are calibrated so that with per-fold Optuna retuning the total
wall-clock time remains tractable:
  daily  : 5 folds, ~500 rows minimum train / ~43 rows test (~2 months per fold).
  hourly : 5 folds, ~3 000 rows train (~125 days) / ~600 rows test (~25 days).
  5min   : 5 folds, ~2 400 rows train (~8 days) / ~600 rows test (~2 days).

`run_walk_forward` orchestrates the full per-fold pipeline:
  1. Slice train / test rows by integer position.
  2. Select top-k |Pearson corr| features on train fold only (leakage-free).
  3. Each model handles its own preprocessing internally:
       GB  -> sklearn Pipeline (imputer + QuantileClipper + RobustScaler).
       NN  -> CryptoNetRegressor.fit() (RobustScaler on train-before-val portion).
  4. Run Optuna to retune hyperparameters (CV inside train fold, no test leakage).
  5. Fit final model on full train fold; predict test fold.
  6. Collect compute_metrics + raw OOF predictions.

Returns (metrics_df, oof_df) - metrics_df is per-fold; oof_df is
out-of-fold predictions concatenated across folds for economic simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

from validation import PurgedKFold, cpcv_splits
from metrics import rank_ic, net_cost_sharpe, mwda, mcc, balanced_accuracy
from ml_models import (
    compute_metrics,
    run_optuna_study,
    CryptoNetRegressor,
    CryptoNetClassifier,
    _sample_crypto_hparams,
    set_global_seed,
    DEVICE,
)


# ---------------------------------------------------------------------------
# Split generators
# ---------------------------------------------------------------------------

def expanding_window_splits(
    n: int,
    n_splits: int = 5,
    min_train_frac: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window folds for Daily data. Train set grows each fold; test block
    is the next fixed-size chunk. `min_train_frac` sets how much of `n` the
    first train set covers, leaving `(1 - min_train_frac) * n` rows for the
    `n_splits` test blocks. Each test block is therefore
    `floor((1 - min_train_frac) * n / n_splits)` rows.

    Returns list of (train_idx, test_idx) integer-position arrays.
    """
    min_train = int(np.floor(min_train_frac * n))
    remaining = n - min_train
    test_size = remaining // n_splits

    if test_size < 1:
        raise ValueError(
            f'Not enough rows for {n_splits} folds with min_train_frac={min_train_frac}: '
            f'n={n}, remaining={remaining}'
        )

    splits = []
    train_end = min_train
    for _ in range(n_splits):
        test_end = min(train_end + test_size, n)
        splits.append((np.arange(train_end), np.arange(train_end, test_end)))
        train_end = test_end
        if train_end >= n:
            break
    return splits


def rolling_window_splits(
    n: int,
    train_size: int,
    test_size: int,
    step: Optional[int] = None,
    n_splits: Optional[int] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Rolling-window folds. Train window is fixed-width (`train_size` rows) and
    slides forward by `step` rows each fold. If `step` is None, it defaults to
    `test_size` (non-overlapping test blocks). If `n_splits` is None, generate
    AS MANY folds as fit (roll to the end of the series) — used for the short-
    window / frequent-refit scheme; combine with `retune_every` so per-fold
    Optuna does not blow up the wall-clock.

    Returns list of (train_idx, test_idx) integer-position arrays.
    """
    if step is None:
        step = test_size

    splits = []
    start = 0
    while n_splits is None or len(splits) < n_splits:
        train_end = start + train_size
        test_end = train_end + test_size
        if test_end > n:
            break
        splits.append((np.arange(start, train_end), np.arange(train_end, test_end)))
        start += step

    if not splits:
        raise ValueError(
            f'No valid folds: n={n}, train_size={train_size}, test_size={test_size}'
        )
    return splits


def forward_log_return(price: pd.Series, horizon: int = 1) -> pd.Series:
    """Forward cumulative log-return target over `horizon` bars (T6.3).

    Метка в строке t — сумма СЛЕДУЮЩИХ `horizon` лог-доходностей
    ``R_t^{(h)} = log(P_{t+h-1}) − log(P_{t-1}) = r_t + … + r_{t+h-1}``, где
    ``r_s = log(P_s) − log(P_{s-1})``. Это будущая (forward) кумулятивная
    доходность, поэтому она НЕ пересекается с лагированными признаками (≤ r_{t-1})
    — leakage-safe. При ``horizon == 1`` тождественно равно ``log(P).diff()`` (как в
    исходном одношаговом таргете — байт-в-байт). Последние ``horizon-1`` строк -> NaN
    (нет будущего) и отбрасываются вызывающим кодом.
    """
    if horizon < 1:
        raise ValueError(f'horizon must be >= 1, got {horizon}')
    r = np.log(price.astype(float)).replace([np.inf, -np.inf], np.nan).diff()
    if horizon == 1:
        return r
    return r.rolling(horizon).sum().shift(-(horizon - 1))


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic time-of-day / calendar cyclical features (EPIC 7, T7.1).

    Кодирует время прогнозируемого бара через sin/cos: minute-of-day (период 1440),
    hour (24), day-of-week (7), month (12). Признаки ДЕТЕРМИНИРОВАНЫ по timestamp
    прогноза -> известны заранее, поэтому НЕ лагируются (утечки нет); подаются в
    exog-ветку. Требует DatetimeIndex. Возвращает копию с колонками
    tod_sin/tod_cos, hour_sin/hour_cos, dow_sin/dow_cos, month_sin/month_cos.
    """
    idx = pd.DatetimeIndex(df.index)
    out = df.copy()
    two_pi = 2.0 * np.pi
    mod = idx.hour * 60 + idx.minute
    out['tod_sin'] = np.sin(two_pi * mod / 1440.0)
    out['tod_cos'] = np.cos(two_pi * mod / 1440.0)
    out['hour_sin'] = np.sin(two_pi * idx.hour / 24.0)
    out['hour_cos'] = np.cos(two_pi * idx.hour / 24.0)
    dow = idx.dayofweek
    out['dow_sin'] = np.sin(two_pi * dow / 7.0)
    out['dow_cos'] = np.cos(two_pi * dow / 7.0)
    out['month_sin'] = np.sin(two_pi * idx.month / 12.0)
    out['month_cos'] = np.cos(two_pi * idx.month / 12.0)
    return out


CYCLICAL_FEATURES = ('tod_sin', 'tod_cos', 'hour_sin', 'hour_cos',
                     'dow_sin', 'dow_cos', 'month_sin', 'month_cos')


# ---------------------------------------------------------------------------
# Leakage-safe feature selection and preprocessing utilities
# ---------------------------------------------------------------------------

class QuantileClipper:
    """1st/99th percentile clipper — fit on train fold only."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X: np.ndarray) -> 'QuantileClipper':
        self.lower_bounds_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


def select_top_k_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int,
    method: str = 'pearson',
    random_state: int = 0,
) -> list[str]:
    """
    Top-k feature selection on the TRAIN fold only (leakage-free, T7.2).

    `method`:
      * 'pearson'     — |Pearson corr| с таргетом (по умолчанию; линейная связь).
      * 'mutual_info' — `sklearn.mutual_info_regression` (ловит нелинейные связи).
      * 'model_gain'  — impurity/gain-важность tree-ансамбля (sklearn RandomForest,
                        модельная, in-sample). Намеренно sklearn, а не LightGBM: на
                        macOS LightGBM-libomp конфликтует с torch-libomp в одном
                        процессе (сегфолт); отбор признаков от этого свободен.
      * 'mda'         — Mean Decrease Accuracy: permutation-важность на OUT-OF-SAMPLE
                        хвосте train-фолда (López de Prado; модель-агностична).

    Колонки, полностью состоящие из NaN, исключаются. (CFI-кластеризация
    коррелированных фич и SHAP — расширения P2/P3.)
    """
    cols = [c for c in X_train.columns if X_train[c].notna().any()]
    X = X_train[cols]

    if method == 'pearson':
        score = X.corrwith(y_train).abs()
    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression
        Xi = X.fillna(X.median())
        mi = mutual_info_regression(Xi.to_numpy(), y_train.to_numpy(dtype=float),
                                    random_state=random_state)
        score = pd.Series(mi, index=cols)
    elif method == 'model_gain':
        from sklearn.ensemble import RandomForestRegressor
        Xi = X.fillna(X.median())
        m = RandomForestRegressor(n_estimators=200, random_state=random_state,
                                  n_jobs=-1).fit(Xi, y_train)
        score = pd.Series(m.feature_importances_, index=cols)
    elif method == 'mda':
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.inspection import permutation_importance
        Xi = X.fillna(X.median())
        cut = max(1, int(len(Xi) * 0.7))
        m = HistGradientBoostingRegressor(max_iter=120, random_state=random_state)
        m.fit(Xi.iloc[:cut], y_train.iloc[:cut])
        r = permutation_importance(m, Xi.iloc[cut:], y_train.iloc[cut:],
                                   n_repeats=5, random_state=random_state)
        score = pd.Series(r.importances_mean, index=cols)
    else:
        raise ValueError(f"unknown selection method: {method!r}")

    return list(score.dropna().nlargest(k).index)


def select_best_by_composite(
    agg: pd.DataFrame,
    names,
    by: tuple = ('IC_IR', 'NetSharpe', 'MWDA'),
    tie_break: str = 'MASE',
) -> pd.DataFrame:
    """Composite model selection (EPIC 3, T3.1) — один победитель на частоту.

    Ранжирует по кортежу `by` (по умолчанию IC IR -> net-of-cost Sharpe -> MWDA,
    все по убыванию), tie-break `tie_break` по возрастанию (MASE). Заменяет выбор
    по «голой» Directional Accuracy: DA игнорирует магнитуду движения и сравнивается
    с неверным нулём 0.5 вместо base-rate.

    `agg` — агрегированная по моделям таблица (одна строка на (frequency, model))
    с колонками 'frequency', 'model', метриками `by` и `tie_break`. `names` —
    подмножество моделей-кандидатов (например, только GB или только NN).
    Возвращает по одной строке-победителю на каждую частоту.
    """
    sub = agg[agg['model'].isin(set(names))].copy()
    if sub.empty:
        return sub
    sort_cols = ['frequency', *by, tie_break]
    ascending = [True, *([False] * len(by)), True]
    sub = sub.sort_values(sort_cols, ascending=ascending)
    return sub.groupby('frequency', as_index=False).head(1).reset_index(drop=True)


def preprocess_fold(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit median-imputer + 1st/99th clipper + RobustScaler on `X_train`;
    apply all three fitted transformers to both splits.

    Exposed as a standalone utility for external use (e.g. when a model does
    not include its own preprocessing pipeline). The walk-forward loop itself
    does NOT call this function: GB models carry sklearn Pipelines with
    these steps, and CryptoNetRegressor fits its own RobustScaler internally.
    """
    cols = X_train.columns

    imputer = SimpleImputer(strategy='median').fit(X_train)
    Xtr = imputer.transform(X_train)
    Xte = imputer.transform(X_test)

    clipper = QuantileClipper().fit(Xtr)
    Xtr = clipper.transform(Xtr)
    Xte = clipper.transform(Xte)

    scaler = RobustScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)

    return (
        pd.DataFrame(Xtr, columns=cols, index=X_train.index),
        pd.DataFrame(Xte, columns=cols, index=X_test.index),
    )


# ---------------------------------------------------------------------------
# Optuna objective factories
# ---------------------------------------------------------------------------

# Early-stopping patience for NN training (epochs without val-MAE improvement).
# Поднят с 5 до 8: на доходностях валидационная кривая шумнее, ранний останов по
# 5 эпохам обрывал обучение в шумовом минимуме.
NN_PATIENCE = 8

# Псевдонимы имён Optuna-параметров -> атрибутов эстиматора. Структурные параметры
# совпадают по имени между Optuna и бустингом; только learning_rate сокращён до 'lr',
# а 'n_est' распределяется на первый из n_estimators/iterations/max_iter.
_GB_PARAM_ALIAS = {'lr': 'learning_rate'}
_GB_N_EST_ATTRS = ('n_estimators', 'iterations', 'max_iter')


def _apply_gb_params(model, params: dict) -> None:
    """Применить подобранные Optuna-параметры к свежему GB-Pipeline (refit).

    Пропускает параметры, которых нет у конкретного бустинга (`hasattr` по
    `named_steps['model']`), и распределяет 'n_est' на доступный счётчик деревьев.
    Единый источник правды о маппинге для objective и refit (T3.2).
    """
    step = model.named_steps['model']
    for name, value in params.items():
        if name == 'n_est':
            for attr in _GB_N_EST_ATTRS:
                if hasattr(step, attr):
                    model.set_params(**{f'model__{attr}': value})
                    break
            continue
        attr = _GB_PARAM_ALIAS.get(name, name)
        if hasattr(step, attr):
            model.set_params(**{f'model__{attr}': value})


def _gb_objective(
    model_factory: Callable,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    horizon: int = 1,
    n_splits: int = 3,
    fee: float = 0.0014,
):
    """
    Returns an Optuna objective for a GB sklearn-Pipeline factory (T3.2).

    Объектив — **net-of-cost Sharpe** в пространстве доходностей: на каждом
    PurgedKFold-тест-срезе ``side = sign(r̂)``, доходность стратегии после издержек
    оборота считается `metrics.net_cost_sharpe(side, r_true, fee)`. Возвращает
    ОТРИЦАТЕЛЬНЫЙ средний Sharpe по фолдам (вызов остаётся `direction='minimize'`).
    Цена не нужна — всё в return-space.

    Тюнится не только `lr` + счётчик деревьев, но и полный набор структурных
    гиперпараметров там, где бустинг их экспонирует (`hasattr`): max_depth / depth /
    num_leaves / min_child_samples / subsample / colsample_bytree / reg_lambda /
    l2_regularization / l2_leaf_reg. `GB_HYPERPARAMS` остаются лишь стартовыми
    дефолтами.

    CV uses `PurgedKFold` (purge перекрывающихся меток + embargo) на train-фолде
    вместо `TimeSeriesSplit` (T2.2). Embargo = max(horizon, 1% длины train-фолда).
    """
    from sklearn.base import clone

    embargo = max(horizon, int(np.ceil(0.01 * len(X_train))))
    cv = PurgedKFold(n_splits=n_splits, embargo=embargo, horizon=horizon)

    def objective(trial):
        model = model_factory()
        step = model.named_steps['model']

        if hasattr(step, 'learning_rate'):
            model.set_params(
                model__learning_rate=trial.suggest_float('lr', 1e-3, 0.3, log=True)
            )
        for attr in _GB_N_EST_ATTRS:
            if hasattr(step, attr):
                model.set_params(**{
                    f'model__{attr}': trial.suggest_int('n_est', 100, 800, step=100)
                })
                break

        # Структурные HP — подбираются только если эстиматор их поддерживает, чтобы
        # study.best_params содержал ровно применимые ключи (и refit их применил).
        if hasattr(step, 'max_depth'):
            model.set_params(model__max_depth=trial.suggest_int('max_depth', 2, 8))
        if hasattr(step, 'depth'):           # CatBoost
            model.set_params(model__depth=trial.suggest_int('depth', 2, 8))
        if hasattr(step, 'num_leaves'):      # LightGBM
            model.set_params(model__num_leaves=trial.suggest_int('num_leaves', 15, 255))
        if hasattr(step, 'min_child_samples'):
            model.set_params(model__min_child_samples=trial.suggest_int('min_child_samples', 5, 80))
        if hasattr(step, 'subsample'):
            model.set_params(model__subsample=trial.suggest_float('subsample', 0.6, 1.0))
        if hasattr(step, 'colsample_bytree'):
            model.set_params(model__colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0))
        if hasattr(step, 'reg_lambda'):      # LightGBM / XGBoost
            model.set_params(model__reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True))
        if hasattr(step, 'l2_regularization'):   # HistGradientBoosting
            model.set_params(model__l2_regularization=trial.suggest_float('l2_regularization', 1e-3, 10.0, log=True))
        if hasattr(step, 'l2_leaf_reg'):     # CatBoost
            model.set_params(model__l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1e-1, 10.0, log=True))

        sharpes = []
        for tr, te in cv.split(X_train):
            if len(tr) == 0 or len(te) == 0:
                continue
            m = clone(model)
            m.fit(X_train.iloc[tr], y_train.iloc[tr])
            pred = np.asarray(m.predict(X_train.iloc[te]), dtype=float)
            r_true = y_train.iloc[te].to_numpy(dtype=float)
            sh = net_cost_sharpe(np.sign(pred), r_true, fee=fee)
            if np.isfinite(sh):
                sharpes.append(sh)
        # Максимизируем Sharpe -> возвращаем минус среднего (direction='minimize').
        return -float(np.mean(sharpes)) if sharpes else float('inf')

    return objective


def _nn_objective(
    arch: str,
    seq_train_df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    epochs: int,
    batch_size: int,
    seed: int,
    target_col: str = 'BTC',
    optuna_val_frac: float = 0.2,
):
    """Returns an Optuna objective for one CryptoNet architecture.

    `optuna_val_frac>0` выделяет отдельный held-out срез для оценки трайла,
    разнесённый с es-val (early stopping) — выбор гиперпараметров не переобучается
    под тот же хвост, что и ранний останов (T2.4)."""
    def objective(trial):
        hp = _sample_crypto_hparams(trial, arch, window_size)
        lr = hp.pop('learning_rate')
        weight_decay = hp.pop('weight_decay', 0.0)
        reg = CryptoNetRegressor(
            arch=arch,
            target_col=target_col,
            feature_cols=feature_cols,
            window_size=window_size,
            hp=hp,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            patience=NN_PATIENCE,
            seed=seed,
            weight_decay=weight_decay,
            optuna_val_frac=optuna_val_frac,
        )
        reg.fit(seq_train_df)
        return reg.best_val_score_
    return objective


def _nn_clf_objective(
    arch: str,
    seq_train_df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    epochs: int,
    batch_size: int,
    seed: int,
    target_col: str = 'BTC',
    optuna_val_frac: float = 0.2,
):
    """Returns an Optuna objective for one CryptoNet sign-classifier architecture
    (minimises validation weighted-BCE). `optuna_val_frac>0` оценивает трайл на
    отдельном held-out срезе, разнесённом с es-val (T2.4)."""
    def objective(trial):
        hp = _sample_crypto_hparams(trial, arch, window_size)
        lr = hp.pop('learning_rate')
        weight_decay = hp.pop('weight_decay', 0.0)
        clf = CryptoNetClassifier(
            arch=arch, target_col=target_col, feature_cols=feature_cols,
            window_size=window_size, hp=hp, lr=lr, epochs=epochs,
            batch_size=batch_size, patience=NN_PATIENCE, seed=seed, weight_decay=weight_decay,
            optuna_val_frac=optuna_val_frac,
        )
        clf.fit(seq_train_df)
        return clf.best_val_score_
    return objective


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def run_walk_forward(
    model_family: str,
    model_name: str,
    df: pd.DataFrame,
    target_col: str,
    all_feature_cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    top_k: int,
    optuna_n_trials: int,
    nn_config: Optional[dict] = None,
    gb_model_factory: Optional[Callable] = None,
    seed: int = 42,
    retune_every: int = 1,
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward loop for one model (GB or NN).

    Parameters
    ----------
    model_family     : 'GB' or 'NN'
    model_name       : display name, e.g. 'LightGBM' or 'GRU (sequence NN)'
    df               : full raw per-frequency DataFrame (includes `target_col`)
    target_col       : column to forecast, 'BTC'
    all_feature_cols : candidate feature pool; top_k selected per fold
    splits           : (train_idx, test_idx) arrays from split generator functions
    top_k            : features to keep per fold by |Pearson| ranking
    optuna_n_trials  : Optuna trials per fold
    nn_config        : {'window_size', 'epochs', 'batch_size'} (NN only)
    gb_model_factory : zero-arg callable returning a fresh sklearn Pipeline (GB only)
    seed             : master RNG seed; fold i uses seed + i
    retune_every     : re-run Optuna only every `retune_every` folds, reusing the last
                       tuned hyperparameters in between (1 = tune every fold, as before).
                       For the short-window / many-fold scheme set e.g. 20 so per-fold
                       tuning stays tractable — refit is cheap, re-tuning is not.
    horizon          : forecast horizon h (число баров кумулятивной forward-метки, T6.3).
                       При h>1 из каждого train-фолда отбрасываются последние (h−1) строк
                       (embargo/purge: их forward-метки перекрывают test-блок). h=1 — без
                       изменений. Также прокидывается в PurgedKFold Optuna-objective.

    Returns
    -------
    metrics_df : per-fold metrics (fold, model, MAE, RMSE, SMAPE, MASE, DA, RankIC,
                 MWDA, MCC, BalAcc, AUC, n_train, n_test)
    oof_df     : OOF predictions (timestamp, actual, predicted, fold, model)
    """
    if model_family not in ('GB', 'NN', 'NN_CLF'):
        raise ValueError(f'model_family must be "GB", "NN" or "NN_CLF", got {model_family!r}')

    metrics_rows = []
    oof_rows = []
    cached_params = None   # last Optuna best_params; reused between retune folds

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_seed = seed + fold_idx

        # Embargo/purge для forward-метки горизонта h: последние (h−1) строк train
        # перекрываются по времени с метками test -> отбрасываем их (h=1 -> без эффекта).
        if horizon > 1 and len(train_idx) > horizon - 1:
            train_idx = train_idx[:-(horizon - 1)]

        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        y_train = df_train[target_col]
        y_test = df_test[target_col]

        # Top-k feature selection on train fold only (Pearson |corr| with target)
        selected = select_top_k_features(df_train[all_feature_cols], y_train, k=top_k)
        clf_auc = np.nan

        if model_family == 'GB':
            X_train = df_train[selected]
            X_test = df_test[selected]

            # Optuna tunes lr + tree count + full structural HP space by net-of-cost
            # Sharpe inside a PurgedKFold (purge+embargo) on the train fold (T3.2).
            # Runs only on retune folds; else reuse cached params.
            if cached_params is None or fold_idx % retune_every == 0:
                study = run_optuna_study(
                    _gb_objective(gb_model_factory, X_train, y_train, horizon=horizon),
                    direction='minimize',   # objective returns -Sharpe
                    n_trials=optuna_n_trials,
                    seed=fold_seed,
                    study_name=f'{model_name}_fold{fold_idx}',
                )
                cached_params = dict(study.best_params)

            best = gb_model_factory()
            _apply_gb_params(best, cached_params)
            best.fit(X_train, y_train)
            y_pred = pd.Series(best.predict(X_test), index=y_test.index)

        elif model_family == 'NN':  # regression NN
            window_size = nn_config['window_size']
            epochs = nn_config['epochs']
            batch_size = nn_config['batch_size']
            arch = model_name.split()[0]   # 'GRU (sequence NN)' -> 'GRU'

            seq_cols = selected + [target_col]
            seq_train_df = df_train[seq_cols]

            # Prepend `window_size` history rows before test block so the first
            # test-fold window has a complete lookback sequence.
            hist_start = max(0, int(train_idx[-1]) + 1 - window_size)
            seq_test_df = df.iloc[hist_start : int(test_idx[-1]) + 1][seq_cols]

            if cached_params is None or fold_idx % retune_every == 0:
                study = run_optuna_study(
                    _nn_objective(arch, seq_train_df, selected, window_size, epochs, batch_size, fold_seed, target_col=target_col),
                    direction='minimize',
                    n_trials=optuna_n_trials,
                    seed=fold_seed,
                    study_name=f'{model_name}_fold{fold_idx}',
                )
                cached_params = dict(study.best_params)

            best_params = dict(cached_params)
            lr = best_params.pop('learning_rate')
            weight_decay = best_params.pop('weight_decay', 0.0)
            set_global_seed(fold_seed)
            reg = CryptoNetRegressor(
                arch=arch,
                target_col=target_col,
                feature_cols=selected,
                window_size=window_size,
                hp=best_params,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                patience=NN_PATIENCE,
                seed=fold_seed,
                device=DEVICE,
                weight_decay=weight_decay,
            )
            reg.fit(seq_train_df)
            y_pred = reg.predict(seq_test_df)

        else:  # NN_CLF — взвешенная классификация знака доходности
            window_size = nn_config['window_size']
            epochs = nn_config['epochs']
            batch_size = nn_config['batch_size']
            arch = model_name.split()[0]   # 'GRU (sign clf)' -> 'GRU'

            seq_cols = selected + [target_col]
            seq_train_df = df_train[seq_cols]
            hist_start = max(0, int(train_idx[-1]) + 1 - window_size)
            seq_test_df = df.iloc[hist_start : int(test_idx[-1]) + 1][seq_cols]

            if cached_params is None or fold_idx % retune_every == 0:
                study = run_optuna_study(
                    _nn_clf_objective(arch, seq_train_df, selected, window_size, epochs, batch_size, fold_seed, target_col=target_col),
                    direction='minimize',
                    n_trials=optuna_n_trials,
                    seed=fold_seed,
                    study_name=f'{model_name}_fold{fold_idx}',
                )
                cached_params = dict(study.best_params)

            best_params = dict(cached_params)
            lr = best_params.pop('learning_rate')
            weight_decay = best_params.pop('weight_decay', 0.0)
            set_global_seed(fold_seed)
            clf = CryptoNetClassifier(
                arch=arch, target_col=target_col, feature_cols=selected,
                window_size=window_size, hp=best_params, lr=lr, epochs=epochs,
                batch_size=batch_size, patience=NN_PATIENCE, seed=fold_seed,
                device=DEVICE, weight_decay=weight_decay,
            )
            clf.fit(seq_train_df)
            proba = clf.predict_proba(seq_test_df).reindex(y_test.index)
            # Псевдо-доходность (p - 0.5): знак = направленная ставка, проходит в общий
            # OOF/бэктест без изменений (сигнал = p > 0.5). Величина для PnL не нужна —
            # бэктест считает PnL по фактической цене, прогноз нужен только для сигнала.
            y_pred = proba - 0.5
            mask = proba.notna() & y_test.notna()
            y_true_dir = (y_test[mask].to_numpy(dtype=float) > 0).astype(int)
            if y_true_dir.size > 0 and y_true_dir.min() != y_true_dir.max():
                clf_auc = float(roc_auc_score(y_true_dir, proba[mask].to_numpy(dtype=float)))

        y_pred_aligned = y_pred.reindex(y_test.index)
        yt = y_test.to_numpy(dtype=float)
        yp = y_pred_aligned.to_numpy(dtype=float)
        metrics = compute_metrics(y_true=yt, y_pred=yp, y_train=y_train.to_numpy(dtype=float))
        is_clf = (model_family == 'NN_CLF')

        # Композитные метрики селекции (EPIC 3): Rank IC и magnitude-weighted DA в
        # return-space для всех моделей; MCC / balanced accuracy — только для
        # классификатора знака (`yp` = proba−0.5 -> sign = ставка направления).
        rankic_v = rank_ic(yt, yp)
        mwda_v = mwda(yt, yp)
        mcc_v = mcc(yt, yp) if is_clf else np.nan
        balacc_v = balanced_accuracy(yt, yp) if is_clf else np.nan

        metrics_rows.append({
            'fold': fold_idx,
            'model': model_name,
            # Для классификатора метрики ВЕЛИЧИНЫ не определены (прогноз = вероятность),
            # поэтому MAE/RMSE/SMAPE/MASE = NaN; содержательны DA / AUC / MCC / BalAcc.
            'MAE': np.nan if is_clf else metrics['mae'],
            'RMSE': np.nan if is_clf else metrics['rmse'],
            'SMAPE': np.nan if is_clf else metrics['smape'],
            'MASE': np.nan if is_clf else metrics['mase'],
            'DA': metrics['da'],
            'RankIC': rankic_v,
            'MWDA': mwda_v,
            'MCC': mcc_v,
            'BalAcc': balacc_v,
            'AUC': clf_auc,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
        })
        oof_rows.append(pd.DataFrame({
            'timestamp': y_test.index,
            'actual': y_test.to_numpy(dtype=float),
            'predicted': y_pred_aligned.to_numpy(dtype=float),
            'fold': fold_idx,
            'model': model_name,
        }))

    metrics_df = pd.DataFrame(metrics_rows)
    oof_df = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    return metrics_df, oof_df


def run_cpcv_gb(
    df: pd.DataFrame,
    target_col: str,
    all_feature_cols: list[str],
    gb_model_factory: Callable,
    top_k: int,
    N: int = 6,
    k: int = 2,
    embargo_frac: float = 0.01,
    horizon: int = 1,
    fee: float = 0.0014,
) -> pd.DataFrame:
    """
    CPCV-распределение метрик для GB-модели (опциональный режим, T2.3).

    `run_walk_forward` остаётся production-simulation (один каузальный путь);
    эта функция дополняет его комбинаторно-purged CV (AFML гл. 12): для каждого из
    ``C(N, k)`` путей выполняется leakage-safe отбор top-k на train-части пути,
    свежая GB-модель фитится и прогнозирует test-часть. Возвращает per-path
    метрики (Rank IC, net-of-cost Sharpe, DA) — распределение по путям нужно для
    **DSR/PBO** (EPIC 5/8): прогон множества путей, а не одной OOS-траектории.

    Optuna здесь НЕ запускается — цель в распределении метрик при фиксированной
    конфигурации, а не в тюнинге. NN-архитектуры не поддержаны: их
    forward-history prepend несовместим с некаузальными CPCV-сплитами (train
    может идти после test) — TODO интеграции на уровне P2/P3.

    Returns
    -------
    DataFrame с колонками: path, rank_ic, net_cost_sharpe, da, n_train, n_test.
    """
    n = len(df)
    y = df[target_col]
    rows = []
    for path_idx, (train_idx, test_idx) in enumerate(
        cpcv_splits(n, N=N, k=k, embargo_frac=embargo_frac, horizon=horizon)
    ):
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        selected = select_top_k_features(df_train[all_feature_cols], df_train[target_col], k=top_k)

        model = gb_model_factory()
        model.fit(df_train[selected], df_train[target_col])
        pred = np.asarray(model.predict(df_test[selected]), dtype=float)
        y_true = df_test[target_col].to_numpy(dtype=float)

        rows.append({
            'path': path_idx,
            'rank_ic': rank_ic(y_true, pred),
            'net_cost_sharpe': net_cost_sharpe(np.sign(pred), y_true, fee=fee),
            'da': float(np.mean((y_true > 0) == (pred > 0))),
            'n_train': len(train_idx),
            'n_test': len(test_idx),
        })
    return pd.DataFrame(rows)
