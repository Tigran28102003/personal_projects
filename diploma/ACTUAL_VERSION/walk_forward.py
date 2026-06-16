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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score

from ml_models import (
    smape,
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
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Rolling-window folds for Hourly/5-min data. Train window is fixed-width
    (`train_size` rows) and slides forward by `step` rows each fold. If `step`
    is None, it defaults to `test_size` (non-overlapping test blocks).

    Returns list of (train_idx, test_idx) integer-position arrays.
    """
    if step is None:
        step = test_size

    splits = []
    start = 0
    while len(splits) < n_splits:
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
) -> list[str]:
    """
    Top-k features by |Pearson correlation| with the target, computed on the
    training fold only. Columns with all-NaN are excluded before ranking.
    """
    corrs = X_train.corrwith(y_train).abs().dropna()
    return list(corrs.nlargest(k).index)


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

def _gb_objective(model_factory: Callable, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Returns an Optuna objective for a GB sklearn-Pipeline factory.

    Tunes `learning_rate` and `n_estimators`/`iterations`/`max_iter` for the
    inner estimator; all structural hyperparameters (max_depth, num_leaves, etc.)
    stay as specified in `model_factory`. CV uses TimeSeriesSplit(n_splits=3)
    on the provided train fold — no test-set leakage by construction.
    """
    from sklearn.base import clone

    def objective(trial):
        model = model_factory()
        step = model.named_steps['model']

        if hasattr(step, 'learning_rate'):
            model.set_params(
                model__learning_rate=trial.suggest_float('lr', 0.01, 0.2, log=True)
            )
        for attr in ('n_estimators', 'iterations', 'max_iter'):
            if hasattr(step, attr):
                model.set_params(**{
                    f'model__{attr}': trial.suggest_int('n_est', 100, 500, step=100)
                })

        cv = TimeSeriesSplit(n_splits=3)
        scorer = make_scorer(smape, greater_is_better=False)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer, n_jobs=1)
        return -float(np.mean(scores))

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
):
    """Returns an Optuna objective for one CryptoNet architecture."""
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
):
    """Returns an Optuna objective for one CryptoNet sign-classifier architecture
    (minimises validation weighted-BCE)."""
    def objective(trial):
        hp = _sample_crypto_hparams(trial, arch, window_size)
        lr = hp.pop('learning_rate')
        weight_decay = hp.pop('weight_decay', 0.0)
        clf = CryptoNetClassifier(
            arch=arch, target_col=target_col, feature_cols=feature_cols,
            window_size=window_size, hp=hp, lr=lr, epochs=epochs,
            batch_size=batch_size, patience=NN_PATIENCE, seed=seed, weight_decay=weight_decay,
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

    Returns
    -------
    metrics_df : per-fold metrics (fold, model, MAE, RMSE, SMAPE, MASE, DA, n_train, n_test)
    oof_df     : OOF predictions (timestamp, actual, predicted, fold, model)
    """
    if model_family not in ('GB', 'NN', 'NN_CLF'):
        raise ValueError(f'model_family must be "GB", "NN" or "NN_CLF", got {model_family!r}')

    metrics_rows = []
    oof_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_seed = seed + fold_idx

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

            # Optuna tunes lr + n_estimators inside a TimeSeriesSplit on train fold
            study = run_optuna_study(
                _gb_objective(gb_model_factory, X_train, y_train),
                direction='minimize',
                n_trials=optuna_n_trials,
                seed=fold_seed,
                study_name=f'{model_name}_fold{fold_idx}',
            )

            best = gb_model_factory()
            step = best.named_steps['model']
            bp = study.best_params
            if 'lr' in bp and hasattr(step, 'learning_rate'):
                best.set_params(model__learning_rate=bp['lr'])
            for attr in ('n_estimators', 'iterations', 'max_iter'):
                if 'n_est' in bp and hasattr(step, attr):
                    best.set_params(**{f'model__{attr}': bp['n_est']})

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

            study = run_optuna_study(
                _nn_objective(arch, seq_train_df, selected, window_size, epochs, batch_size, fold_seed, target_col=target_col),
                direction='minimize',
                n_trials=optuna_n_trials,
                seed=fold_seed,
                study_name=f'{model_name}_fold{fold_idx}',
            )

            best_params = dict(study.best_params)
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

            study = run_optuna_study(
                _nn_clf_objective(arch, seq_train_df, selected, window_size, epochs, batch_size, fold_seed, target_col=target_col),
                direction='minimize',
                n_trials=optuna_n_trials,
                seed=fold_seed,
                study_name=f'{model_name}_fold{fold_idx}',
            )

            best_params = dict(study.best_params)
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
        metrics = compute_metrics(
            y_true=y_test.to_numpy(dtype=float),
            y_pred=y_pred_aligned.to_numpy(dtype=float),
            y_train=y_train.to_numpy(dtype=float),
        )
        is_clf = (model_family == 'NN_CLF')
        metrics_rows.append({
            'fold': fold_idx,
            'model': model_name,
            # Для классификатора метрики ВЕЛИЧИНЫ не определены (прогноз = вероятность),
            # поэтому MAE/RMSE/SMAPE/MASE = NaN; содержательны DA и AUC.
            'MAE': np.nan if is_clf else metrics['mae'],
            'RMSE': np.nan if is_clf else metrics['rmse'],
            'SMAPE': np.nan if is_clf else metrics['smape'],
            'MASE': np.nan if is_clf else metrics['mase'],
            'DA': metrics['da'],
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
