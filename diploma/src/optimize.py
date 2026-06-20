"""Hyperparameter tuning on the PURGED WALK-FORWARD (not random, not plain k-fold).

Optuna TPE + MedianPruner; the objective is mean **MCC** across walk-forward folds (the
pre-registered selection metric). The label **threshold is never in the search space**
(it is the target definition — tuning it would be label snooping, R12 / config note).

The same walk-forward OOF machinery is used at tuning time and to produce the final
out-of-fold predictions that feed ``metrics.evaluate``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import optuna
from sklearn.metrics import matthews_corrcoef

try:
    from . import config, models, validation
except ImportError:  # standalone
    import config, models, validation

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_splits(n: int, n_splits: Optional[int] = None) -> list:
    """Purged walk-forward splits from config (honours WINDOW_MODE/PURGE/EMBARGO)."""
    return validation.purged_walk_forward_splits(
        n, n_splits=n_splits or config.wf_splits(), purge=config.PURGE,
        embargo=config.EMBARGO, window_mode=config.WINDOW_MODE,
        train_window=config.TRAIN_WINDOW)


def _make_setup(setup_key: str, params: dict):
    factory = models.SETUPS[setup_key]
    return factory(params=params)


def walk_forward_oof(setup_key: str, X, y3, fwd_ret, splits, params: Optional[dict] = None):
    """Прогнать постановку по walk-forward фолдам, собрать OOF-предсказания.

    Возвращает dict с конкатенированными по тестовым блокам массивами: ``y_true``,
    ``y_pred``, ``proba`` (n,3), ``signal``, ``fwd``, ``fold`` (id фолда), ``pos``
    (позиционный индекс в исходном X) и ``per_fold_mcc``."""
    import pandas as pd
    X = pd.DataFrame(X)
    y3 = np.asarray(y3).astype(int)
    fwd = np.asarray(fwd_ret, dtype=float)
    params = params or {}

    yt, yp, pr, sg, fw, fold, pos = [], [], [], [], [], [], []
    per_fold_mcc = []
    for fi, (tr, te) in enumerate(splits):
        setup = _make_setup(setup_key, params)
        setup.fit(X.iloc[tr], y3[tr], y_ret=fwd[tr])
        proba = setup.predict_proba3(X.iloc[te])
        pred = np.asarray(models.CLASSES)[proba.argmax(axis=1)]
        yt.append(y3[te]); yp.append(pred); pr.append(proba)
        sg.append(proba[:, 2] - proba[:, 0]); fw.append(fwd[te])
        fold.append(np.full(te.size, fi)); pos.append(te)
        per_fold_mcc.append(float(matthews_corrcoef(y3[te], pred)) if np.unique(y3[te]).size > 1 else np.nan)
    return {
        "y_true": np.concatenate(yt), "y_pred": np.concatenate(yp),
        "proba": np.concatenate(pr), "signal": np.concatenate(sg),
        "fwd": np.concatenate(fw), "fold": np.concatenate(fold),
        "pos": np.concatenate(pos), "per_fold_mcc": per_fold_mcc,
    }


def _suggest_params(trial: optuna.Trial) -> dict:
    """LightGBM HP-пространство (порог метки НЕ входит — это определение таргета)."""
    return {
        # Ёмкость срезана под низкий SNR / ~500 независимых наблюдений: мелкие
        # неглубокие деревья + высокий пол min_child + обязательный сабсэмплинг (<1).
        # n_estimators — лишь ПОТОЛОК: фактическое число выбирает early stopping (_fit_es).
        "num_leaves": trial.suggest_int("num_leaves", 8, 64, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, config.n_estimators_cap()),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 500, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
    }


def _make_objective(setup_key: str, X, y3, fwd, splits):
    """Objective-замыкание (mean MCC по walk-forward фолдам, с pruning по фолдам)."""
    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        scores = []
        for fi, (tr, te) in enumerate(splits):
            setup = _make_setup(setup_key, params)
            setup.fit(X.iloc[tr], y3[tr], y_ret=fwd[tr])
            pred = setup.predict(X.iloc[te])
            sc = matthews_corrcoef(y3[te], pred) if np.unique(y3[te]).size > 1 else 0.0
            scores.append(sc)
            trial.report(float(np.mean(scores)), step=fi)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))
    return objective


def _persist_best(artifact_path, setup_key, best_params, n_done, best_value):
    if not artifact_path:
        return
    import json
    from pathlib import Path
    p = Path(artifact_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"setup": setup_key, "best_params": best_params,
                             "n_trials": int(n_done), "best_value": float(best_value)},
                            indent=2))


def _tune_serial(setup_key, X, y3, fwd, splits, n_trials, seed):
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(_make_objective(setup_key, X, y3, fwd, splits),
                   n_trials=n_trials, show_progress_bar=False)
    return study


def _worker_optimize(study_name, storage_path, setup_key, X_arr, cols, idx_vals,
                     y3, fwd, splits, n_trials, lgbm_threads, seed):
    """Pool-воркер: пинит потоки, грузит общую study из JournalStorage, гонит свою долю
    триалов. X_arr приходит read-only memmap'ом (joblib) — RAM шарится между воркерами."""
    try:
        from . import compute
    except ImportError:
        import compute
    compute.set_worker_threads(lgbm_threads)
    import pandas as pd
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    storage = JournalStorage(JournalFileBackend(storage_path))
    study = optuna.load_study(study_name=study_name, storage=storage,
                              sampler=optuna.samplers.TPESampler(seed=seed),
                              pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    X = pd.DataFrame(np.asarray(X_arr), columns=cols, index=pd.DatetimeIndex(idx_vals))
    study.optimize(_make_objective(setup_key, X, y3, fwd, splits),
                   n_trials=n_trials, show_progress_bar=False)
    return len(study.trials)


def _tune_parallel(setup_key, X, y3, fwd, splits, n_trials, seed, parallel_fits, lgbm_threads):
    """Параллельный Optuna через общий JournalStorage + loky-пул (W1).

    V1: параллельный TPE НЕ бит-воспроизводим (сэмплер кондиционируется на множестве
    завершённых триалов → гонка). best-параметры персистятся; финальный refit/оценка
    детерминированы. Для строго воспроизводимого отчёта используйте serial-путь."""
    import os
    import uuid
    import pandas as pd
    from joblib import Parallel, delayed
    rundir = config.ROOT / "reports" / "optuna"
    rundir.mkdir(parents=True, exist_ok=True)
    study_name = f"{setup_key}_{uuid.uuid4().hex[:8]}"
    storage_path = str(rundir / f"{study_name}.log")
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    storage = JournalStorage(JournalFileBackend(storage_path))
    optuna.create_study(direction="maximize", study_name=study_name, storage=storage,
                        sampler=optuna.samplers.TPESampler(seed=seed),
                        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    Xf = pd.DataFrame(X).astype("float32")
    X_arr = Xf.to_numpy()
    cols, idx_vals = list(Xf.columns), np.asarray(Xf.index.values)
    per = [n_trials // parallel_fits + (1 if i < n_trials % parallel_fits else 0)
           for i in range(parallel_fits)]
    Parallel(n_jobs=parallel_fits, backend="loky")(
        delayed(_worker_optimize)(study_name, storage_path, setup_key, X_arr, cols, idx_vals,
                                  y3, fwd, splits, k, lgbm_threads, seed + wi)
        for wi, k in enumerate(per) if k > 0)
    return optuna.load_study(study_name=study_name, storage=storage)


def tune_setup(setup_key: str, X, y3, fwd_ret, splits=None, n_trials: Optional[int] = None,
               seed: int = config.SEED, parallel: Optional[bool] = None,
               artifact_path: Optional[str] = None):
    """Optuna-тюнинг постановки по walk-forward; отбор по среднему MCC.

    ``parallel``: None → авто (parallel если ``COMPUTE_CFG.parallel_fits>1``), False →
    serial (DETERMINISTIC, для воспроизводимого отчёта). Возвращает ``(best_params,
    study)``; best-параметры персистятся в ``artifact_path`` (V1). ``study.trials`` идёт
    в n_trials для DSR (R12)."""
    import pandas as pd
    X = pd.DataFrame(X)
    y3 = np.asarray(y3).astype(int)
    fwd = np.asarray(fwd_ret, dtype=float)
    splits = splits if splits is not None else make_splits(len(X))
    n_trials = n_trials or config.n_trials()
    cfg = config.COMPUTE_CFG
    if parallel is None:
        # auto: parallel only when there are enough trials to amortise loky cold-start
        # (QUICK locally → serial & deterministic; FULL on the box → parallel).
        use_parallel = cfg.get("parallel_fits", 1) > 1 and n_trials >= 2 * cfg["parallel_fits"]
    else:
        use_parallel = parallel

    if use_parallel:
        try:
            study = _tune_parallel(setup_key, X, y3, fwd, splits, n_trials, seed,
                                   cfg["parallel_fits"], cfg["lgbm_threads"])
        except Exception as e:  # noqa: BLE001 — fallback to serial on any pool/storage issue
            import logging
            logging.getLogger(__name__).warning("parallel Optuna failed (%s) — serial fallback", e)
            study = _tune_serial(setup_key, X, y3, fwd, splits, n_trials, seed)
    else:
        study = _tune_serial(setup_key, X, y3, fwd, splits, n_trials, seed)

    _persist_best(artifact_path, setup_key, study.best_params, len(study.trials), study.best_value)
    return study.best_params, study
