"""Параллельный walk-forward по независимым прогонам (модель × частота) — T7.6.

Каждый вызов `run_walk_forward` независим, поэтому прогоны раскидываются по ПРОЦЕССАМ
(joblib/loky) на многоядерный CPU — основной рычаг ускорения (узкое место пайплайна
CPU-bound: бустинги + Optuna). NN-прогоны делят один GPU (модели мелкие). Чтобы 36 ядер
не передрались (oversubscription), каждому воркеру ограничивается число потоков
(`inner_threads`). Есть последовательный fallback (`n_workers <= 1`).

Замечания по окружению:
- loky пиклит замыкания через cloudpickle, поэтому спеки/функции работают и из ноутбука.
- В воркере выставляется `KMP_DUPLICATE_LIB_OK=TRUE` (страховка от конфликта libomp
  torch ↔ LightGBM в одном процессе).
- Большой `df` пиклится в каждый таск — на фоне времени обучения это пренебрежимо.
"""

from __future__ import annotations

import os
from typing import Callable, Optional


def _apply_thread_limit(n: Optional[int]) -> None:
    """Ограничить число потоков воркера ДО тяжёлых вычислений (анти-oversubscription)."""
    if n and n > 0:
        for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[var] = str(n)
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


def _run_single(spec: dict):
    """Один walk-forward-прогон по спецификации (исполняется в воркере).

    Возвращает ``(model_name, metrics_df, oof_df)``. Тяжёлые импорты — внутри функции,
    чтобы воркер инициализировал их уже с применённым лимитом потоков.
    """
    _apply_thread_limit(spec.get('inner_threads'))
    from sklearn.base import clone
    from walk_forward import run_walk_forward

    gb_factory = None
    if spec['model_family'] == 'GB':
        template = spec['gb_template']
        gb_factory = lambda: clone(template)   # noqa: E731 — свежий клон на каждый фит

    mdf, odf = run_walk_forward(
        model_family=spec['model_family'],
        model_name=spec['model_name'],
        df=spec['df'],
        target_col=spec['target_col'],
        all_feature_cols=spec['feat_pool'],
        splits=spec['splits'],
        top_k=spec['top_k'],
        optuna_n_trials=spec['optuna_trials'],
        nn_config=spec.get('nn_config'),
        gb_model_factory=gb_factory,
        seed=spec['seed'],
        retune_every=spec['retune_every'],
        horizon=spec.get('horizon', 1),
    )
    return spec['model_name'], mdf, odf


def build_run_specs(
    frequency: str,
    df,
    target_col: str,
    feat_pool,
    splits,
    top_k: int,
    *,
    gb_templates: dict,
    nn_archs,
    run_clf: bool,
    nn_config: dict,
    seed: int,
    retune_every: int,
    gb_trials: int,
    nn_trials: int,
    horizon: int = 1,
    inner_threads: Optional[int] = None,
) -> list[dict]:
    """Список спецификаций прогонов для одной частоты: GB-модели + NN-архитектуры
    (+ NN_CLF). Всё содержимое picklable (df, sklearn-шаблоны, конфиг) -> раскидывается
    по процессам. `gb_templates`: {имя -> sklearn Pipeline-шаблон}."""
    common = dict(
        frequency=frequency, df=df, target_col=target_col, feat_pool=list(feat_pool),
        splits=splits, top_k=top_k, seed=seed, retune_every=retune_every,
        horizon=horizon, inner_threads=inner_threads,
    )
    specs: list[dict] = []
    for name, template in gb_templates.items():
        specs.append({**common, 'model_family': 'GB', 'model_name': name,
                      'optuna_trials': gb_trials, 'gb_template': template})
    for arch in nn_archs:
        specs.append({**common, 'model_family': 'NN', 'model_name': f'{arch} (sequence NN)',
                      'optuna_trials': nn_trials, 'nn_config': nn_config})
    if run_clf:
        for arch in nn_archs:
            specs.append({**common, 'model_family': 'NN_CLF', 'model_name': f'{arch} (sign clf)',
                          'optuna_trials': nn_trials, 'nn_config': nn_config})
    return specs


def run_specs_parallel(specs, n_workers: int = 1, runner: Optional[Callable] = None):
    """Выполнить прогоны. `n_workers <= 1` (или один спек) -> последовательно (fallback),
    иначе joblib/loky по процессам. Порядок результатов соответствует `specs`.

    `runner` — функция одного прогона (по умолчанию `_run_single`); параметр для тестов.
    """
    run = runner or _run_single
    if n_workers <= 1 or len(specs) <= 1:
        return [run(s) for s in specs]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_workers, backend='loky')(delayed(run)(s) for s in specs)
