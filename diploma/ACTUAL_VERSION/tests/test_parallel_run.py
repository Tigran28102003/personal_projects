"""T7.6 — параллельный раннер: построение спеков + joblib/loky-оркестрация.

Реальные прогоны run_walk_forward (torch/lightgbm) НЕ исполняются — тестируем
оркестрацию через тривиальный picklable runner (быстро, без libomp-рисков).
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from parallel_run import build_run_specs, run_specs_parallel


def _frame(n=60):
    idx = pd.date_range('2021-01-01', periods=n, freq='D')
    rng = np.random.default_rng(0)
    return pd.DataFrame({'BTC': rng.normal(size=n), 'f0': rng.normal(size=n)}, index=idx)


def _gb_templates():
    return {
        'LightGBM': Pipeline([('model', HistGradientBoostingRegressor(random_state=0))]),
        'XGBoost': Pipeline([('model', HistGradientBoostingRegressor(random_state=1))]),
        'CatBoost': Pipeline([('model', HistGradientBoostingRegressor(random_state=2))]),
    }


def _specs(run_clf=True):
    df = _frame()
    splits = [(np.arange(30), np.arange(30, 40)), (np.arange(40), np.arange(40, 50))]
    return build_run_specs(
        'daily', df, 'BTC', ['f0'], splits, top_k=1,
        gb_templates=_gb_templates(), nn_archs=['LSTM', 'GRU'], run_clf=run_clf,
        nn_config={'window_size': 5, 'epochs': 1, 'batch_size': 8},
        seed=42, retune_every=5, gb_trials=3, nn_trials=2, horizon=1, inner_threads=4,
    )


# --------------------------------------------------------------------------
# build_run_specs
# --------------------------------------------------------------------------

def test_build_run_specs_counts_and_families():
    specs = _specs(run_clf=True)
    fams = [s['model_family'] for s in specs]
    assert fams.count('GB') == 3 and fams.count('NN') == 2 and fams.count('NN_CLF') == 2
    assert len(specs) == 7
    names = {s['model_name'] for s in specs}
    assert 'CatBoost' in names and 'LSTM (sequence NN)' in names and 'GRU (sign clf)' in names


def test_build_run_specs_no_clf():
    specs = _specs(run_clf=False)
    assert [s['model_family'] for s in specs].count('NN_CLF') == 0
    assert len(specs) == 5


def test_specs_have_required_fields_and_trials():
    specs = _specs()
    req = {'model_family', 'model_name', 'df', 'target_col', 'feat_pool', 'splits',
           'top_k', 'optuna_trials', 'seed', 'retune_every', 'horizon', 'inner_threads'}
    for s in specs:
        assert req <= set(s)
    gb = next(s for s in specs if s['model_family'] == 'GB')
    nn = next(s for s in specs if s['model_family'] == 'NN')
    assert gb['optuna_trials'] == 3 and 'gb_template' in gb
    assert nn['optuna_trials'] == 2 and 'nn_config' in nn


def test_specs_are_picklable():
    # для раскидывания по процессам спеки обязаны пиклиться
    pickle.dumps(_specs())


# --------------------------------------------------------------------------
# run_specs_parallel — оркестрация (тривиальный runner)
# --------------------------------------------------------------------------

def _double_runner(spec):
    """Тривиальный picklable прогон-заглушка (без torch/lightgbm)."""
    return spec['model_name'], spec['x'] * 2, None


def test_run_specs_sequential_fallback():
    specs = [{'model_name': 'a', 'x': 1}, {'model_name': 'b', 'x': 2}]
    out = run_specs_parallel(specs, n_workers=1, runner=_double_runner)
    assert [o[0] for o in out] == ['a', 'b']
    assert [o[1] for o in out] == [2, 4]


def test_run_specs_single_spec_runs_inline():
    out = run_specs_parallel([{'model_name': 'solo', 'x': 5}], n_workers=8, runner=_double_runner)
    assert out == [('solo', 10, None)]


def test_run_specs_parallel_loky():
    specs = [{'model_name': str(i), 'x': i} for i in range(4)]
    out = run_specs_parallel(specs, n_workers=2, runner=_double_runner)
    assert sorted(o[1] for o in out) == [0, 2, 4, 6]   # порядок сохраняется/полнота
    assert [o[1] for o in out] == [0, 2, 4, 6]
