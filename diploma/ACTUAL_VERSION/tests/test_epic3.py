"""EPIC 3 — метрика селекции и лоссы.

Покрывает:
  * focal_mw_bce: сведение к взвешенной BCE при gamma=0 и даунвейт «лёгких» баров;
  * _gb_objective: тюнинг полного HP-пространства по net-of-cost Sharpe (>2 параметра,
    конечное значение);
  * _apply_gb_params: применение применимых параметров и пропуск отсутствующих;
  * select_best_by_composite: ранжирование по (IC IR, NetSharpe, MWDA), tie-break MASE.
"""

import numpy as np
import pandas as pd
import pytest

import optuna
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from ml_models import focal_mw_bce
from walk_forward import _gb_objective, _apply_gb_params, select_best_by_composite

optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------
# focal_mw_bce (T3.3)
# --------------------------------------------------------------------------

def test_focal_reduces_to_weighted_bce_at_gamma0():
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0, 0.1])
    y = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    w = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.8])
    bce = nn.BCEWithLogitsLoss(reduction='none')
    expected = (bce(logits, y) * w).mean()
    got = focal_mw_bce(logits, y, w, gamma=0.0)
    assert torch.allclose(got, expected, atol=1e-6)


def test_focal_downweights_easy_bars():
    # уверенно и верно классифицированные («лёгкие») бары при gamma>0 весят меньше
    logits = torch.tensor([5.0, -5.0, 6.0])
    y = torch.tensor([1.0, 0.0, 1.0])
    w = torch.ones(3)
    l0 = focal_mw_bce(logits, y, w, gamma=0.0)
    l2 = focal_mw_bce(logits, y, w, gamma=2.0)
    assert float(l2) < float(l0)


def test_focal_is_differentiable():
    logits = torch.tensor([0.7, -0.4, 1.2], requires_grad=True)
    y = torch.tensor([1.0, 0.0, 1.0])
    w = torch.tensor([1.0, 1.0, 1.0])
    loss = focal_mw_bce(logits, y, w, gamma=2.0)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# --------------------------------------------------------------------------
# _gb_objective + _apply_gb_params (T3.2)
# --------------------------------------------------------------------------

def _make_gb_factory():
    def factory():
        return Pipeline([('model', HistGradientBoostingRegressor(random_state=0))])
    return factory


@pytest.fixture
def gb_data():
    rng = np.random.default_rng(0)
    n = 160
    f0 = rng.normal(size=n)
    f1 = rng.normal(size=n)
    X = pd.DataFrame({'f0': f0, 'f1': f1, 'f2': rng.normal(size=n)})
    # доходность со слабым, но реальным сигналом от f0 -> sign(pred) несёт edge
    y = pd.Series(0.01 * f0 + 0.002 * rng.normal(size=n))
    return X, y


def test_gb_objective_tunes_full_space_and_finite(gb_data):
    X, y = gb_data
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(_gb_objective(_make_gb_factory(), X, y), n_trials=6)
    # objective = -mean(Sharpe) -> конечно (не SMAPE-вырождение, не inf)
    assert np.isfinite(study.best_value)
    # тюнится больше двух параметров (lr + n_est + структурные)
    assert len(study.best_params) > 2
    assert {'lr', 'n_est'}.issubset(study.best_params)
    # хотя бы один структурный параметр, поддерживаемый HistGB
    assert {'max_depth', 'l2_regularization'} & set(study.best_params)


def test_apply_gb_params_applies_applicable_and_skips_missing():
    model = _make_gb_factory()()
    _apply_gb_params(model, {
        'lr': 0.07, 'n_est': 150, 'max_depth': 4, 'l2_regularization': 0.3,
        'num_leaves': 99, 'reg_lambda': 1.0,   # не существуют у HistGB -> пропуск
    })
    step = model.named_steps['model']
    assert step.learning_rate == 0.07
    assert step.max_iter == 150           # 'n_est' -> max_iter
    assert step.max_depth == 4
    assert step.l2_regularization == 0.3
    # неприменимые параметры не пробрасываются и не роняют set_params
    assert not hasattr(step, 'num_leaves')


# --------------------------------------------------------------------------
# select_best_by_composite (T3.1)
# --------------------------------------------------------------------------

def _agg_row(freq, model, ic_ir, netsharpe, mwda, mase):
    return {'frequency': freq, 'model': model, 'IC_IR': ic_ir,
            'NetSharpe': netsharpe, 'MWDA': mwda, 'MASE': mase}


def test_composite_picks_highest_ic_ir():
    agg = pd.DataFrame([
        _agg_row('daily', 'A', ic_ir=2.0, netsharpe=0.1, mwda=0.55, mase=1.0),
        _agg_row('daily', 'B', ic_ir=0.1, netsharpe=5.0, mwda=0.90, mase=0.5),
    ])
    best = select_best_by_composite(agg, ['A', 'B'])
    assert list(best['model']) == ['A']   # IC IR ведёт, несмотря на лучший Sharpe у B


def test_composite_tie_break_mase():
    # равны по IC_IR/NetSharpe/MWDA -> побеждает меньший MASE
    agg = pd.DataFrame([
        _agg_row('daily', 'A', 1.0, 1.0, 0.6, mase=0.9),
        _agg_row('daily', 'B', 1.0, 1.0, 0.6, mase=0.4),
    ])
    best = select_best_by_composite(agg, ['A', 'B'])
    assert list(best['model']) == ['B']


def test_composite_one_winner_per_frequency():
    agg = pd.DataFrame([
        _agg_row('daily', 'A', 2.0, 0.1, 0.5, 1.0),
        _agg_row('daily', 'B', 0.1, 0.1, 0.5, 1.0),
        _agg_row('hourly', 'A', 0.1, 0.1, 0.5, 1.0),
        _agg_row('hourly', 'B', 3.0, 0.1, 0.5, 1.0),
    ])
    best = select_best_by_composite(agg, ['A', 'B'])
    assert set(best['frequency']) == {'daily', 'hourly'}
    assert best.set_index('frequency').loc['daily', 'model'] == 'A'
    assert best.set_index('frequency').loc['hourly', 'model'] == 'B'


def test_composite_empty_when_no_names_match():
    agg = pd.DataFrame([_agg_row('daily', 'A', 1.0, 1.0, 0.5, 1.0)])
    assert select_best_by_composite(agg, ['Z']).empty
