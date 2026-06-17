"""Тесты модуля metrics.py (EPIC 0, T0.3).

Проверки на синтетике: идеальный прогноз → mwda=1 / return_capture=1; случайный →
IC≈0; симметрия MCC; знак net_cost_sharpe; границы [0,1] вероятностных метрик.
Запуск: `pytest -q`.
"""

import numpy as np
import pytest

import metrics as M


@pytest.fixture
def rng():
    return np.random.default_rng(20260617)


# ==================== rank_ic / ic_ir ====================

def test_rank_ic_perfect_monotonic():
    r = np.linspace(-0.05, 0.05, 200)
    # монотонное нелинейное преобразование сохраняет ранги -> IC = 1
    assert M.rank_ic(r, np.exp(3 * r)) == pytest.approx(1.0, abs=1e-9)


def test_rank_ic_inverted():
    r = np.linspace(-0.05, 0.05, 200)
    assert M.rank_ic(r, -r) == pytest.approx(-1.0, abs=1e-9)


def test_rank_ic_random_near_zero(rng):
    r = rng.normal(size=5000)
    pred = rng.normal(size=5000)
    assert abs(M.rank_ic(r, pred)) < 0.06


def test_rank_ic_degenerate_returns_nan():
    assert np.isnan(M.rank_ic(np.ones(10), np.arange(10)))
    assert np.isnan(M.rank_ic(np.arange(1), np.arange(1)))


def test_ic_ir_basic():
    # mean=0.1, std(ddof=1) известна -> проверяем знак и масштаб
    ics = [0.1, 0.2, 0.0, 0.1]
    val = M.ic_ir(ics)
    expected = np.mean(ics) / (np.std(ics, ddof=1) + 1e-12)
    assert val == pytest.approx(expected, rel=1e-6)


def test_ic_ir_drops_nans():
    assert M.ic_ir([0.1, np.nan, 0.2, np.nan]) == pytest.approx(
        np.mean([0.1, 0.2]) / (np.std([0.1, 0.2], ddof=1) + 1e-12), rel=1e-6
    )
    assert np.isnan(M.ic_ir([np.nan, 0.1]))


# ==================== mwda / return_capture ====================

def test_mwda_perfect_is_one(rng):
    r = rng.normal(scale=0.02, size=1000)
    assert M.mwda(r, r) == pytest.approx(1.0)


def test_mwda_inverted_is_zero(rng):
    r = rng.normal(scale=0.02, size=1000)
    # знак противоположный (нулей в r нет с вероятностью 1)
    assert M.mwda(r, -r) == pytest.approx(0.0)


def test_mwda_in_unit_interval(rng):
    r = rng.normal(scale=0.02, size=1000)
    pred = rng.normal(size=1000)
    assert 0.0 <= M.mwda(r, pred) <= 1.0


def test_return_capture_perfect_is_one(rng):
    r = rng.normal(scale=0.02, size=1000)
    assert M.return_capture(r, r) == pytest.approx(1.0)


def test_return_capture_inverted_is_minus_one(rng):
    r = rng.normal(scale=0.02, size=1000)
    assert M.return_capture(r, -r) == pytest.approx(-1.0)


def test_return_capture_random_near_zero(rng):
    r = rng.normal(scale=0.02, size=20000)
    pred = rng.normal(size=20000)
    assert abs(M.return_capture(r, pred)) < 0.05


# ==================== mcc / balanced_accuracy ====================

def test_mcc_perfect():
    yt = np.array([1, 1, -1, -1, 1, -1])
    assert M.mcc(yt, yt) == pytest.approx(1.0)


def test_mcc_symmetry(rng):
    a = rng.choice([-1.0, 1.0], size=500)
    b = rng.choice([-1.0, 1.0], size=500)
    assert M.mcc(a, b) == pytest.approx(M.mcc(b, a), rel=1e-9)


def test_mcc_single_class_returns_zero():
    # прогноз всегда положительный -> один множитель знаменателя = 0 -> 0.0
    yt = np.array([1, -1, 1, -1])
    yp = np.array([1, 1, 1, 1])
    assert M.mcc(yt, yp) == 0.0


def test_balanced_accuracy_perfect():
    yt = np.array([1, 1, -1, -1])
    assert M.balanced_accuracy(yt, yt) == pytest.approx(1.0)


def test_balanced_accuracy_majority_guess_is_half():
    # 8 up / 2 down, предсказываем всегда up -> TPR=1, TNR=0 -> 0.5
    yt = np.array([1] * 8 + [-1] * 2)
    yp = np.ones_like(yt)
    assert M.balanced_accuracy(yt, yp) == pytest.approx(0.5)


# ==================== net_cost_sharpe ====================

def test_net_cost_sharpe_sign(rng):
    r = rng.normal(scale=0.02, size=2000)
    long_perfect = M.net_cost_sharpe(np.sign(r), r, fee=0.0)
    short_perfect = M.net_cost_sharpe(-np.sign(r), r, fee=0.0)
    assert long_perfect > 0
    assert short_perfect < 0
    assert long_perfect == pytest.approx(-short_perfect, rel=1e-9)


def test_net_cost_sharpe_fee_reduces(rng):
    # переменная магнитуда -> доходность стратегии не константна -> std>0, Sharpe конечен;
    # частые перевороты позиции делают издержки на оборот ощутимыми
    r = rng.normal(scale=0.02, size=2000)
    side = np.sign(r)
    s_free = M.net_cost_sharpe(side, r, fee=0.0)
    s_costly = M.net_cost_sharpe(side, r, fee=0.01)
    assert np.isfinite(s_free) and np.isfinite(s_costly)
    assert s_costly < s_free


def test_net_cost_sharpe_annualization():
    r = np.tile([0.01, -0.005, 0.008, -0.003], 100)
    side = np.ones_like(r)
    s1 = M.net_cost_sharpe(side, r, fee=0.0, ann=1.0)
    s252 = M.net_cost_sharpe(side, r, fee=0.0, ann=252.0)
    assert s252 == pytest.approx(np.sqrt(252.0) * s1, rel=1e-9)


# ==================== psr / deflated_sharpe / pbo ====================

def test_psr_in_unit_interval(rng):
    r = rng.normal(loc=0.001, scale=0.02, size=1000)
    p = M.psr(r, sr_benchmark=0.0)
    assert 0.0 <= p <= 1.0


def test_psr_strong_positive_high():
    rng = np.random.default_rng(1)
    r = rng.normal(loc=0.01, scale=0.01, size=2000)  # SR ~ 1 per-period, длинная выборка
    assert M.psr(r, sr_benchmark=0.0) > 0.99


def test_psr_zero_mean_is_half():
    # симметричная выборка с ŜR≈0 -> PSR(0)≈0.5
    r = np.tile([0.02, -0.02], 1000)
    assert M.psr(r, sr_benchmark=0.0) == pytest.approx(0.5, abs=0.05)


def test_deflated_sharpe_in_unit_interval(rng):
    sharpes = rng.normal(loc=0.05, scale=0.1, size=30)
    dsr = M.deflated_sharpe(sharpes, observed_sr=0.5, T=500)
    assert 0.0 <= dsr <= 1.0


def test_deflated_sharpe_more_trials_lower():
    # больше испытаний -> выше планка SR0 -> ниже DSR при том же observed_sr
    base = np.random.default_rng(0).normal(0.05, 0.1, size=200)
    dsr_few = M.deflated_sharpe(base[:5], observed_sr=0.4, T=500)
    dsr_many = M.deflated_sharpe(base, observed_sr=0.4, T=500)
    assert dsr_many <= dsr_few


def test_pbo_random_matrix_near_half(rng):
    # чистый шум: ни одна конфигурация не имеет преимущества -> PBO ~ 0.5
    M_perf = rng.normal(size=(64, 20))
    pbo = M.prob_backtest_overfitting(M_perf, n_splits=10)
    assert 0.0 <= pbo <= 1.0
    assert abs(pbo - 0.5) < 0.3


def test_pbo_genuine_signal_low(rng):
    # одна конфигурация стабильно лучшая во всех срезах -> низкий PBO
    T, N = 64, 20
    noise = rng.normal(scale=0.1, size=(T, N))
    noise[:, 0] += 1.0  # конфигурация 0 доминирует и IS, и OOS
    pbo = M.prob_backtest_overfitting(noise, n_splits=10)
    assert pbo < 0.2


def test_metrics_import_is_lightweight():
    # модуль метрик не должен тянуть тяжёлый ML-стек
    import sys
    import importlib
    importlib.reload(M)
    assert "torch" not in M.__dict__
    assert "catboost" not in M.__dict__
