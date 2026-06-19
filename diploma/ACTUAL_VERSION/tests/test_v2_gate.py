"""Unit tests for the M3 distributional ablation gate (v2_microstructure/src/evaluate.py)."""

import numpy as np
import pandas as pd
import pytest

from v2_microstructure.src import evaluate as ev, labeling as lb


def test_gate_passes_when_flow_clearly_beats_control():
    rng = np.random.default_rng(0)
    control = rng.normal(-7.0, 0.5, 15)
    flow = control + 1.4 + rng.normal(0, 0.2, 15)   # consistent positive edge
    g = ev.distributional_gate(flow, control)
    assert g["passed"]
    assert g["majority_flow_gt_control"] > 0.5
    assert g["delta_ci_low"] > 0.0


def test_gate_fails_when_difference_is_noise():
    rng = np.random.default_rng(1)
    control = rng.normal(-6.0, 1.0, 15)
    flow = control + rng.normal(0, 1.0, 15)         # zero-mean difference
    g = ev.distributional_gate(flow, control)
    assert not g["passed"]                          # CI should straddle 0


def test_gate_fails_when_flow_is_worse():
    rng = np.random.default_rng(2)
    control = rng.normal(-5.0, 0.5, 15)
    flow = control - 1.0 + rng.normal(0, 0.2, 15)
    g = ev.distributional_gate(flow, control)
    assert not g["passed"]
    assert g["majority_flow_gt_control"] < 0.5


def test_paired_bootstrap_ci_brackets_mean():
    d = np.full(20, 1.5)
    mean, lo, hi = ev.paired_bootstrap_ci(d)
    assert mean == 1.5 and lo <= 1.5 <= hi


def test_forward_return_is_one_bar_ahead():
    close = pd.Series([100, 110, 99], index=pd.date_range("2020", periods=3, freq="h"))
    fr = ev.forward_return(close)
    assert fr.iloc[0] == pytest.approx(0.1)           # 110/100 - 1
    assert np.isnan(fr.iloc[-1])                       # no bar after the last


# --------------------------------------------------------------------------- de-overlap
def test_de_overlap_selects_non_overlapping_bets():
    entry = np.array([0, 1, 3, 5]); t1 = np.array([3, 4, 6, 8])
    side = np.ones(4); pe = np.full(4, 100.0); p1 = np.full(4, 110.0)
    rets, holds = ev.de_overlapped_trade_returns(entry, t1, side, pe, p1, cost_rt=0.0)
    # greedy: take (0,3) then (3,6); (1,4) and (5,8) overlap a taken bet
    assert len(rets) == 2
    assert list(holds) == [3, 3]
    assert rets == pytest.approx([0.1, 0.1])


def test_de_overlap_skips_zero_side():
    entry = np.array([0, 4]); t1 = np.array([2, 6])
    side = np.array([0, 1]); pe = np.array([100.0, 100.0]); p1 = np.array([110.0, 90.0])
    rets, holds = ev.de_overlapped_trade_returns(entry, t1, side, pe, p1, cost_rt=0.0)
    assert len(rets) == 1 and rets[0] == pytest.approx(-0.1)  # only the +1 bet, which lost


def test_de_overlap_applies_side_and_cost():
    entry = np.array([0]); t1 = np.array([3])
    side = np.array([-1]); pe = np.array([100.0]); p1 = np.array([110.0])
    rets, _ = ev.de_overlapped_trade_returns(entry, t1, side, pe, p1, cost_rt=0.0014)
    assert rets[0] == pytest.approx(-0.1 - 0.0014)            # short into a rally, minus round-trip


def test_trade_sharpe_positive_and_nan_guard():
    rng = np.random.default_rng(0)
    rets = 0.01 + rng.normal(0, 0.002, 50); holds = np.full(50, 24.0)
    assert ev.trade_sharpe(rets, holds) > 0
    assert np.isnan(ev.trade_sharpe(np.array([0.01]), np.array([24.0])))  # <2 trades


def test_build_trade_info_aligns_positions_and_prices():
    close = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 1.0, 300)),
                      index=pd.date_range("2021-01-01", periods=300, freq="h"))
    L = lb.triple_barrier_labels(close, k=1.5, H=12, sigma_span=20)
    ti = ev.build_trade_info(L, close)
    assert (ti["t1_pos"] - ti["entry_pos"] == L.frame["bars_held"].to_numpy()).all()
    assert ti["entry_price"].to_numpy() == pytest.approx(close.reindex(L.frame.index).to_numpy())
