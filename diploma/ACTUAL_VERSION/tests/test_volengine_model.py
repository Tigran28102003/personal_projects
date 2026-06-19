"""Unit tests for the Vol-Engine ML ladder + Gate 1 (A-M3)."""

import numpy as np
import pandas as pd
import pytest

from v2_volengine.src import vol_model as vm, vol_evaluate as ve


def _synthetic_X(n=520, seed=0):
    """Persistent log-RV well-explained by HAR; exo columns are pure noise (no signal over HAR)."""
    rng = np.random.default_rng(seed)
    lr = np.zeros(n)
    for i in range(1, n):
        lr[i] = 0.8 * lr[i - 1] + rng.normal(0, 0.3)
    idx = pd.date_range("2019-09-15", periods=n, freq="D", name="date")
    logrv = pd.Series(lr - 8.0, index=idx)
    X = pd.DataFrame(index=idx)
    X["har_d"] = logrv
    X["har_w"] = logrv.rolling(5).mean()
    X["har_m"] = logrv.rolling(22).mean()
    X["rq_term"] = np.sqrt(np.exp(2 * logrv))
    X["noise1"] = rng.normal(0, 1, n)
    X["noise2"] = rng.normal(0, 1, n)
    X["target"] = logrv.shift(-1)
    return X.dropna()


ML_COLS = ["har_d", "har_w", "har_m", "rq_term", "noise1", "noise2"]


def test_wf_oof_all_arms_positive_rv_and_present():
    X = _synthetic_X()
    oof = vm.wf_oof_forecasts(X, ML_COLS, n_splits=5, embargo=5)
    assert len(oof) > 50
    for a in vm.ARMS:
        assert f"{a}_rv" in oof.columns and (oof[f"{a}_rv"].dropna() > 0).all()


def test_summarize_arms_has_dm_vs_base():
    X = _synthetic_X()
    oof = vm.wf_oof_forecasts(X, ML_COLS, n_splits=5, embargo=5)
    tbl = ve.summarize_arms(oof)
    assert set(tbl["arm"]) == set(vm.ARMS)
    assert {"QLIKE", "R2_logRV", "DM_vs_base_stat", "DM_vs_base_p", "QLIKE_vs_base_pct"} <= set(tbl.columns)
    # base row (HARQ) has 0% vs itself
    assert tbl.loc[tbl["arm"] == "HARQ", "QLIKE_vs_base_pct"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_cpcv_paths_count_and_gate_keys():
    X = _synthetic_X()
    paths = vm.cpcv_arm_losses(X, ML_COLS)
    from validation import n_cpcv_paths
    from v2_volengine.src import config
    assert len(paths) == n_cpcv_paths(config.CPCV_N, config.CPCV_K)
    g = ve.gate1_forecasting(paths, ml_arm="Ridge")
    for key in ("passed", "majority_DM_sig", "delta_ci_low", "delta_ci_high", "verdict"):
        assert key in g
    assert 0.0 <= g["majority_DM_sig"] <= 1.0


def test_noise_exo_does_not_pass_gate1():
    # honesty guard: with pure-noise exo, ML must NOT significantly beat HARQ (gate is not rigged)
    X = _synthetic_X()
    paths = vm.cpcv_arm_losses(X, ML_COLS)
    g = ve.gate1_forecasting(paths, ml_arm="Ridge")
    assert g["passed"] is False
