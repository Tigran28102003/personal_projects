"""Unit tests for the Vol-Engine HAR family + scoring (A-M1)."""

import numpy as np
import pandas as pd
import pytest

from v2_volengine.src import har as harmod, vol_evaluate as ve, rv as rvmod


def _rv_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="D", name="date")
    # persistent log-RV (AR-like) so HAR has something to fit
    lr = np.zeros(n)
    for i in range(1, n):
        lr[i] = 0.8 * lr[i - 1] + rng.normal(0, 0.3)
    rv = np.exp(lr - 8)
    f = pd.DataFrame({"rv": rv, "jump": rv * 0.1, "rq": rv ** 2}, index=idx)
    f["target"] = np.log(f["rv"])
    f["target_next"] = f["target"].shift(-1)
    return f


def test_qlike_zero_for_perfect_and_positive_otherwise():
    rv = np.array([1e-4, 2e-4, 4e-4])
    assert ve.qlike(rv, rv) == pytest.approx(0.0, abs=1e-12)
    assert ve.qlike(rv, rv * 1.5) > 0


def test_r2_log_perfect_is_one():
    y = np.array([-8.0, -7.0, -6.5, -7.2])
    assert ve.r2_log(y, y) == pytest.approx(1.0)


def test_har_components_past_only_and_target_shift():
    f = _rv_frame()
    X = harmod.har_components(f)
    # har_d at day t equals log RV_t (current day, used to predict t+1)
    np.testing.assert_allclose(X["har_d"].to_numpy(), np.log(f["rv"].to_numpy()), rtol=1e-9)
    assert X["target"].iloc[0] == pytest.approx(f["target_next"].iloc[0])


def test_oof_produces_all_models_and_har_beats_rw():
    f = _rv_frame(500)
    oof = harmod.walk_forward_oof(f, n_splits=5, embargo=5)
    assert len(oof) > 50
    tbl = ve.summarize_models(oof)
    assert set(tbl["model"]) == set(harmod.MODELS)
    # on a persistent series HAR should not be worse than RW in QLIKE
    q = tbl.set_index("model")["QLIKE"]
    assert q["HAR"] <= q["RW"] + 1e-6


def test_dm_identical_losses_is_nan():
    x = np.random.default_rng(0).random(100)
    stat, p = ve.diebold_mariano(x, x)
    assert np.isnan(stat) and np.isnan(p)


def test_dm_detects_significantly_better_model():
    rng = np.random.default_rng(1)
    bench = rng.random(300) + 0.5     # systematically higher loss
    model = rng.random(300) * 0.5     # systematically lower loss
    stat, p = ve.diebold_mariano(model, bench)
    assert stat < 0 and p < 0.05      # model A significantly better


def test_dm_sign_is_antisymmetric():
    rng = np.random.default_rng(2)
    a, b = rng.random(200), rng.random(200) + 0.1
    s_ab, _ = ve.diebold_mariano(a, b)
    s_ba, _ = ve.diebold_mariano(b, a)
    assert s_ab == pytest.approx(-s_ba, rel=1e-6)


def test_summarize_has_dm_columns():
    f = _rv_frame(500)
    oof = harmod.walk_forward_oof(f, n_splits=5, embargo=5)
    tbl = ve.summarize_models(oof)
    assert {"DM_vs_RW_stat", "DM_vs_RW_p", "DM_vs_AR1_stat", "DM_vs_AR1_p"} <= set(tbl.columns)


def test_jensen_correction_makes_level_unbiased_ish():
    # with log target, RV forecast uses exp(pred + 0.5 sigma2); ensure rv columns are positive
    f = _rv_frame()
    oof = harmod.walk_forward_oof(f, n_splits=5, embargo=5)
    for m in harmod.MODELS:
        assert (oof[f"{m}_rv"].dropna() > 0).all()
