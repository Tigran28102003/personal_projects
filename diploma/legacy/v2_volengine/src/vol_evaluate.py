"""Vol-forecast scoring (A-M1: QLIKE + R²-logRV + HAR-vs-naive table).

QLIKE is the proper volatility loss on the **variance/RV scale** (robust to the noisy RV proxy):
    QLIKE(σ², σ̂²) = mean( σ²/σ̂² − log(σ²/σ̂²) − 1 )   (lower is better; 0 = perfect)
applied to RV vs the Jensen-corrected RV-level forecast. R²-logRV is reported as a secondary,
scale-readable number. The Layer-1 result is framed as "HAR beats naive RV persistence by X%"
(QLIKE improvement vs RW) — the MASE discipline. Gate 1 (Diebold-Mariano) is added at A-M3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as _student_t

from . import config
from .har import MODELS


def qlike_series(rv_true, rv_pred) -> np.ndarray:
    """Per-observation QLIKE loss σ²/σ̂² − log(σ²/σ̂²) − 1 (finite, positive pairs only)."""
    rt = np.asarray(rv_true, dtype=float)
    rp = np.asarray(rv_pred, dtype=float)
    ratio = rt / rp
    loss = ratio - np.log(ratio) - 1.0
    loss[~np.isfinite(loss)] = np.nan
    return loss


def diebold_mariano(loss_a, loss_b, h: int = 1, hac_lags: int | None = None):
    """Diebold-Mariano with Harvey-Leybourne-Newbold small-sample correction (HAC-robust).

    Tests equal predictive accuracy on the loss differential d = loss_a − loss_b. Returns
    ``(stat, pvalue)`` against a t(T−1) distribution. **stat < 0 ⇒ model A has the lower loss
    (better)**; two-sided p. HAC long-run variance via Bartlett (Newey-West); default lag =
    max(h−1, round(T^(1/3))) to absorb the autocorrelation of volatility-forecast losses.
    Reused by Gate 1 at M3 (ML+exo vs HAR).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    T = d.size
    if T < 8 or np.allclose(d, 0):
        return float("nan"), float("nan")
    dbar = d.mean()
    dc = d - dbar
    gamma0 = float(np.mean(dc ** 2))
    L = hac_lags if hac_lags is not None else max(h - 1, int(round(T ** (1 / 3))))
    lrv = gamma0
    for k in range(1, L + 1):
        cov = float(np.mean(dc[k:] * dc[:-k]))
        lrv += 2.0 * (1.0 - k / (L + 1)) * cov
    lrv = max(lrv, 1e-18)
    dm = dbar / np.sqrt(lrv / T)
    hln = np.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 1e-12))
    stat = dm * hln
    pval = float(2.0 * _student_t.cdf(-abs(stat), df=T - 1))
    return float(stat), pval


def qlike(rv_true, rv_pred) -> float:
    rt = np.asarray(rv_true, dtype=float)
    rp = np.asarray(rv_pred, dtype=float)
    mask = np.isfinite(rt) & np.isfinite(rp) & (rt > 0) & (rp > 0)
    rt, rp = rt[mask], rp[mask]
    if rt.size == 0:
        return float("nan")
    ratio = rt / rp
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def r2_log(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if yt.size < 2:
        return float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2)) or 1.0
    return float(1.0 - ss_res / ss_tot)


def summarize_models(oof: pd.DataFrame, dm_h: int = 1) -> pd.DataFrame:
    """One row per model: QLIKE (variance scale), R²-logRV, QLIKE improvement vs RW (%), and
    Diebold-Mariano(HLN) significance of each model's QLIKE loss vs RW and vs AR1
    (``DM_*_stat`` < 0 with ``DM_*_p`` < 0.05 ⇒ the model is significantly better). ``dm_h`` is the
    forecast horizon, passed to the HLN correction / HAC lag (h-step losses are MA(h−1)-correlated)."""
    y_rv = oof["y_rv"]
    rw_loss = qlike_series(y_rv, oof["RW_rv"])
    ar1_loss = qlike_series(y_rv, oof["AR1_rv"])
    rows = []
    for m in MODELS:
        m_loss = qlike_series(y_rv, oof[f"{m}_rv"])
        dm_rw = diebold_mariano(m_loss, rw_loss, h=dm_h) if m != "RW" else (np.nan, np.nan)
        dm_ar1 = diebold_mariano(m_loss, ar1_loss, h=dm_h) if m not in ("RW", "AR1") else (np.nan, np.nan)
        rows.append({"model": m,
                     "QLIKE": qlike(y_rv, oof[f"{m}_rv"]),
                     "R2_logRV": r2_log(oof["y_logrv"], oof[f"{m}_logrv"]),
                     "DM_vs_RW_stat": dm_rw[0], "DM_vs_RW_p": dm_rw[1],
                     "DM_vs_AR1_stat": dm_ar1[0], "DM_vs_AR1_p": dm_ar1[1]})
    df = pd.DataFrame(rows)
    rw_q = float(df.loc[df["model"] == "RW", "QLIKE"].iloc[0])
    df["QLIKE_vs_RW_pct"] = (rw_q - df["QLIKE"]) / abs(rw_q) * 100.0   # >0 = better than RW
    return df.sort_values("QLIKE").reset_index(drop=True)


def multi_horizon_summary(frame: pd.DataFrame, horizons=config.HAR_HORIZONS,
                          n_splits: int = config.WF_N_SPLITS, embargo: int = 5) -> pd.DataFrame:
    """Direct multi-horizon HAR-vs-naive table. For each h the target is the **average** RV over
    [t+1, t+h] (Corsi direct forecasting); each horizon is scored with its own horizon-aware
    Diebold-Mariano(HLN). Returns a tidy frame with a leading ``horizon`` column.

    Pre-stated expectation (not tuned): HAR's edge over AR1 is insignificant at h=1 and should
    strengthen at h=5, 22 (long memory matters more for the cumulative path than for one day)."""
    from .har import walk_forward_oof
    parts = []
    for h in horizons:
        oof = walk_forward_oof(frame, h=h, n_splits=n_splits, embargo=embargo)
        t = summarize_models(oof, dm_h=h)
        t.insert(0, "horizon", int(h))
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- A-M3: ML ladder + Gate 1
def paired_bootstrap_ci(diff, ci: float = config.GATE_BOOTSTRAP_CI, n_boot: int | None = None,
                        seed: int = config.SEED):
    """Two-sided bootstrap CI for the mean paired difference (resample paths with replacement).
    ``n_boot`` defaults to config.n_bootstrap() (QUICK vs FULL). Returns (mean, lo, hi)."""
    n_boot = config.n_bootstrap() if n_boot is None else n_boot
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    lo = float(np.percentile(means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(means, 100 * (1 + ci) / 2))
    return float(d.mean()), lo, hi


def summarize_arms(oof: pd.DataFrame, arms=("HARQ", "Ridge", "LGBM"), base: str = "HARQ") -> pd.DataFrame:
    """Headline WF-OOF table for the M3 ladder: QLIKE, R²-logRV, and Diebold-Mariano(HLN) of each
    arm's QLIKE loss vs the HAR baseline (``DM_vs_base_stat`` < 0 & p < α ⇒ arm beats HARQ)."""
    y_rv = oof["y_rv"]
    base_loss = qlike_series(y_rv, oof[f"{base}_rv"])
    rows = []
    for a in arms:
        a_loss = qlike_series(y_rv, oof[f"{a}_rv"])
        dm = diebold_mariano(a_loss, base_loss, h=1) if a != base else (np.nan, np.nan)
        rows.append({"arm": a, "QLIKE": qlike(y_rv, oof[f"{a}_rv"]),
                     "R2_logRV": r2_log(oof["y_logrv"], oof[f"{a}_logrv"]),
                     "DM_vs_base_stat": dm[0], "DM_vs_base_p": dm[1]})
    df = pd.DataFrame(rows)
    base_q = float(df.loc[df["arm"] == base, "QLIKE"].iloc[0])
    df["QLIKE_vs_base_pct"] = (base_q - df["QLIKE"]) / abs(base_q) * 100.0   # >0 = better than HARQ
    return df


def gate1_forecasting(paths, ml_arm: str, base_arm: str = "HARQ") -> dict:
    """Pre-registered Gate 1 (forecasting): does ML+exo beat the HAR baseline on QLIKE?

    Distributional over CPCV paths. **Primary** = Diebold-Mariano(HLN) significant in ML's favour
    (stat < 0 & p < DM_ALPHA) on a **majority** of paths. **Complement** = paired-bootstrap Δ-CI of
    the per-path mean QLIKE difference (base − ml; >0 ⇒ ml better) with lower bound > margin.
    Both must hold to pass; otherwise ML adds nothing and HAR stands (no tune-to-pass)."""
    sig, wins, deltas = [], [], []
    for p in paths:
        lb, lm = p[base_arm], p[ml_arm]
        stat, pval = diebold_mariano(lm, lb, h=1)            # stat < 0 ⇒ ml lower loss (better)
        sig.append(bool(np.isfinite(stat) and stat < 0 and pval < config.DM_ALPHA))
        d = float(np.nanmean(lb) - np.nanmean(lm))           # >0 ⇒ ml better
        deltas.append(d)
        wins.append(d > 0)
    deltas = np.asarray(deltas, dtype=float)
    n = int(deltas.size)
    majority_sig = float(np.mean(sig)) if n else float("nan")
    majority_win = float(np.mean(wins)) if n else float("nan")
    mean_d, lo, hi = paired_bootstrap_ci(deltas)
    primary = bool(n and majority_sig > config.GATE1_CPCV_MAJORITY)
    complement = bool(np.isfinite(lo) and lo > config.GATE_DELTA_MARGIN)
    passed = bool(primary and complement)
    return {
        "ml_arm": ml_arm, "base_arm": base_arm, "loss": "QLIKE", "n_paths": n,
        "majority_DM_sig": majority_sig, "majority_QLIKE_win": majority_win,
        "mean_delta_qlike": mean_d, "delta_ci_low": lo, "delta_ci_high": hi,
        "ci_level": config.GATE_BOOTSTRAP_CI, "dm_alpha": config.DM_ALPHA,
        "majority_threshold": config.GATE1_CPCV_MAJORITY, "delta_margin": config.GATE_DELTA_MARGIN,
        "primary_pass_DM_majority": primary, "complement_pass_bootstrap_ci": complement,
        "passed": passed,
        "verdict": (f"{ml_arm} beats HARQ — ML+exo adds forecast value" if passed
                    else f"{ml_arm} does NOT beat HARQ — ML null, HAR stands (Layer-1 deliverable)"),
    }


# --------------------------------------------------------------------------- A-M5: Gate 2 (economics)
def ann_sharpe(r, ann: int = config.DAYS_PER_YEAR) -> float:
    """Annualised Sharpe on per-period (daily) returns (Rf=0). NaN if degenerate."""
    a = np.asarray(r, dtype=float)
    a = a[np.isfinite(a)]
    sd = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    return float(np.mean(a) / sd * np.sqrt(ann)) if sd > 1e-15 else float("nan")


def _max_drawdown(equity) -> float:
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(eq)
    return float(np.max((peak - eq) / np.where(peak != 0, peak, 1.0)))


def breakeven_cost_bps(out: pd.DataFrame, sr_bh: float) -> float:
    """Round-trip bps at which the vol-timing net Sharpe falls to buy&hold's. gross = pos·ret;
    net(rt) = gross − (rt/1e4)·turnover − funding. Returns the first sweep level where it crosses
    (0 if it never beats b&h even at zero cost; the top of the sweep if it never crosses down)."""
    gross = (out["pos"] * out["ret"]).to_numpy()
    turn = out["turnover"].to_numpy()
    fund = out["funding_cost"].to_numpy()
    prev = float("inf")
    for bps in config.COST_SWEEP_RT_BPS:
        net = gross - (bps / 1e4) * turn - fund
        sr = ann_sharpe(net)
        if not np.isfinite(sr) or sr < sr_bh:
            return float(bps)
        prev = sr
    return float(config.COST_SWEEP_RT_BPS[-1])


def gate2_economics(out: pd.DataFrame) -> dict:
    """Pre-registered Gate 2 (economic kill-switch): does the vol-timing overlay beat buy&hold on
    **net-cost Sharpe**, distributionally over CPCV (majority of paths + paired-bootstrap Δ-CI)?

    Reports PSR(vt vs b&h per-period Sharpe), DSR and PBO over the small pre-registered config set
    {vol_timing, buy&hold, constant-vol-target} (N is small by design — no tuning), a breakeven-cost
    sweep, and full-sample Sharpe/MDD. Fail → trading layer null; the thesis rests on Layer-1."""
    from validation import cpcv_splits
    from metrics import psr, deflated_sharpe, prob_backtest_overfitting
    from scipy.stats import skew as _skew, kurtosis as _kurt

    rvt = out["ret_vt"].to_numpy(dtype=float)
    rbh = out["ret_bh"].to_numpy(dtype=float)
    rcvt = (float(out["pos"].mean()) * out["ret"]).to_numpy(dtype=float)   # constant-vol benchmark
    n = len(rvt)

    vt_p, bh_p = [], []
    for tr, te in cpcv_splits(n, N=config.CPCV_N, k=config.CPCV_K, horizon=1):
        vt_p.append(ann_sharpe(rvt[te]))
        bh_p.append(ann_sharpe(rbh[te]))
    vt_p, bh_p = np.asarray(vt_p), np.asarray(bh_p)
    diff = vt_p - bh_p
    majority = float(np.mean(vt_p > bh_p)) if vt_p.size else float("nan")
    mean_d, lo, hi = paired_bootstrap_ci(diff)
    passed = bool(vt_p.size and majority > config.GATE2_CPCV_MAJORITY and np.isfinite(lo)
                  and lo > config.GATE_DELTA_MARGIN)

    sr_vt, sr_bh, sr_cvt = ann_sharpe(rvt), ann_sharpe(rbh), ann_sharpe(rcvt)
    finite = rvt[np.isfinite(rvt)]
    sk = float(_skew(finite, bias=False)) if finite.size > 2 else 0.0
    ku = float(_kurt(finite, fisher=False, bias=False)) if finite.size > 2 else 3.0
    pp = lambda r: float(np.mean(r[np.isfinite(r)]) / (np.std(r[np.isfinite(r)], ddof=1) + 1e-18))
    psr_vs_bh = psr(rvt, sr_benchmark=pp(rbh))
    trial_sr = [pp(rvt), pp(rbh), pp(rcvt)]                  # per-period Sharpes of the configs tried
    dsr = deflated_sharpe(trial_sr, observed_sr=pp(rvt), T=n, skew_=sk, kurt_=ku)
    perf = np.column_stack([rvt, rbh, rcvt])
    pbo = prob_backtest_overfitting(perf)
    be_bps = breakeven_cost_bps(out, sr_bh)

    return {
        "return_basis": "daily_net_cost", "n_days": int(n), "n_cpcv_paths": int(vt_p.size),
        "n_trials": 3, "avg_exposure": float(out["pos"].mean()),
        "sharpe_vol_timing": sr_vt, "sharpe_buy_hold": sr_bh, "sharpe_const_vol": sr_cvt,
        "mdd_vol_timing": _max_drawdown(out["equity_vt"]), "mdd_buy_hold": _max_drawdown(out["equity_bh"]),
        "cpcv_majority_vt_gt_bh": majority, "mean_delta_sharpe": mean_d,
        "delta_ci_low": lo, "delta_ci_high": hi, "ci_level": config.GATE_BOOTSTRAP_CI,
        "majority_threshold": config.GATE2_CPCV_MAJORITY, "delta_margin": config.GATE_DELTA_MARGIN,
        "psr_vs_buy_hold": psr_vs_bh, "deflated_sharpe": dsr, "pbo": pbo,
        "dsr_min": config.DSR_MIN, "pbo_max": config.PBO_MAX,
        "breakeven_rt_bps": be_bps, "baseline_rt_bps": round_trip_bps_baseline(),
        "passed": passed,
        "verdict": ("vol-timing BEATS buy&hold (net-cost Sharpe, distributional)" if passed
                    else "vol-timing does NOT beat buy&hold — trading layer null, thesis rests on Layer-1"),
    }


def round_trip_bps_baseline() -> float:
    """Pre-registered baseline round-trip cost in bps (taker fee both sides + slippage both sides)."""
    return (2 * config.FEE_RATE + 2 * (config.SLIPPAGE_BPS / 1e4)) * 1e4
