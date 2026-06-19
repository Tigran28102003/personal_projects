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


def summarize_models(oof: pd.DataFrame) -> pd.DataFrame:
    """One row per model: QLIKE (variance scale), R²-logRV, QLIKE improvement vs RW (%), and
    Diebold-Mariano(HLN) significance of each model's QLIKE loss vs RW and vs AR1
    (``DM_*_stat`` < 0 with ``DM_*_p`` < 0.05 ⇒ the model is significantly better)."""
    y_rv = oof["y_rv"]
    rw_loss = qlike_series(y_rv, oof["RW_rv"])
    ar1_loss = qlike_series(y_rv, oof["AR1_rv"])
    rows = []
    for m in MODELS:
        m_loss = qlike_series(y_rv, oof[f"{m}_rv"])
        dm_rw = diebold_mariano(m_loss, rw_loss) if m != "RW" else (np.nan, np.nan)
        dm_ar1 = diebold_mariano(m_loss, ar1_loss) if m not in ("RW", "AR1") else (np.nan, np.nan)
        rows.append({"model": m,
                     "QLIKE": qlike(y_rv, oof[f"{m}_rv"]),
                     "R2_logRV": r2_log(oof["y_logrv"], oof[f"{m}_logrv"]),
                     "DM_vs_RW_stat": dm_rw[0], "DM_vs_RW_p": dm_rw[1],
                     "DM_vs_AR1_stat": dm_ar1[0], "DM_vs_AR1_p": dm_ar1[1]})
    df = pd.DataFrame(rows)
    rw_q = float(df.loc[df["model"] == "RW", "QLIKE"].iloc[0])
    df["QLIKE_vs_RW_pct"] = (rw_q - df["QLIKE"]) / abs(rw_q) * 100.0   # >0 = better than RW
    return df.sort_values("QLIKE").reset_index(drop=True)
