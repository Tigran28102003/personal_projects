"""Vol-scaled triple-barrier labeling (M1) — the new target.

Replaces one-step ``r_t`` / directional-accuracy with a learnable, path-dependent,
vol-normalised question: starting at bar ``t``, does price travel ``+k·sigma_t`` before
``-k·sigma_t`` within ``H`` bars?  Label ``+1`` (upper / TP first), ``-1`` (lower / SL
first), ``0`` (neither — the vertical barrier, an honest "no actionable move").

Design choices (pre-registered in config.py):
* ``sigma_t`` is a STRICTLY TRAILING EWMA std of log-returns (known at the entry bar t; no
  future bar enters the estimate). Audited by the t-1 tripwire.
* Barriers and path are measured in LOG-return space for symmetry; the realised log-return
  at the touch ``ret_log`` drives the magnitude weight.
* Labels overlap (each event spans ``[t, t1]``), so we compute AFML average-uniqueness; the
  sum of uniqueness is the EFFECTIVE number of independent bets used everywhere downstream.
* The unified ``sample_weight`` = average_uniqueness · |ret_log| · class_balance_factor is
  defined ONCE here so magnitude / uniqueness / class-imbalance are never double-counted.

Nothing here looks past ``t+H``; the last ``H`` rows get no label (NaN).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- volatility
def trailing_sigma(close: pd.Series, span: int = config.SIGMA_SPAN_DEFAULT) -> pd.Series:
    """Strictly trailing EWMA std of 1-bar log-returns (causal: value at t uses returns ≤ t)."""
    logret = np.log(close).diff()
    sigma = logret.ewm(span=span, min_periods=span).std()
    sigma.name = "sigma"
    return sigma


# --------------------------------------------------------------------------- labeling
@dataclass
class Labels:
    """Triple-barrier output, indexed by entry timestamp (valid entries only).

    frame columns: label {+1,0,-1}, t1 (touch timestamp), ret_log (log-return at t1),
    bars_held, sigma, up_thr, dn_thr, avg_uniqueness, sample_weight.
    """
    frame: pd.DataFrame

    @property
    def label(self) -> pd.Series:
        return self.frame["label"]

    @property
    def sample_weight(self) -> pd.Series:
        return self.frame["sample_weight"]


def triple_barrier_labels(close: pd.Series, k: float = config.K_DEFAULT,
                          H: int = config.H_DEFAULT,
                          sigma_span: int = config.SIGMA_SPAN_DEFAULT) -> Labels:
    """First-touch vol-scaled triple-barrier labels for every valid entry bar.

    Returns a :class:`Labels` whose frame is indexed by the entry timestamp. An entry is
    valid iff ``sigma_t`` is defined and a full ``H``-bar forward window exists.
    """
    close = close.astype(float)
    idx = close.index
    n = len(close)
    logc = np.log(close.to_numpy())
    sigma = trailing_sigma(close, sigma_span)
    thr = (k * sigma).to_numpy()  # barrier half-width in log-return units

    # Forward log-return path L[t, s-1] = log(close[t+s]) - log(close[t]), s = 1..H.
    L = np.full((n, H), np.nan)
    for s in range(1, H + 1):
        L[: n - s, s - 1] = logc[s:] - logc[: n - s]

    up = L >= thr[:, None]
    dn = L <= -thr[:, None]
    first_up = np.where(up.any(axis=1), up.argmax(axis=1), H)   # H == no touch
    first_dn = np.where(dn.any(axis=1), dn.argmax(axis=1), H)

    label = np.zeros(n, dtype=float)
    hit_off = np.full(n, H - 1, dtype=int)                      # vertical → last bar of window
    up_first = first_up < first_dn
    dn_first = first_dn < first_up
    label[up_first] = 1.0
    label[dn_first] = -1.0
    hit_off[up_first] = first_up[up_first]
    hit_off[dn_first] = first_dn[dn_first]

    bars_held = hit_off + 1
    ret_log = L[np.arange(n), hit_off]

    pos = np.arange(n)
    valid = (~np.isnan(thr)) & (pos <= n - 1 - H)
    vidx = pos[valid]
    t1_pos = vidx + bars_held[valid]

    frame = pd.DataFrame({
        "label": label[valid].astype(int),
        "t1": idx[t1_pos],
        "ret_log": ret_log[valid],
        "bars_held": bars_held[valid].astype(int),
        "sigma": sigma.to_numpy()[valid],
        "up_thr": thr[valid],
        "dn_thr": -thr[valid],
    }, index=idx[vidx])
    frame.index.name = idx.name or "timestamp"

    avg_uniq = _average_uniqueness(vidx, t1_pos, n)
    frame["avg_uniqueness"] = avg_uniq
    frame["sample_weight"] = _unified_sample_weight(frame["label"].to_numpy(),
                                                     frame["ret_log"].to_numpy(), avg_uniq)
    return Labels(frame=frame)


def _average_uniqueness(entry_pos: np.ndarray, t1_pos: np.ndarray, n: int) -> np.ndarray:
    """AFML average uniqueness per event over its [entry, t1] span (overlap down-weighting)."""
    conc = np.zeros(n)
    for p, q in zip(entry_pos, t1_pos):
        conc[p:q + 1] += 1.0
    inv = np.divide(1.0, conc, out=np.zeros_like(conc), where=conc > 0)
    out = np.empty(len(entry_pos))
    for i, (p, q) in enumerate(zip(entry_pos, t1_pos)):
        out[i] = inv[p:q + 1].mean()
    return out


def _unified_sample_weight(label: np.ndarray, ret_log: np.ndarray,
                           avg_uniq: np.ndarray) -> np.ndarray:
    """sample_weight = average_uniqueness · |ret_log| · class_balance_factor (mean-normalised).

    Class imbalance enters ONLY through ``class_balance_factor`` (sklearn 'balanced' style),
    so it is never applied a second time as a model class_weight.
    """
    classes, counts = np.unique(label, return_counts=True)
    n, n_cls = len(label), len(classes)
    freq = {c: cnt for c, cnt in zip(classes, counts)}
    cbf = np.array([n / (n_cls * freq[c]) for c in label])
    w = avg_uniq * np.abs(ret_log) * cbf
    mean = w.mean()
    return w / mean if mean > 0 else w


# --------------------------------------------------------------------------- sealed holdout
def seal_holdout(index: pd.Index, cut: str = config.HOLDOUT_CUT):
    """Return (dev_mask, holdout_mask) boolean Series. Holdout (index >= cut) is sealed until M7."""
    cut_ts = pd.Timestamp(cut)
    dev = pd.Series(index < cut_ts, index=index, name="dev")
    holdout = pd.Series(index >= cut_ts, index=index, name="holdout")
    return dev, holdout


# --------------------------------------------------------------------------- reporting
def label_report(labels: Labels, cost_rt: float | None = None) -> dict:
    """Label balance, trade frequency, effective sample size, and a barrier-vs-cost check.

    Reported EARLY (M1): if verticals (0) dominate, there are too few trades, caught before
    any modelling. ``cost_rt`` defaults to the configured round-trip cost.
    """
    from backtest import round_trip_cost
    if cost_rt is None:
        cost_rt = round_trip_cost(config.FEE_RATE, config.SLIPPAGE_BPS)

    f = labels.frame
    n = len(f)
    counts = f["label"].value_counts().reindex([1, 0, -1]).fillna(0).astype(int)
    fracs = (counts / n).round(4)
    sum_uniq = float(f["avg_uniqueness"].sum())
    # Barrier half-width in return space: exp(k·sigma) - 1 ≈ k·sigma. Compare to round-trip cost.
    med_barrier = float(np.expm1(f["up_thr"].median()))
    return {
        "k": None, "H": None,  # filled by the grid driver
        "n_valid": int(n),
        "counts": {"+1": int(counts[1]), "0": int(counts[0]), "-1": int(counts[-1])},
        "fractions": {"+1": float(fracs[1]), "0": float(fracs[0]), "-1": float(fracs[-1])},
        "trade_frequency": float((counts[1] + counts[-1]) / n),
        "mean_bars_held": float(f["bars_held"].mean()),
        "median_bars_held": float(f["bars_held"].median()),
        "sum_avg_uniqueness": sum_uniq,
        "effective_frac": float(sum_uniq / n) if n else float("nan"),
        "median_barrier_return": med_barrier,
        "round_trip_cost": float(cost_rt),
        "barrier_to_cost_ratio": float(med_barrier / cost_rt) if cost_rt else float("inf"),
    }
