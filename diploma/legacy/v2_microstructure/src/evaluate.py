"""Evaluation: CPCV path economics and the pre-registered M3 distributional gate.

The M3 gate compares the flow arm against the close-price control arm on **net-of-cost Sharpe
across CPCV paths** — not a single point estimate, and **not** on raw overlapping bar returns.
Because triple-barrier bets span ``[entry, t1]`` and adjacent bars' forward returns overlap,
a Sharpe/Δ-CI computed on raw bar returns overstates significance. The gate therefore runs on
**de-overlapped, one-return-per-bet trade-level returns**: per CPCV test path we greedily select
non-overlapping bets (next entry ≥ previous ``t1``) and form
``r_i = side_i·(P_{t1_i}/P_{entry_i} − 1) − round_trip_cost``. Flow passes only if it wins on a
MAJORITY of paired paths AND a paired-bootstrap CI of the per-path Sharpe difference excludes the
pre-registered margin. A noisy point win does not qualify; failure is a kill-switch to backlog B.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .labeling import Labels
from .primary_model import make_model, fit_model, predict_side_proba

from validation import cpcv_splits  # noqa: E402
from backtest import round_trip_cost  # noqa: E402

_EPS = 1e-12


def forward_return(close: pd.Series) -> pd.Series:
    """1-bar-ahead simple return at each bar t: close_{t+1}/close_t - 1 (NaN on the last bar)."""
    return close.pct_change().shift(-1)


def default_cost() -> float:
    return round_trip_cost(config.FEE_RATE, config.SLIPPAGE_BPS)


def cpcv_split_list(n: int, H: int):
    """Deterministic CPCV split list — generate ONCE and reuse for both arms so paths pair up."""
    return list(cpcv_splits(n, N=config.CPCV_N, k=config.CPCV_K, horizon=H))


# --------------------------------------------------------------------------- trade-level returns
def build_trade_info(labels: Labels, close: pd.Series) -> pd.DataFrame:
    """Per-event entry/exit positions and prices (aligned to the label index).

    Columns: ``entry_pos``, ``t1_pos`` (integer positions into ``close``), ``entry_price``,
    ``t1_price``. Used to form de-overlapped trade-level returns for the gate.
    """
    f = labels.frame
    pos = pd.Series(np.arange(len(close)), index=close.index)
    return pd.DataFrame({
        "entry_pos": pos.reindex(f.index).to_numpy(),
        "t1_pos": pos.reindex(f["t1"]).to_numpy(),
        "entry_price": close.reindex(f.index).to_numpy(),
        "t1_price": close.reindex(f["t1"]).to_numpy(),
    }, index=f.index)


def de_overlapped_trade_returns(entry_pos, t1_pos, side, entry_price, t1_price,
                                cost_rt: float):
    """Greedy non-overlapping bets → one trade-level return per bet.

    Walk events in entry order; take a bet only if its side is ±1 and it starts at/after the
    previous taken bet's exit (``entry ≥ cursor``); set the cursor to its ``t1``. Each bet's
    net return is ``side·(P_t1/P_entry − 1) − cost_rt`` (one round-trip cost per bet).
    Returns ``(rets, holds)`` as float arrays (holds in bars).
    """
    ep = np.asarray(entry_pos); tp = np.asarray(t1_pos)
    sd = np.asarray(side); pe = np.asarray(entry_price); p1 = np.asarray(t1_price)
    order = np.argsort(ep, kind="stable")
    cursor = -1
    rets, holds = [], []
    for i in order:
        if sd[i] == 0 or not np.isfinite(pe[i]) or not np.isfinite(p1[i]):
            continue
        if ep[i] < cursor:                       # overlaps a bet already taken
            continue
        rets.append(sd[i] * (p1[i] / pe[i] - 1.0) - cost_rt)
        holds.append(tp[i] - ep[i])
        cursor = tp[i]                           # next bet must start at/after this exit
    return np.asarray(rets, dtype=float), np.asarray(holds, dtype=float)


def trade_sharpe(rets: np.ndarray, holds: np.ndarray,
                 ann_bars: int = config.BARS_PER_YEAR) -> float:
    """Sharpe of de-overlapped trade returns, annualised by √(bets_per_year).

    ``bets_per_year = ann_bars / mean_holding_bars`` (≈ IID across non-overlapping bets). Both
    arms use their own bet frequency, which is the economically honest per-strategy annualisation.
    """
    if rets.size < 2:
        return float("nan")
    sd = rets.std(ddof=1)
    if sd <= _EPS:
        return float("nan")
    bets_per_year = ann_bars / max(holds.mean(), 1.0)
    return float(rets.mean() / sd * np.sqrt(bets_per_year))


# --------------------------------------------------------------------------- per-arm CPCV paths
def arm_path_sharpes(X: pd.DataFrame, y: pd.Series, w: pd.Series, trade_info: pd.DataFrame,
                     cols, splits, model_name: str = "lgbm",
                     cost_rt: float | None = None, seed: int = config.SEED) -> np.ndarray:
    """Per-CPCV-path Sharpe on **de-overlapped trade-level returns** for one feature arm.

    Each path: fit Stage-1 on train, predict side on the test events, keep ±1, select
    non-overlapping bets, form trade returns held entry→barrier (``t1``), and annualise. The
    selection uses the arm's own predicted side (its own bet set) — fair, identical folds/budget.
    """
    cost_rt = default_cost() if cost_rt is None else cost_rt
    Xc = X[cols]
    ep = trade_info["entry_pos"].to_numpy()
    tp = trade_info["t1_pos"].to_numpy()
    pe = trade_info["entry_price"].to_numpy()
    p1 = trade_info["t1_price"].to_numpy()
    sharpes = []
    for tr, te in splits:
        m = fit_model(make_model(model_name, seed), Xc.iloc[tr], y.iloc[tr], w.iloc[tr])
        side, _, _ = predict_side_proba(m, Xc.iloc[te])
        rets, holds = de_overlapped_trade_returns(ep[te], tp[te], side, pe[te], p1[te], cost_rt)
        sharpes.append(trade_sharpe(rets, holds))
    return np.asarray(sharpes, dtype=float)


# --------------------------------------------------------------------------- gate
def paired_bootstrap_ci(diff: np.ndarray, ci: float = config.GATE_BOOTSTRAP_CI,
                        n_boot: int | None = None, seed: int = config.SEED):
    """Two-sided bootstrap CI for the mean paired difference (resample paths with replacement).

    ``n_boot`` defaults to ``config.n_bootstrap()`` (QUICK vs FULL).
    """
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


def distributional_gate(flow_sharpes: np.ndarray, control_sharpes: np.ndarray) -> dict:
    """Pre-registered M3 gate on de-overlapped trade-return Sharpes. Flow passes iff CPCV-majority
    win AND bootstrap Δ-CI lower bound > margin."""
    flow = np.asarray(flow_sharpes, dtype=float)
    ctrl = np.asarray(control_sharpes, dtype=float)
    mask = np.isfinite(flow) & np.isfinite(ctrl)
    flow, ctrl = flow[mask], ctrl[mask]
    diff = flow - ctrl
    majority = float(np.mean(flow > ctrl)) if flow.size else float("nan")
    mean_d, lo, hi = paired_bootstrap_ci(diff)
    passed = bool(flow.size and majority > config.GATE_CPCV_MAJORITY
                  and lo > config.GATE_DELTA_MARGIN)
    return {
        "return_basis": "de_overlapped_trade_level",
        "n_paths": int(flow.size),
        "flow_sharpe_mean": float(np.mean(flow)) if flow.size else float("nan"),
        "flow_sharpe_median": float(np.median(flow)) if flow.size else float("nan"),
        "control_sharpe_mean": float(np.mean(ctrl)) if ctrl.size else float("nan"),
        "control_sharpe_median": float(np.median(ctrl)) if ctrl.size else float("nan"),
        "majority_flow_gt_control": majority,
        "mean_delta_sharpe": mean_d,
        "delta_ci_low": lo,
        "delta_ci_high": hi,
        "ci_level": config.GATE_BOOTSTRAP_CI,
        "majority_threshold": config.GATE_CPCV_MAJORITY,
        "delta_margin": config.GATE_DELTA_MARGIN,
        "passed": passed,
        "verdict": ("FLOW PASSES — continue C" if passed
                    else "FLOW FAILS gate — kill-switch to backlog B"),
    }
