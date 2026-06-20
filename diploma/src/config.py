"""Global configuration, paths, reproducibility (single source of truth).

Everything that controls a run lives here so the notebook stays a thin orchestrator
and a run is byte-reproducible. Two design rules enforced by this module:

* ``DATA_END`` is a **fixed cutoff** (never ``today``) — reproducibility.
* ``THRESHOLD`` is the **label definition**, never a tuned hyperparameter — it must
  never enter an Optuna search space (different thresholds give incomparable base
  rates → label snooping). See ``labeling.make_target``.

``QUICK_MODE`` is a *real* toggle: it shrinks trials / history / fold count so the
notebook runs end-to-end in minutes for a smoke test, then flips to the full grid for
the deliverable. Accessor functions (``n_trials()``, ``wf_splits()`` …) read the
current value of ``QUICK_MODE`` at call time, so flipping the flag in the notebook
takes effect immediately.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np

# ───────────────────────────── reproducibility ─────────────────────────────
SEED = 42

# ─────────────────────────────── data window ───────────────────────────────
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
DATA_START = "2017-08-17"   # BTCUSDT spot inception on Binance
DATA_END = "2026-06-01"     # FIXED cutoff (never "today") — reproducibility

# ───────────────────────────── target / horizon ────────────────────────────
HORIZON = 24                # forecast horizon in bars (hours)
THRESHOLD = 0.02            # ±2% — LABEL DEFINITION, never an Optuna knob (R10: simple return)
TARGET_TYPES = ("pointwise", "triple_barrier", "vol_scaled")

# vol-scaled label: thr_t = k * sigma_t, sigma_t = causal EWMA vol of returns,
# shifted by 1 (no look-ahead) and scaled by sqrt(HORIZON). NOT rolling-std
# (rolling-std has a ghosting artefact in the label).
VOL_SCALED_K = (1.0, 1.5, 2.0)   # pre-registered set; each k is a separate experiment
EWMA_VOL_SPAN = 72               # span for the causal EWMA volatility

# ──────────────────────────── validation / split ───────────────────────────
SPLIT = (0.70, 0.15, 0.15)       # chronological train/val/test; test opened once
PURGE = HORIZON                  # R1: purge EXACTLY HORIZON train bars before each test block
EMBARGO = 24                     # extra buffer (bars) after the purge
assert PURGE >= HORIZON, "PURGE must be >= HORIZON (over-purge by a bar beats under-purge)"

# adaptation scaffold (R4) — PRE-REGISTERED ablation knobs, NOT Optuna parameters
WINDOW_MODE = "expanding"        # {"expanding", "rolling"}
TRAIN_WINDOW: Optional[int] = None   # bars; only used when WINDOW_MODE == "rolling"
RECENCY_HALFLIFE: Optional[int] = None  # bars; None disables recency weighting
RETRAIN_EVERY = HORIZON          # operational walk-forward refit cadence (bars)

# ───────────────────────────── annualisation ───────────────────────────────
BARS_PER_YEAR = 24 * 365         # hourly bars per year (for bar-level Sharpe only)
BETS_PER_YEAR = 365              # ~one non-overlapping 24h bet per day (trade-level Sharpe)

# ─────────────────────────────── backtest cfg ──────────────────────────────
BACKTEST_CFG = {
    "fee_bps": 7.0,              # round-trip taker proxy in bps (sweep 5-10)
    "slippage_bps": 0.0,         # folded into fee_bps; kept explicit for sweeps
    "entry": "next_open",        # signal on close[t] -> enter at open[t+1]
    "hold": HORIZON,             # hold horizon bars (or barrier exit for triple_barrier)
    "nonoverlap_step": HORIZON,  # one bet per HORIZON bars per phase
    "phase_average": True,       # R8: average headline metrics over all 24 entry phases
    "allow_short": True,
    "funding": False,            # optional perp funding carry
    "vol_target_bh": 0.60,       # annualised vol target for the vol-targeted B&H benchmark
}

# ───────────────────────────── QUICK_MODE toggle ───────────────────────────
QUICK_MODE = False

_QUICK = dict(n_trials=10, history_months=18, wf_splits=3, cpcv_N=6, cpcv_k=2,
              n_estimators_cap=200)
_FULL = dict(n_trials=75, history_months=None, wf_splits=8, cpcv_N=10, cpcv_k=2,
             n_estimators_cap=600)   # потолок; реальное число деревьев решает early stopping


def settings() -> dict:
    """Current QUICK/FULL settings dict (read at call time so the flag is live)."""
    return dict(_QUICK if QUICK_MODE else _FULL)


def n_trials() -> int:
    """Optuna trials per setup (per-setup; DSR n_trials multiplies by the full grid)."""
    return settings()["n_trials"]


def wf_splits() -> int:
    """Number of purged walk-forward folds."""
    return settings()["wf_splits"]


def history_months() -> Optional[int]:
    """How many trailing months of data to keep (None == full history)."""
    return settings()["history_months"]


def n_estimators_cap() -> int:
    """Upper bound on LightGBM n_estimators in the search space."""
    return settings()["n_estimators_cap"]


# ─────────────────────────────────── paths ─────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
PICTURES = ROOT / "pictures"
LEGACY = ROOT / "legacy" / "ACTUAL_VERSION"

FEATURES_CACHE = PROCESSED / "features_hourly.csv"
FEATURES_PARQUET = PROCESSED / "features_hourly.parquet"
RAW_SNAPSHOT = RAW / "btc_ohlcv_hourly.parquet"
RAW_MANIFEST = RAW / "snapshot_manifest.json"

for _d in (DATA, RAW, PROCESSED, PICTURES):
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────── LightGBM determinism ──────────────────────────
LGBM_DETERMINISTIC = dict(
    deterministic=True,
    force_row_wise=True,
    num_threads=1,
    seed=SEED,
    bagging_seed=SEED,
    feature_fraction_seed=SEED,
    data_random_seed=SEED,
    verbosity=-1,
)


def set_determinism(seed: int = SEED) -> None:
    """Fix every RNG used by the pipeline (target: a byte-reproducible run).

    Seeds python ``random``, ``numpy``, ``PYTHONHASHSEED``; LightGBM determinism is
    applied per-estimator via :data:`LGBM_DETERMINISTIC`. XGBoost/torch seeds set if
    those libraries are importable (they are optional for the core A/B/C comparison).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # optional
        import torch
        torch.manual_seed(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001 — torch is optional
        pass


# ─────────────────────────── compute profile (W1) ──────────────────────────
# Hardware detection / thread budget (parallel_fits × lgbm_threads ≤ n_physical).
# Lazy import keeps `config` light and avoids a hard psutil/torch dependency.
def _detect_compute() -> dict:
    try:
        from . import compute
    except ImportError:
        import compute
    return compute.detect()


COMPUTE_CFG = _detect_compute()
