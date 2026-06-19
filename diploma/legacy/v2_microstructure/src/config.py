"""Single source of truth for paths and pre-registered parameters (redesign / direction C).

Every path used anywhere in `v2_microstructure` comes from here — no hard-coded paths in
modules or scripts. Every modelling knob that matters for the pre-registered protocol
(barrier grid, costs, CPCV, sealed-holdout cut, M3 gate margin, deployment rule) is fixed
here *before* any run, so results cannot be tuned by quietly editing thresholds.

`v2_microstructure` is a **standalone package at the git repo root** (`diploma/`); the existing
pipeline modules (`free_features`, `validation`, `metrics`, `backtest`, `walk_forward`,
`meta_labeling`, `fracdiff`, ...) live in the **sibling** ``ACTUAL_VERSION/`` legacy library. We add
that sibling (``LEGACY_LIB``) to ``sys.path`` here so `from free_features import ...` works regardless
of the current working directory, and we ``assert`` it exists so a wrong layout fails loud immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --------------------------------------------------------------------------- paths
SRC_DIR = Path(__file__).resolve().parent
PKG_ROOT = SRC_DIR.parent                      # diploma/v2_microstructure  (standalone package)
REPO_ROOT = PKG_ROOT.parent                    # diploma/  (git repo root)
LEGACY_LIB = REPO_ROOT / "ACTUAL_VERSION"      # sibling legacy library (free_features, validation, ...)

FIGURES_DIR = PKG_ROOT / "figures"             # ALL plots go here, via this constant only
DATA_DIR = PKG_ROOT / "data"                   # frozen snapshot + its manifest
REPORTS_DIR = PKG_ROOT / "reports"             # per-run artifact contracts

SNAPSHOT_PATH = DATA_DIR / "features_snapshot.parquet"
SNAPSHOT_MANIFEST = DATA_DIR / "snapshot_manifest.json"
ENV_LOCKFILE = DATA_DIR / "env_lock.txt"       # pip freeze, hashed into every run manifest

# Attach the sibling legacy library explicitly (no path guessing); fail loud on a wrong layout.
if str(LEGACY_LIB) not in sys.path:
    sys.path.insert(0, str(LEGACY_LIB))
assert (LEGACY_LIB / "free_features.py").exists(), (
    f"legacy library not found in {LEGACY_LIB} — v2_microstructure expects ACTUAL_VERSION as a "
    "sibling of the package at the repo root"
)


def ensure_dirs() -> None:
    """Create the artifact directories if missing (idempotent)."""
    for d in (FIGURES_DIR, DATA_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- reproducibility
SEED = 42

# --------------------------------------------------------------------------- market / data
FREQ = "hourly"
INTERVAL = "1h"                                # Binance kline interval for FREQ
SYMBOL = "BTCUSDT"                             # perp == spot symbol on Binance
# Trading instrument & PnL are on the PERPETUAL (funding/basis/OI + native short).
# Order-flow proxies are computed from PERP klines; basis = perp - spot (spot pulled too).
ORDERFLOW_MARKET = "futures"                   # 'futures' == USDⓈ-M perp on Binance
BARS_PER_YEAR = 24 * 365                       # annualisation factor for hourly Sharpe (8760)

# Earliest available history is bounded by the perp (≈2019-09-08); the effective snapshot
# range is the intersection of all sources, computed live by data_snapshot.audit_sources().

# --------------------------------------------------------------------------- labeling grid
# Pre-registered triple-barrier grid. Every (k, H, sigma_span) combination is a TRIAL and
# is counted in the DSR deflation N — logged in the config-trial registry.
K_GRID = (1.0, 1.5, 2.0)                       # TP/SL barrier in multiples of trailing sigma
H_GRID = (12, 24)                              # vertical barrier, in bars (hours)
SIGMA_SPAN_GRID = (24, 48, 72)                 # trailing EWMA span for sigma_t (bars)
# Defaults used outside a grid sweep:
K_DEFAULT = 1.5
H_DEFAULT = 24
SIGMA_SPAN_DEFAULT = 48

# --------------------------------------------------------------------------- costs (perp taker)
# round_trip_cost(FEE_RATE, SLIPPAGE_BPS) = 2*FEE_RATE + 2*(SLIPPAGE_BPS/1e4)
#   = 2*0.0005 + 2*0.0002 = 0.0014 = 14 bp  (perp taker baseline).
FEE_RATE = 0.0005                              # 0.05% perp taker, one side
SLIPPAGE_BPS = 2.0                             # one side, basis points
ASSUME_TAKER = True
COST_SWEEP_RT_BPS = tuple(range(0, 41, 2))     # breakeven sweep: round-trip cost 0..40 bp

# --------------------------------------------------------------------------- validation
CPCV_N = 6                                     # CPCV groups
CPCV_K = 2                                     # test groups per split  -> C(6,2)=15 paths
WF_N_SPLITS = 5                                # walk-forward OOF folds
WF_MIN_TRAIN_FRAC = 0.5

# --------------------------------------------------------------------------- sealed holdout
# Last ~12 months are SEALED at M1 and unsealed exactly once at M7. Fixed now.
HOLDOUT_CUT = "2025-06-01"                      # dev := index < cut ; holdout := index >= cut

# --------------------------------------------------------------------------- M3 distributional gate
# Flow core passes iff: (a) net-cost Sharpe beats the control on a MAJORITY of CPCV paths,
# AND (b) a paired bootstrap CI of Δ(net-cost Sharpe) excludes 0 (lower bound > MARGIN).
GATE_CPCV_MAJORITY = 0.5                        # strictly > 50% of paths
GATE_BOOTSTRAP_CI = 0.95                        # two-sided CI level for paired Δ
GATE_DELTA_MARGIN = 0.0                         # CI lower bound must exceed this (Sharpe units)
GATE_N_BOOTSTRAP = 10_000

# --------------------------------------------------------------------------- deployment rule
DSR_MIN = 0.95
PBO_MAX = 0.5

# --------------------------------------------------------------------------- QUICK vs FULL
# QUICK_MODE shrinks the grid / model size so a notebook Restart-&-Run-All is fast. The FULL
# path (QUICK_MODE = False) is unchanged. Use the accessors below so callers never branch.
QUICK_MODE = True
QUICK_K_GRID = (1.5,)
QUICK_H_GRID = (24,)
QUICK_N_BOOTSTRAP = 2_000
QUICK_LGBM_N_EST = 120
FULL_LGBM_N_EST = 300
QUICK_GATE_MODEL = "lgbm"


def k_grid():
    return QUICK_K_GRID if QUICK_MODE else K_GRID


def h_grid():
    return QUICK_H_GRID if QUICK_MODE else H_GRID


def n_bootstrap():
    return QUICK_N_BOOTSTRAP if QUICK_MODE else GATE_N_BOOTSTRAP


def lgbm_n_estimators():
    return QUICK_LGBM_N_EST if QUICK_MODE else FULL_LGBM_N_EST
