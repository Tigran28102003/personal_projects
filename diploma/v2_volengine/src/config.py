"""Single source of truth for paths and pre-registered parameters — direction A (Vol-Engine).

`v2_volengine` is a standalone package at the git repo root, a sibling of `v2_microstructure`
(which it reuses for the perp-era exo snapshot) and of `ACTUAL_VERSION` (the legacy library,
attached via an explicit `sys.path` entry below). All knobs that matter for the pre-registered
protocol (RV estimator, HAR set, CPCV, holdout cut, Gate-1/Gate-2 margins, deploy rule) are
fixed here BEFORE any run, so results cannot be tuned by quietly editing thresholds.

Three layers (deliberately decoupled): L1 RV/HAR forecasting (guaranteed deliverable),
L2 vol-timing overlay (may fail its economic gate), L3 DVOL/VRP (backlog, never claimed valid).
"""

from __future__ import annotations

import sys
from pathlib import Path

# --------------------------------------------------------------------------- paths
SRC_DIR = Path(__file__).resolve().parent
PKG_ROOT = SRC_DIR.parent                       # diploma/v2_volengine  (standalone package)
REPO_ROOT = PKG_ROOT.parent                     # diploma/  (git repo root)
LEGACY_LIB = REPO_ROOT / "ACTUAL_VERSION"       # sibling legacy library (validation/metrics/backtest/...)
V2_MICRO = REPO_ROOT / "v2_microstructure"      # sibling v2 package (perp-era exo snapshot reuse)

FIGURES_DIR = PKG_ROOT / "figures"              # ALL plots, via this constant only
DATA_DIR = PKG_ROOT / "data"                    # frozen spot-OHLC snapshot + its manifest
REPORTS_DIR = PKG_ROOT / "reports"              # per-run artifact contracts

SPOT_SNAPSHOT_PATH = DATA_DIR / "spot_ohlc_snapshot.parquet"
SPOT_MANIFEST = DATA_DIR / "spot_ohlc_manifest.json"
ENV_LOCKFILE = DATA_DIR / "env_lock.txt"

# Attach the sibling legacy library + the v2 package explicitly; fail loud on a wrong layout.
for _p in (LEGACY_LIB, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
assert (LEGACY_LIB / "free_features.py").exists(), (
    f"legacy library not found in {LEGACY_LIB} — v2_volengine expects ACTUAL_VERSION as a sibling")
assert (V2_MICRO / "src" / "data_snapshot.py").exists(), (
    f"v2_microstructure not found in {V2_MICRO} — needed for the perp-era exo snapshot")


def ensure_dirs() -> None:
    for d in (FIGURES_DIR, DATA_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- reproducibility
SEED = 42

# --------------------------------------------------------------------------- market / data
SYMBOL = "BTCUSDT"
SPOT_INTERVAL = "1h"                             # intraday bars for RV
SPOT_START = "2017-08-01"                        # Binance spot BTCUSDT ≈ 2017-08-17 (R1: long history)
BARS_PER_DAY = 24                                # expected hourly bars per UTC day (audited in M0)
DAYS_PER_YEAR = 365                              # annualisation for daily-RV / vol-timing Sharpe

# Perp-era exo (funding/basis/on-chain) reused from the frozen v2_microstructure snapshot
# (≈2019-09→); read via v2_microstructure.src.data_snapshot.load_snapshot().

# --------------------------------------------------------------------------- RV / target (L1)
RV_ESTIMATOR = "close"                           # "close" (Σ Δlogclose²) | "range" (Σ Garman-Klass/RS)
RV_RANGE_METHOD = "garman_klass"                 # if RV_ESTIMATOR == "range": garman_klass | rogers_satchell
TARGET_TRANSFORM = "log"                         # "log" (primary) | "sqrt"
RV_FLOOR = 1e-12                                 # numerical floor before log

# --------------------------------------------------------------------------- HAR baselines (L1)
HAR_WINDOWS = (1, 5, 22)                         # Corsi daily / weekly / monthly
HAR_VARIANTS = ("HAR", "HARJ", "HARQ")          # HARQ central (R2)

# --------------------------------------------------------------------------- samples (R1 discipline)
# HAR baseline + Gate 2 on the FULL spot history; Gate 1 (ML+exo vs HAR) on the perp-era subsample
# (exo only exists there), both arms on the same subsample.
PERP_ERA_START = "2019-09-10"
HOLDOUT_CUT = "2025-06-01"                       # last ~12 months sealed; unsealed once at the end

# --------------------------------------------------------------------------- validation
CPCV_N = 6
CPCV_K = 2                                       # C(6,2)=15 paths
WF_N_SPLITS = 5
WF_MIN_TRAIN_FRAC = 0.5

# --------------------------------------------------------------------------- Gate 1 (forecasting)
# ML+exo beats HAR family on QLIKE: Diebold-Mariano (HLN) primary + paired-bootstrap Δ-CI complement,
# distributional over CPCV paths.
DM_ALPHA = 0.05                                  # Diebold-Mariano significance (HLN-corrected)
GATE1_CPCV_MAJORITY = 0.5                        # ML must win QLIKE on > 50% of paths

# --------------------------------------------------------------------------- Gate 2 (economics)
# vol-timing net-cost Sharpe > buy&hold, distributional over CPCV paths.
GATE2_CPCV_MAJORITY = 0.5
GATE_BOOTSTRAP_CI = 0.95
GATE_DELTA_MARGIN = 0.0                          # paired Δ-CI lower bound must exceed this
GATE_N_BOOTSTRAP = 10_000

# vol-timing perp costs (taker baseline; funding added on the holding leg)
FEE_RATE = 0.0005
SLIPPAGE_BPS = 2.0
COST_SWEEP_RT_BPS = tuple(range(0, 41, 2))
VOL_TARGET_ANN = 0.60                            # annualised target vol for the timing overlay
VOL_TIMING_CAP = 1.0                             # max gross exposure (long-flat 0..cap in L2; no short)

# --------------------------------------------------------------------------- deployment rule
DSR_MIN = 0.95
PBO_MAX = 0.5

# --------------------------------------------------------------------------- QUICK vs FULL
QUICK_MODE = True
QUICK_N_BOOTSTRAP = 2_000
QUICK_LGBM_N_EST = 120
FULL_LGBM_N_EST = 400


def n_bootstrap():
    return QUICK_N_BOOTSTRAP if QUICK_MODE else GATE_N_BOOTSTRAP


def lgbm_n_estimators():
    return QUICK_LGBM_N_EST if QUICK_MODE else FULL_LGBM_N_EST
