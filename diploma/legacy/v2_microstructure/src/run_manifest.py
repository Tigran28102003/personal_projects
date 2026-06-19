"""Run-contract utilities (M0 skeleton; extended at M5).

Every experiment emits ``reports/<run_id>/run_manifest.json`` plus the artifact set described
in the plan. This module provides the reproducibility primitives:

* ``new_run_id`` / ``run_dir`` — stable per-run directory under REPORTS_DIR.
* ``env_lockfile`` — freeze the environment (pip freeze) and hash it.
* ``ConfigTrialRegistry`` — record EVERY config tried (k, H, sigma_span, feature set, model)
  so the DSR deflation N is the true number of trials, not the number of CPCV paths.
* ``build_run_manifest`` / ``write_run_manifest`` — assemble + persist the manifest.
* ``delta_to_previous`` — metric delta vs the previous run (full impl at M5; stub here).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")


def run_dir(run_id: str) -> Path:
    d = config.REPORTS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def env_lockfile() -> dict:
    """Write `pip freeze` to ENV_LOCKFILE and return {path, sha256}. Reproducibility pin."""
    config.ensure_dirs()
    try:
        frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except (subprocess.SubprocessError, OSError) as e:  # never crash a run over this
        frozen = f"# pip freeze failed: {e}\n"
    config.ENV_LOCKFILE.write_text(frozen)
    return {"path": str(config.ENV_LOCKFILE), "sha256": sha256_text(frozen),
            "python": sys.version.split()[0]}


class ConfigTrialRegistry:
    """Append-only log of every configuration evaluated, for honest DSR deflation.

    ``count`` is the N fed to ``deflated_sharpe``: the number of distinct trials the best
    config was selected from (barrier grid x feature sets x models x ...).
    """

    def __init__(self) -> None:
        self._trials: list[dict] = []

    def register(self, **cfg) -> dict:
        rec = {"trial": len(self._trials), **cfg}
        self._trials.append(rec)
        return rec

    @property
    def count(self) -> int:
        return len(self._trials)

    def to_list(self) -> list[dict]:
        return list(self._trials)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self._trials, indent=2))


def build_run_manifest(run_id: str, *, snapshot_manifest: Optional[dict] = None,
                       trial_registry: Optional[ConfigTrialRegistry] = None,
                       feature_list: Optional[list] = None,
                       train_bounds=None, dev_bounds=None, holdout_bounds=None,
                       extra: Optional[dict] = None) -> dict:
    """Assemble the machine-readable run manifest (decisions are made from this, not stdout)."""
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": config.SEED,
        "freq": config.FREQ,
        "symbol": config.SYMBOL,
        "env": env_lockfile(),
        "snapshot": {
            "path": str(config.SNAPSHOT_PATH),
            "content_sha256": (snapshot_manifest or {}).get("content_sha256"),
            "effective_range": (snapshot_manifest or {}).get("effective_range"),
        },
        "config": {
            "k_grid": list(config.K_GRID), "h_grid": list(config.H_GRID),
            "sigma_span_grid": list(config.SIGMA_SPAN_GRID),
            "fee_rate": config.FEE_RATE, "slippage_bps": config.SLIPPAGE_BPS,
            "cost_sweep_rt_bps": list(config.COST_SWEEP_RT_BPS),
            "cpcv_n": config.CPCV_N, "cpcv_k": config.CPCV_K,
            "holdout_cut": config.HOLDOUT_CUT,
            "gate": {"cpcv_majority": config.GATE_CPCV_MAJORITY,
                     "bootstrap_ci": config.GATE_BOOTSTRAP_CI,
                     "delta_margin": config.GATE_DELTA_MARGIN},
            "deploy_rule": {"dsr_min": config.DSR_MIN, "pbo_max": config.PBO_MAX},
        },
        "dsr_trial_count": (trial_registry.count if trial_registry else None),
        "feature_list": feature_list,
        "bounds": {"train": _b(train_bounds), "dev": _b(dev_bounds), "holdout": _b(holdout_bounds)},
    }
    if extra:
        manifest.update(extra)
    return manifest


def _b(bounds):
    if bounds is None:
        return None
    lo, hi = bounds
    return [str(lo), str(hi)]


def write_run_manifest(run_id: str, manifest: dict) -> Path:
    d = run_dir(run_id)
    path = d / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def delta_to_previous(run_id: str, metrics: dict) -> dict:
    """Metric delta vs the previous run's headline metrics. Full implementation at M5."""
    # TODO(M5): locate the prior run_id under REPORTS_DIR, diff headline metrics, write delta.json
    return {"note": "delta computation implemented at M5", "current": metrics}
