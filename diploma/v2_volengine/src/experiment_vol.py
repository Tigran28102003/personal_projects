"""Vol-Engine experiment orchestration (A-M1: HAR-family-vs-naive on the dev series).

Emits reports/<run_id>/{manifest.json, har_vs_naive_<estimator>.csv}. Reads the frozen spot-OHLC
snapshot; computes RV (primary estimator + robustness arm), walk-forward OOF for the HAR family and
naive baselines, and the QLIKE / R²-logRV table. M2+ (ML/exo, gates) are not here yet.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

from . import config, rv as rvmod, har as harmod, vol_evaluate as ve
from .data_spot import load_spot_snapshot

logger = logging.getLogger(__name__)


def _env_lock_sha() -> str:
    try:
        frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except (subprocess.SubprocessError, OSError) as e:
        frozen = f"# pip freeze failed: {e}\n"
    config.ensure_dirs()
    config.ENV_LOCKFILE.write_text(frozen)
    return hashlib.sha256(frozen.encode()).hexdigest()


def run_har_baselines(estimators=("range", "close"), embargo: int = 5) -> dict:
    """M1: HAR family vs naive RV baselines on the dev series, for each RV estimator."""
    config.ensure_dirs()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    rdir = config.REPORTS_DIR / run_id
    rdir.mkdir(parents=True, exist_ok=True)

    spot = load_spot_snapshot()
    spot_sha = json.loads(config.SPOT_MANIFEST.read_text()).get("content_sha256")

    tables = {}
    for est in estimators:
        frame = rvmod.build_rv(spot, estimator=est)
        dev, _ = rvmod.seal_holdout(frame.index)            # holdout sealed: M1 on dev only
        oof = harmod.walk_forward_oof(frame[dev.to_numpy()], embargo=embargo)
        tbl = ve.summarize_models(oof)
        tbl.to_csv(rdir / f"har_vs_naive_{est}.csv", index=False)
        tables[est] = tbl
        logger.info("estimator=%s OOF days=%d", est, len(oof))

    manifest = {
        "run_id": run_id, "milestone": "A-M1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": config.SEED,
        "config": {
            "rv_estimator_primary": config.RV_ESTIMATOR, "rv_range_method": config.RV_RANGE_METHOD,
            "estimators_evaluated": list(estimators), "target_transform": config.TARGET_TRANSFORM,
            "har_windows": list(config.HAR_WINDOWS), "har_variants": list(config.HAR_VARIANTS),
            "wf_n_splits": config.WF_N_SPLITS, "embargo": embargo,
            "min_bars_keep": config.MIN_BARS_KEEP, "holdout_cut": config.HOLDOUT_CUT,
        },
        "spot_snapshot_sha256": spot_sha,
        "env_lock_sha256": _env_lock_sha(),
        "dev_range": {"start": str(rvmod.build_rv(spot, estimators[0]).index.min()),
                      "holdout_cut": config.HOLDOUT_CUT},
        "tables": {est: t.to_dict(orient="records") for est, t in tables.items()},
    }
    (rdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"run_id={run_id}")
    for est, t in tables.items():
        print(f"\n=== HAR family vs naive — RV estimator: {est} ===")
        print(t.to_string(index=False))
    print(f"\nartifacts: {rdir}")
    return {"run_id": run_id, "tables": tables}


def _main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_har_baselines()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
