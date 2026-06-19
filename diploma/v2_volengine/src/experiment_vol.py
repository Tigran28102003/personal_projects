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

from . import (config, rv as rvmod, har as harmod, vol_evaluate as ve,
               features_vol as fv, tripwires_vol as tw)
from .data_spot import load_spot_snapshot
from v2_microstructure.src.data_snapshot import load_snapshot as load_exo_snapshot

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

    tables, mh_tables = {}, {}
    for est in estimators:
        frame = rvmod.build_rv(spot, estimator=est)
        dev, _ = rvmod.seal_holdout(frame.index)            # holdout sealed: M1 on dev only
        fdev = frame[dev.to_numpy()]
        oof = harmod.walk_forward_oof(fdev, embargo=embargo)   # h=1 detail (per-model)
        tbl = ve.summarize_models(oof)
        tbl.to_csv(rdir / f"har_vs_naive_{est}.csv", index=False)
        mh = ve.multi_horizon_summary(fdev, horizons=config.HAR_HORIZONS, embargo=embargo)
        mh.to_csv(rdir / f"har_multihorizon_{est}.csv", index=False)
        tables[est], mh_tables[est] = tbl, mh
        logger.info("estimator=%s OOF days=%d", est, len(oof))

    manifest = {
        "run_id": run_id, "milestone": "A-M1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": config.SEED,
        "config": {
            "rv_estimator_primary": config.RV_ESTIMATOR, "rv_range_method": config.RV_RANGE_METHOD,
            "estimators_evaluated": list(estimators), "target_transform": config.TARGET_TRANSFORM,
            "har_windows": list(config.HAR_WINDOWS), "har_variants": list(config.HAR_VARIANTS),
            "har_horizons": list(config.HAR_HORIZONS),
            "wf_n_splits": config.WF_N_SPLITS, "embargo": embargo,
            "min_bars_keep": config.MIN_BARS_KEEP, "holdout_cut": config.HOLDOUT_CUT,
        },
        "spot_snapshot_sha256": spot_sha,
        "env_lock_sha256": _env_lock_sha(),
        "dev_range": {"start": str(rvmod.build_rv(spot, estimators[0]).index.min()),
                      "holdout_cut": config.HOLDOUT_CUT},
        "tables": {est: t.to_dict(orient="records") for est, t in tables.items()},
        "multi_horizon": {est: t.to_dict(orient="records") for est, t in mh_tables.items()},
    }
    (rdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"run_id={run_id}")
    for est, t in tables.items():
        print(f"\n=== HAR family vs naive (h=1) — RV estimator: {est} ===")
        print(t.to_string(index=False))
    for est, t in mh_tables.items():
        print(f"\n=== Multi-horizon HAR vs naive (mean RV over [t+1,t+h]) — RV estimator: {est} ===")
        print(t.to_string(index=False))
    print(f"\nartifacts: {rdir}")
    return {"run_id": run_id, "tables": tables, "multi_horizon": mh_tables}


def _lgbm_reg():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=config.lgbm_n_estimators(), num_leaves=31,
                             learning_rate=0.05, min_child_samples=50, subsample=0.8,
                             colsample_bytree=0.8, random_state=config.SEED, n_jobs=-1, verbose=-1)


def run_m2_features(estimator: str = config.RV_ESTIMATOR, top_k: int = 10) -> dict:
    """A-M2: build the ML+exo matrix (perp-era 2019→), train-only select, prove no leakage.

    Hard gate: all leakage tripwires must be GREEN. No Gate-1 comparison, no holdout touch, no M3.
    Records both snapshot hashes (spot RV-snapshot + C exo-snapshot) in the manifest.
    """
    config.ensure_dirs()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    rdir = config.REPORTS_DIR / run_id
    rdir.mkdir(parents=True, exist_ok=True)

    spot, exo = load_spot_snapshot(), load_exo_snapshot()
    spot_sha = json.loads(config.SPOT_MANIFEST.read_text()).get("content_sha256")
    exo_sha = json.loads((config.V2_MICRO / "data" / "snapshot_manifest.json").read_text()).get("content_sha256")

    X, groups = fv.build_feature_matrix(spot, estimator=estimator, exo_hourly=exo)
    X_dev = fv.dev_matrix(X)
    selected, clusters = fv.stable_select(X_dev, groups["all"], k=top_k)

    # --- leakage tripwires (hard gate) ---
    y = X_dev["target"].to_numpy()

    def fp(Xtr, ytr, Xev):
        m = _lgbm_reg(); m.fit(Xtr, ytr); return m.predict(Xev)

    m_full = _lgbm_reg(); m_full.fit(X_dev[selected], y)
    imp = dict(zip(selected, m_full.feature_importances_.astype(float)))
    checks = [
        tw.label_shuffle_reg(X_dev[selected], y, fp),                     # shuffled R² -> ~0
        tw.t_minus_1_availability_audit(X_dev[selected], X_dev["target"]),  # no feature ~1 with future RV
        tw.bars_per_day_guard(spot),
        tw.feature_dominance_flag(imp),
        tw.constant_prediction_guard(m_full.predict(X_dev[selected])),
    ]
    summary = tw.summarize(checks)

    gate_bounds = {"perp_era_start": config.PERP_ERA_START, "dev_start": str(X_dev.index.min()),
                   "dev_end": str(X_dev.index.max()), "holdout_cut": config.HOLDOUT_CUT,
                   "n_dev_rows": int(len(X_dev))}
    manifest = {
        "run_id": run_id, "milestone": "A-M2",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": config.SEED, "estimator": estimator,
        "spot_snapshot_sha256": spot_sha, "exo_snapshot_sha256": exo_sha,
        "env_lock_sha256": _env_lock_sha(),
        "gate_subsample": gate_bounds,
        "feature_groups": groups, "selected_features": selected, "collinear_clusters": clusters,
        "tripwires": checks, "tripwires_all_passed": summary["all_passed"],
    }
    (rdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (rdir / "feature_selection.json").write_text(json.dumps(
        {"candidates": groups["all"], "selected": selected, "clusters": clusters}, indent=2))

    print(f"run_id={run_id}")
    print(f"gate subsample (perp-era dev): {gate_bounds['dev_start']} → {gate_bounds['dev_end']} "
          f"| rows={gate_bounds['n_dev_rows']}")
    print(f"candidates ({len(groups['all'])}): {groups['all']}")
    print(f"collinear clusters: {clusters or 'none'}")
    print(f"selected ({len(selected)}): {selected}")
    print("tripwires:")
    for c in checks:
        print("  ", c)
    print("TRIPWIRE SUMMARY:", summary)
    assert summary["all_passed"], f"M2 leakage tripwires RED: {summary['red']}"
    print(f"\n=== A-M2: tripwires GREEN ===\nartifacts: {rdir}")
    return {"run_id": run_id, "selected": selected, "summary": summary, "bounds": gate_bounds}


def _main(argv=None) -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Vol-Engine experiment (M1 baselines / M2 features).")
    p.add_argument("cmd", choices=["m1", "m2"], nargs="?", default="m1")
    args = p.parse_args(argv)
    run_har_baselines() if args.cmd == "m1" else run_m2_features()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
