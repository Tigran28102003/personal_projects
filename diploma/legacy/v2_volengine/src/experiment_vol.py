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
               features_vol as fv, tripwires_vol as tw, vol_model as vm,
               strategy_vol as sv, plots_vol as vp)
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


def run_m3_gate1(estimator: str = config.RV_ESTIMATOR) -> dict:
    """A-M3: ML ladder (HARQ → Ridge(HAR+exo) → LightGBM) on the perp-era 2019→ gate subsample +
    Gate 1 (forecasting). Both arms share identical rows/folds. Diebold-Mariano(HLN) primary +
    paired-bootstrap Δ-CI complement, distributional over CPCV. No holdout touch, no M4/M5."""
    config.ensure_dirs()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    rdir = config.REPORTS_DIR / run_id
    rdir.mkdir(parents=True, exist_ok=True)

    spot, exo = load_spot_snapshot(), load_exo_snapshot()
    spot_sha = json.loads(config.SPOT_MANIFEST.read_text()).get("content_sha256")
    exo_sha = json.loads((config.V2_MICRO / "data" / "snapshot_manifest.json").read_text()).get("content_sha256")

    X, groups = fv.build_feature_matrix(spot, estimator=estimator, exo_hourly=exo)
    X_dev = fv.dev_matrix(X)                                  # perp-era dev (holdout sealed)
    ml_cols = groups["all"]                                   # full candidate set (no selection leak)

    # headline WF-OOF ladder table (clean, non-overlapping) + distributional CPCV gate
    oof = vm.wf_oof_forecasts(X_dev, ml_cols)
    ladder = ve.summarize_arms(oof)
    ladder.to_csv(rdir / "m3_ladder_wf_oof.csv", index=False)
    paths = vm.cpcv_arm_losses(X_dev, ml_cols)
    gates = {a: ve.gate1_forecasting(paths, ml_arm=a) for a in vm.ML_ARMS}
    any_pass = any(g["passed"] for g in gates.values())

    bounds = {"perp_era_start": config.PERP_ERA_START, "dev_start": str(X_dev.index.min()),
              "dev_end": str(X_dev.index.max()), "holdout_cut": config.HOLDOUT_CUT,
              "n_dev_rows": int(len(X_dev)), "n_oof_rows": int(len(oof)), "n_cpcv_paths": len(paths)}
    manifest = {
        "run_id": run_id, "milestone": "A-M3",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": config.SEED, "estimator": estimator,
        "spot_snapshot_sha256": spot_sha, "exo_snapshot_sha256": exo_sha,
        "env_lock_sha256": _env_lock_sha(),
        "gate_subsample": bounds, "ml_candidate_features": ml_cols, "harq_baseline": vm.HARQ_COLS,
        "config": {"cpcv_N": config.CPCV_N, "cpcv_k": config.CPCV_K, "ridge_alpha": config.RIDGE_ALPHA,
                   "dm_alpha": config.DM_ALPHA, "gate1_cpcv_majority": config.GATE1_CPCV_MAJORITY,
                   "gate_delta_margin": config.GATE_DELTA_MARGIN, "ci": config.GATE_BOOTSTRAP_CI,
                   "n_bootstrap": config.n_bootstrap(), "quick_mode": config.QUICK_MODE,
                   "lgbm_n_estimators": config.lgbm_n_estimators()},
        "ladder_wf_oof": ladder.to_dict(orient="records"),
        "gate1": gates, "gate1_any_pass": any_pass,
    }
    (rdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"run_id={run_id}")
    print(f"gate subsample (perp-era dev): {bounds['dev_start']} → {bounds['dev_end']} "
          f"| dev_rows={bounds['n_dev_rows']} oof_rows={bounds['n_oof_rows']} cpcv_paths={bounds['n_cpcv_paths']}")
    print(f"HARQ baseline: {vm.HARQ_COLS}")
    print(f"ML candidate features ({len(ml_cols)}): {ml_cols}")
    print(f"QUICK_MODE={config.QUICK_MODE} (lgbm n_est={config.lgbm_n_estimators()}, n_boot={config.n_bootstrap()})")
    print("\n=== M3 ladder (WF-OOF) — QLIKE / R²-logRV / DM(HLN) vs HARQ ===")
    print(ladder.round(4).to_string(index=False))
    print("\n=== Gate 1 (forecasting) — distributional over CPCV ===")
    for a, g in gates.items():
        print(f"  [{a} vs HARQ] majority_DM_sig={g['majority_DM_sig']:.2f} "
              f"majority_QLIKE_win={g['majority_QLIKE_win']:.2f} "
              f"ΔQLIKE_mean={g['mean_delta_qlike']:.5f} Δ-CI=[{g['delta_ci_low']:.5f},{g['delta_ci_high']:.5f}] "
              f"→ {'PASS' if g['passed'] else 'FAIL'}  ({g['verdict']})")
    print(f"\n=== A-M3 Gate 1: {'ML+exo ADDS value' if any_pass else 'ML NULL — HAR stands'} ===")
    print(f"artifacts: {rdir}")
    return {"run_id": run_id, "ladder": ladder, "gate1": gates, "any_pass": any_pass, "bounds": bounds}


def run_m45_voltiming(estimator: str = config.RV_ESTIMATOR) -> dict:
    """A-M4/M5: Layer-2 vol-timing overlay on the HARQ forecast + Gate 2 (economic kill-switch).

    Inverse-vol sizing (long-flat [0,1], cap=1, daily rebalance) on the dev period (perp-era
    2019→2025-05); net-cost Sharpe vs buy&hold distributionally over CPCV + PSR/DSR/PBO + breakeven
    sweep. Holdout stays SEALED (unseal = M6/M7). Pre-registered target_vol/costs, no tune-to-pass."""
    config.ensure_dirs()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    rdir = config.REPORTS_DIR / run_id
    rdir.mkdir(parents=True, exist_ok=True)

    spot, exo = load_spot_snapshot(), load_exo_snapshot()
    spot_sha = json.loads(config.SPOT_MANIFEST.read_text()).get("content_sha256")
    exo_sha = json.loads((config.V2_MICRO / "data" / "snapshot_manifest.json").read_text()).get("content_sha256")

    frame = rvmod.build_rv(spot, estimator=estimator)
    dev, _ = rvmod.seal_holdout(frame.index)                         # holdout sealed
    frame_dev = frame[dev.to_numpy()].loc[pd.Timestamp(config.PERP_ERA_START):]   # perp-era dev only
    perp_close_daily = exo["perp_close"].groupby(exo.index.normalize()).last()

    out = sv.run_overlay(frame_dev, perp_close_daily, exo, target_vol_ann=config.VOL_TARGET_ANN)
    assert out.index.max() < pd.Timestamp(config.HOLDOUT_CUT), "holdout leaked into the overlay"
    gate = ve.gate2_economics(out)
    fig = vp.plot_vol_timing(out)

    out.to_csv(rdir / "m5_vol_timing_daily.csv")
    bounds = {"perp_era_start": config.PERP_ERA_START, "dev_start": str(out.index.min()),
              "dev_end": str(out.index.max()), "holdout_cut": config.HOLDOUT_CUT, "n_days": int(len(out))}
    manifest = {
        "run_id": run_id, "milestone": "A-M4/M5",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": config.SEED, "estimator": estimator,
        "spot_snapshot_sha256": spot_sha, "exo_snapshot_sha256": exo_sha,
        "env_lock_sha256": _env_lock_sha(), "dev_range": bounds,
        "config": {"target_vol_ann": config.VOL_TARGET_ANN, "cap": config.VOL_TIMING_CAP,
                   "fee_rate": config.FEE_RATE, "slippage_bps": config.SLIPPAGE_BPS,
                   "cpcv_N": config.CPCV_N, "cpcv_k": config.CPCV_K,
                   "gate2_cpcv_majority": config.GATE2_CPCV_MAJORITY,
                   "gate_delta_margin": config.GATE_DELTA_MARGIN, "ci": config.GATE_BOOTSTRAP_CI,
                   "n_bootstrap": config.n_bootstrap(), "quick_mode": config.QUICK_MODE,
                   "dsr_min": config.DSR_MIN, "pbo_max": config.PBO_MAX},
        "gate2": gate, "figure": str(fig.relative_to(config.PKG_ROOT)),
    }
    (rdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"run_id={run_id}")
    print(f"dev (perp-era, holdout sealed): {bounds['dev_start']} → {bounds['dev_end']} | days={bounds['n_days']}")
    print(f"target_vol_ann={config.VOL_TARGET_ANN} cap={config.VOL_TIMING_CAP} "
          f"avg_exposure={gate['avg_exposure']:.3f} | baseline_rt={gate['baseline_rt_bps']:.1f}bps")
    print("\n=== M4 vol-timing vs buy&hold (full-sample, annualised) ===")
    print(f"  Sharpe   vol-timing={gate['sharpe_vol_timing']:.3f}  buy&hold={gate['sharpe_buy_hold']:.3f}  "
          f"const-vol={gate['sharpe_const_vol']:.3f}")
    print(f"  MaxDD    vol-timing={gate['mdd_vol_timing']:.3f}  buy&hold={gate['mdd_buy_hold']:.3f}")
    print("\n=== M5 Gate 2 (economics) — distributional over CPCV ===")
    print(f"  majority(vt>bh)={gate['cpcv_majority_vt_gt_bh']:.2f} (thr {gate['majority_threshold']}) "
          f"| ΔSharpe_mean={gate['mean_delta_sharpe']:.3f} Δ-CI=[{gate['delta_ci_low']:.3f},{gate['delta_ci_high']:.3f}]")
    print(f"  PSR(vt>bh)={gate['psr_vs_buy_hold']:.3f}  DSR={gate['deflated_sharpe']:.3f} (≥{gate['dsr_min']}?)  "
          f"PBO={gate['pbo']:.3f} (≤{gate['pbo_max']}?)  breakeven={gate['breakeven_rt_bps']:.0f}bps")
    print(f"\n=== A-M5 Gate 2: {'PASS' if gate['passed'] else 'FAIL'} — {gate['verdict']} ===")
    print(f"figure: {fig}\nartifacts: {rdir}")
    return {"run_id": run_id, "out": out, "gate2": gate}


def _main(argv=None) -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Vol-Engine experiment (M1 / M2 / M3 / M45).")
    p.add_argument("cmd", choices=["m1", "m2", "m3", "m45"], nargs="?", default="m1")
    args = p.parse_args(argv)
    {"m1": run_har_baselines, "m2": run_m2_features, "m3": run_m3_gate1,
     "m45": run_m45_voltiming}[args.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
