"""Experiment orchestration (M3: ladder + distributional ablation gate; expands to the full
Stage-1→Stage-2→strategy→economics run at M4/M5).

Replaces the old r_t walk-forward core. Wires snapshot → labeling → features → Stage-1 ladder
→ CPCV path economics → pre-registered flow-vs-control gate, and emits the run-contract
artifacts under ``reports/<run_id>/``. Decisions are made from those artifacts, not stdout.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import config, labeling as lb, features as ft, evaluate as ev, plots
from . import run_manifest as rm

logger = logging.getLogger(__name__)


def run_ablation_gate(k: float = config.K_DEFAULT, H: int = config.H_DEFAULT,
                      sigma_span: int = config.SIGMA_SPAN_DEFAULT,
                      model_name: str = "lgbm") -> dict:
    """M3: run the Stage-1 ladder on the flow arm and the flow-vs-control distributional gate.

    Emits reports/<run_id>/{run_manifest.json, ladder.csv, ablation.csv, gate.json} plus the
    label-balance and ablation figures. Returns the gate dict.
    """
    config.ensure_dirs()
    run_id = rm.new_run_id()
    rdir = rm.run_dir(run_id)
    snap_manifest = json.loads(config.SNAPSHOT_MANIFEST.read_text())

    snap = ev_load_snapshot()
    dev, hold = lb.seal_holdout(snap.index)
    X, meta = ft.build_features(snap, dev_mask=dev)
    close_dev = snap.loc[dev.to_numpy(), "perp_close"]
    L = lb.triple_barrier_labels(close_dev, k=k, H=H, sigma_span=sigma_span)
    _, y, w = ft.align_xy(X, L)
    trade_info = ev.build_trade_info(L, close_dev)   # de-overlapped trade-level returns for the gate
    splits = ev.cpcv_split_list(len(y), H=H)

    # config-trial registry → DSR N (every arm×model under the fixed barrier config)
    reg = rm.ConfigTrialRegistry()

    # ladder on the flow arm + control workhorse
    ladder_rows, path_sharpes = [], {}
    for arm, cols in (("flow", meta["arm_flow"]), ("control", meta["arm_control"])):
        models = ("baseline", "logistic", "lgbm") if arm == "flow" else ("lgbm",)
        for name in models:
            reg.register(k=k, H=H, sigma_span=sigma_span, arm=arm, model=name)
            s = ev.arm_path_sharpes(X, y, w, trade_info, cols, splits, model_name=name)
            path_sharpes[f"{arm}/{name}"] = s
            ladder_rows.append({"arm": arm, "model": name, "n_paths": int(np.isfinite(s).sum()),
                                "mean_sharpe": float(np.nanmean(s)),
                                "median_sharpe": float(np.nanmedian(s)),
                                "min_sharpe": float(np.nanmin(s)), "max_sharpe": float(np.nanmax(s))})

    gate = ev.distributional_gate(path_sharpes[f"flow/{model_name}"],
                                  path_sharpes[f"control/{model_name}"])
    gate["gate_model"] = model_name
    gate["dsr_trial_count"] = reg.count

    # ---- artifacts ----
    pd.DataFrame(ladder_rows).to_csv(rdir / "ladder.csv", index=False)
    pd.DataFrame({k_: pd.Series(v) for k_, v in path_sharpes.items()}).to_csv(
        rdir / "ablation.csv", index_label="cpcv_path")
    (rdir / "gate.json").write_text(json.dumps(gate, indent=2))
    reg.save(rdir / "config_trials.json")

    manifest = rm.build_run_manifest(
        run_id, snapshot_manifest=snap_manifest, trial_registry=reg,
        feature_list={"flow": meta["arm_flow"], "control": meta["arm_control"]},
        dev_bounds=(close_dev.index.min(), close_dev.index.max()),
        holdout_bounds=(snap.index[hold.to_numpy()][0], snap.index[hold.to_numpy()][-1]),
        extra={"milestone": "M3", "barrier": {"k": k, "H": H, "sigma_span": sigma_span},
               "d_star": meta["d_star"], "gate": gate, "ladder": ladder_rows},
    )
    rm.write_run_manifest(run_id, manifest)

    # ---- figures ----
    rep = lb.label_report(L); rep["k"], rep["H"] = k, H
    plots.plot_label_balance([rep], name=f"{run_id}_label_balance.png")
    plots.plot_cpcv_ablation(path_sharpes[f"flow/{model_name}"],
                             path_sharpes[f"control/{model_name}"], gate,
                             name=f"{run_id}_ablation_gate.png")

    logger.info("M3 run %s — gate: %s", run_id, gate["verdict"])
    print(f"run_id={run_id}\n{json.dumps(gate, indent=2)}")
    print(f"artifacts: {rdir}")
    return gate


def ev_load_snapshot():
    """Local import shim so the module imports without touching the network/snapshot."""
    from .data_snapshot import load_snapshot
    return load_snapshot()


def _main(argv=None) -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Direction-C experiment (M3 ablation gate).")
    p.add_argument("--k", type=float, default=config.K_DEFAULT)
    p.add_argument("--H", type=int, default=config.H_DEFAULT)
    p.add_argument("--sigma-span", type=int, default=config.SIGMA_SPAN_DEFAULT)
    p.add_argument("--model", default="lgbm")
    args = p.parse_args(argv)
    run_ablation_gate(k=args.k, H=args.H, sigma_span=args.sigma_span, model_name=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
