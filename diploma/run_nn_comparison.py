"""NN learner-comparison for formulation A — runs in a SEPARATE process (torch isolated
from LightGBM). Builds the SAME dev walk-forward folds as notebook §10 (V5: emits a split
fingerprint the notebook asserts against) for the SAME target as the notebook (default
triple_barrier — the balanced primary), trains MLP/GRU/TCN on those folds, runs the NN
leakage auto-tests, and writes ``reports/nn_comparison.json`` + figures to ``pictures/<tag>/``.

Imports only LightGBM-free modules (config/data_io/features/labeling/validation/metrics/
plotting/nn_models) — never ``optimize``/``models`` (which pull LightGBM).

Usage:  .venv/bin/python run_nn_comparison.py [--cpu] [--quick] [--tag T] [--target triple_barrier] [--vol-k 1.5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import config, data_io, validation, metrics, plotting, nn_models  # noqa: E402


def split_fingerprint(splits) -> str:
    h = hashlib.sha256()
    for tr, te in splits:
        h.update(np.asarray(tr).tobytes()); h.update(np.asarray(te).tobytes())
    return h.hexdigest()[:16]


def wf_oof(setup, X_dev, y_dev, fwd_dev, splits):
    """WF-OOF для NN-обучателя. Для seq-моделей predict-окно расширяется на lookback-1
    предшествующих баров (прошлое, каузально), чтобы трейлинг-окно не обрывалось."""
    pad = getattr(setup, "lookback", 1) - 1
    yt, yp, pr, sg, fw = [], [], [], [], []
    per_fold = []
    for tr, te in splits:
        setup.fit(X_dev.iloc[tr], y_dev[tr], y_ret=fwd_dev[tr])
        lo = max(0, int(te[0]) - pad)
        proba_full = setup.predict_proba3(X_dev.iloc[lo:int(te[-1]) + 1])
        proba = proba_full[-len(te):]
        pred = np.asarray(nn_models.CLASSES)[proba.argmax(1)]
        yt.append(y_dev[te]); yp.append(pred); pr.append(proba)
        sg.append(proba[:, 2] - proba[:, 0]); fw.append(fwd_dev[te])
        per_fold.append(matthews_corrcoef(y_dev[te], pred) if len(np.unique(y_dev[te])) > 1 else np.nan)
    return {"y_true": np.concatenate(yt), "y_pred": np.concatenate(yp),
            "proba": np.concatenate(pr), "signal": np.concatenate(sg),
            "fwd": np.concatenate(fw), "per_fold_mcc": per_fold}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="force CPU (local macOS; V6)")
    ap.add_argument("--quick", action="store_true", help="few epochs / short")
    ap.add_argument("--tag", default="triple_barrier")
    ap.add_argument("--target", default=os.environ.get("NB_TARGET", "triple_barrier"),
                    help="label type: triple_barrier | pointwise | vol_scaled")
    ap.add_argument("--vol-k", type=float, default=float(os.environ.get("NB_VOL_K", "1.5")),
                    help="k for vol_scaled (ignored otherwise)")
    args = ap.parse_args()
    if os.environ.get("NB_QUICK") == "1":
        args.quick = True
    config.QUICK_MODE = args.quick
    config.set_determinism()

    # CUDA if available (P100), else CPU; --cpu forces CPU locally
    try:
        import torch
        use_cpu = args.cpu or not torch.cuda.is_available()
        device = "cpu" if use_cpu else "cuda"
    except Exception:  # noqa: BLE001
        use_cpu, device = True, "cpu"
    print(f"NN comparison: device={device} quick={args.quick} tag={args.tag} target={args.target}")
    config.RUN_SUBDIR = args.tag

    # dataset + SAME dev WF folds as §10 (V5) — ТОТ ЖЕ таргет, что и в ноутбуке
    X, y3, fwd, ohlc, _ = data_io.build_dataset(
        target_type=args.target, k=(args.vol_k if args.target == "vol_scaled" else None),
        threshold=config.THRESHOLD)
    n = len(X); test_start = int((config.SPLIT[0] + config.SPLIT[1]) * n)
    X_dev, y_dev, fwd_dev = X.iloc[:test_start], y3.iloc[:test_start].to_numpy(), fwd.iloc[:test_start].to_numpy()
    splits = validation.purged_walk_forward_splits(
        len(X_dev), n_splits=config.wf_splits(), purge=config.PURGE, embargo=config.EMBARGO,
        window_mode=config.WINDOW_MODE, train_window=config.TRAIN_WINDOW)
    fp = split_fingerprint(splits)
    print(f"dev={len(X_dev)} folds={len(splits)} split_fp={fp}")

    learners = nn_models.build_learners(force_cpu=use_cpu, fast=(not use_cpu), quick=args.quick)
    out = {"tag": args.tag, "device": device, "quick": args.quick, "split_fingerprint": fp,
           "n_dev": int(len(X_dev)), "learners": {}}
    for name, setup in learners.items():
        oof = wf_oof(setup, X_dev, y_dev, fwd_dev, splits)
        rep = metrics.classification_report3(oof["y_true"], oof["y_pred"])
        ic = metrics.rank_ic(oof["fwd"], oof["signal"])
        out["learners"][name] = {"mcc": rep["mcc"], "macro_f1": rep["macro"]["f1"],
                                 "ic": float(ic), "val_mcc": float(getattr(setup, "val_mcc_", np.nan))}
        print(f"  {name:4s}: MCC={rep['mcc']:+.3f} macroF1={rep['macro']['f1']:.3f} IC={ic:+.3f}")
        # cheap quality plots per NN learner -> pictures/<tag>/
        plotting.plot_roc_ovr(oof["y_true"], oof["proba"], name=f"nn_{name}_roc")
        plotting.plot_pr_curves(oof["y_true"], oof["proba"], name=f"nn_{name}_pr")
        plotting.plot_score_separation(oof["y_true"], oof["signal"], name=f"nn_{name}_score_sep")
        plotting.plot_confusion(rep["confusion_norm"], name=f"nn_{name}_confusion")

    # NN leakage auto-tests. The positive control must isolate PIPELINE correctness, so
    # it uses an UNREGULARISED control MLP (dropout=0, weight_decay=0, more epochs) — a
    # production MLP with dropout=0.4 would randomly zero the leak feature and cap it.
    def control_mlp(epochs):
        return nn_models.NNMlp(force_cpu=use_cpu, dropout=0.0, weight_decay=0.0,
                               epochs=epochs, val_frac=0.1)
    print("NN leakage tests (control MLP):")
    Xl = X_dev.copy(); Xl["LEAK_label"] = y_dev.astype(float)
    leak = wf_oof(control_mlp(80), Xl, y_dev, fwd_dev, splits)
    mcc_leak = matthews_corrcoef(leak["y_true"], leak["y_pred"])
    ys = np.random.default_rng(config.SEED).permutation(y_dev)
    shuf = wf_oof(control_mlp(30), X_dev, ys, fwd_dev, splits)
    mcc_shuf = matthews_corrcoef(shuf["y_true"], shuf["y_pred"])
    honest = out["learners"]["MLP"]["mcc"]
    Xp = X_dev.shift(-config.HORIZON); v = Xp.notna().all(axis=1)
    peek = wf_oof(control_mlp(40), Xp[v].reset_index(drop=True),
                  y_dev[v.to_numpy()], fwd_dev[v.to_numpy()],
                  validation.purged_walk_forward_splits(int(v.sum()), n_splits=config.wf_splits(),
                  purge=config.PURGE, embargo=config.EMBARGO, window_mode=config.WINDOW_MODE,
                  train_window=config.TRAIN_WINDOW))
    mcc_peek = matthews_corrcoef(peek["y_true"], peek["y_pred"])
    leak_pass = bool(mcc_leak > 0.7 and abs(mcc_shuf) < 0.15 and mcc_peek > honest + 0.03)
    out["leakage"] = {"label_leak_mcc": float(mcc_leak), "shuffle_mcc": float(mcc_shuf),
                      "peek_mcc": float(mcc_peek), "honest_mcc": float(honest), "pass": leak_pass}
    print(f"  label-leak={mcc_leak:.3f} (>0.7)  shuffle={mcc_shuf:+.3f} (~0)  "
          f"peek={mcc_peek:.3f} vs honest {honest:+.3f}  -> {'PASS' if leak_pass else 'FAIL'}")

    config.REPORTS = ROOT / "reports"; (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "nn_comparison.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("written reports/nn_comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
