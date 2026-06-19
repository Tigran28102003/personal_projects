"""All figures for the redesign — written ONLY under ``config.FIGURES_DIR``.

Use a non-interactive backend; every function returns the saved path. Grown per milestone
(M1: label balance; M5: equity, calibration, importance, CPCV metric distribution,
breakeven, ablation, PnL split).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config


def _save(fig, name: str) -> Path:
    config.ensure_dirs()
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_label_balance(reports: Sequence[dict], name: str = "m1_label_balance.png",
                       title: Optional[str] = None) -> Path:
    """Stacked +1/0/-1 fractions per (k, H[, span]) config, with the barrier-to-cost ratio."""
    labels = [f"k={r.get('k')}\nH={r.get('H')}" for r in reports]
    up = [r["fractions"]["+1"] for r in reports]
    flat = [r["fractions"]["0"] for r in reports]
    dn = [r["fractions"]["-1"] for r in reports]
    x = range(len(reports))

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(reports)), 4.2))
    ax.bar(x, up, label="+1 (TP)", color="#2c7fb8")
    ax.bar(x, flat, bottom=up, label="0 (vertical)", color="#bdbdbd")
    ax.bar(x, dn, bottom=[u + f for u, f in zip(up, flat)], label="-1 (SL)", color="#de2d26")
    for i, r in enumerate(reports):
        ax.text(i, 1.02, f"{r['barrier_to_cost_ratio']:.1f}×", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("class fraction")
    ax.set_title(title or "Triple-barrier label balance (dev) — barrier/cost ratio above bars")
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    return _save(fig, name)


def plot_cpcv_ablation(flow_sharpes, control_sharpes, gate: dict,
                       name: str = "m3_ablation_gate.png") -> Path:
    """Per-CPCV-path net-cost Sharpe: flow vs control, with the gate verdict annotated."""
    flow = list(flow_sharpes)
    ctrl = list(control_sharpes)
    x = range(len(flow))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [2.4, 1]})
    width = 0.4
    ax1.bar([i - width / 2 for i in x], flow, width, label="flow", color="#2c7fb8")
    ax1.bar([i + width / 2 for i in x], ctrl, width, label="control", color="#bdbdbd")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("CPCV path")
    ax1.set_ylabel("Sharpe (de-overlapped trades)")
    ax1.set_title("Flow vs control per CPCV path")
    ax1.legend(fontsize=8)

    diff = [f - c for f, c in zip(flow, ctrl)]
    ax2.boxplot([diff], tick_labels=["Δ = flow − control"])
    ax2.axhline(0, color="r", lw=0.8, ls="--")
    ax2.scatter([1] * len(diff), diff, color="#2c7fb8", alpha=0.6, zorder=3)
    verdict = "PASS" if gate.get("passed") else "FAIL"
    ax2.set_title(f"Δ-Sharpe — gate: {verdict}\n"
                  f"maj={gate.get('majority_flow_gt_control'):.2f} "
                  f"CI=[{gate.get('delta_ci_low'):.2f},{gate.get('delta_ci_high'):.2f}]",
                  fontsize=9)
    return _save(fig, name)


def plot_barriers_on_price(close, labels_frame, n_events: int = 6,
                           name: str = "m1_barriers_on_price.png") -> Path:
    """Illustrate the triple barrier on a price window: a few entries with their TP/SL levels
    (``entry·exp(±k·σ)``), the vertical barrier at ``t1``, and the realised outcome colour."""
    f = labels_frame
    step = max(1, len(f) // n_events)
    sample = f.iloc[::step].head(n_events)
    lo, hi = sample.index.min(), sample["t1"].max()
    px = close.loc[lo:hi]
    colours = {1: "#2c7fb8", -1: "#de2d26", 0: "#7f7f7f"}

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(px.index, px.to_numpy(), color="#333333", lw=0.9, label="perp close")
    for ts, row in sample.iterrows():
        p0 = float(close.loc[ts]); t1 = row["t1"]
        up = p0 * np.exp(row["up_thr"]); dn = p0 * np.exp(row["dn_thr"])
        ax.hlines(up, ts, t1, color="#2c7fb8", lw=0.8, ls="--")
        ax.hlines(dn, ts, t1, color="#de2d26", lw=0.8, ls="--")
        ax.vlines(t1, dn, up, color="#7f7f7f", lw=0.8, ls=":")
        ax.scatter([ts], [p0], color=colours[int(row["label"])], s=28, zorder=3)
    ax.set_ylabel("price"); ax.set_title("Triple barrier on a price window (TP/SL = entry·exp(±k·σ))")
    ax.legend(fontsize=8, loc="best")
    return _save(fig, name)


def plot_confusion(y_true, y_pred, labels=(-1, 0, 1),
                   name: str = "m3_confusion.png", title: Optional[str] = None) -> Path:
    """3×3 confusion matrix (Stage-1 OOF), counts annotated."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title or "Stage-1 confusion (OOF)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    return _save(fig, name)
