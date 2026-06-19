"""Figures for the Vol-Engine — written ONLY under config.FIGURES_DIR (A-M1 demonstration)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
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


def plot_bars_per_day(spot: pd.DataFrame, name: str = "m0_bars_per_day.png") -> Path:
    per_day = spot.groupby(spot.index.normalize()).size()
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(per_day.index, per_day.values, lw=0.5, color="#2c7fb8")
    ax.axhline(config.BARS_PER_DAY, color="k", lw=0.6, ls="--", label="24 bars")
    ax.axhline(config.MIN_BARS_KEEP, color="#de2d26", lw=0.6, ls=":", label=f"drop < {config.MIN_BARS_KEEP}")
    ax.set_ylabel("bars / day"); ax.set_title("R4: bars-per-day audit (spot hourly)")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, name)


def plot_rv_timeseries(frame: pd.DataFrame, name: str = "m0_rv_timeseries.png") -> Path:
    ann = np.sqrt(frame["rv"].clip(lower=config.RV_FLOOR)) * np.sqrt(config.DAYS_PER_YEAR)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(ann.index, ann.values, lw=0.6, color="#333333")
    ax.set_ylabel("annualised vol"); ax.set_title("Realized volatility (annualised, %s RV)" % config.RV_ESTIMATOR)
    return _save(fig, name)


def plot_logrv_acf(frames: dict, lags: int = 30, name: str = "m0_logrv_acf.png") -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colours = {"range": "#2c7fb8", "close": "#999999"}
    for est, fr in frames.items():
        lr = np.log(fr["rv"].clip(lower=config.RV_FLOOR)).dropna()
        acf = [lr.autocorr(k) for k in range(1, lags + 1)]
        ax.plot(range(1, lags + 1), acf, marker="o", ms=3, lw=1,
                color=colours.get(est, None), label=f"{est} (ACF1={acf[0]:.2f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lag (days)"); ax.set_ylabel("ACF of log-RV")
    ax.set_title("log-RV persistence (long memory = predictability)")
    ax.legend(fontsize=8)
    return _save(fig, name)


def plot_har_vs_actual(oof: pd.DataFrame, model: str = "HARQ", last: int = 250,
                       name: str = "m1_har_vs_actual.png") -> Path:
    sub = oof.tail(last)
    ann_true = np.sqrt(sub["y_rv"].clip(lower=config.RV_FLOOR)) * np.sqrt(config.DAYS_PER_YEAR)
    ann_pred = np.sqrt(sub[f"{model}_rv"].clip(lower=config.RV_FLOOR)) * np.sqrt(config.DAYS_PER_YEAR)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(ann_true.index, ann_true.values, lw=0.8, color="#333333", label="actual RV")
    ax.plot(ann_pred.index, ann_pred.values, lw=0.9, color="#2c7fb8", label=f"{model} forecast")
    ax.set_ylabel("annualised vol"); ax.set_title(f"{model} forecast vs actual RV (OOF, last {last} days)")
    ax.legend(fontsize=8)
    return _save(fig, name)
