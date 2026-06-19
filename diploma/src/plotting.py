"""Plotting helpers. EVERY figure is written through :func:`save_fig`, which writes
ONLY under ``config.PICTURES`` — nothing in this project saves images anywhere else.

matplotlib only (seaborn is not a dependency). The backend is left untouched so the
notebook can still display inline; headless scripts should set ``Agg`` before import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from . import config
except ImportError:  # standalone
    import config


def save_fig(fig, name: str, subdir: Optional[str] = None, dpi: int = 120) -> Path:
    """Сохранить фигуру ТОЛЬКО в ``pictures/`` (опц. подпапка). Возвращает путь.

    Если ``subdir`` не задан, используется ``config.RUN_SUBDIR`` (тег эксперимента) —
    при grid-прогоне это раскладывает графики по ``pictures/<tag>/`` без перезатирания."""
    if subdir is None:
        subdir = getattr(config, "RUN_SUBDIR", None)
    out_dir = config.PICTURES / subdir if subdir else config.PICTURES
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ───────────────────────────────── EDA ─────────────────────────────────────
def plot_missing_bars(index: pd.Index, freq: str = "1h", name: str = "eda_missing_bars"):
    idx = pd.DatetimeIndex(index)
    full = pd.date_range(idx.min(), idx.max(), freq=freq)
    missing = full.difference(idx)
    by_month = pd.Series(1, index=missing).resample("MS").sum() if len(missing) else pd.Series(dtype=int)
    fig, ax = plt.subplots(figsize=(10, 3))
    if len(by_month):
        ax.bar(by_month.index, by_month.values, width=20)
    ax.set_title(f"Пропущенные бары по месяцам (всего {len(missing)} из {len(full)})")
    ax.set_ylabel("кол-во")
    return save_fig(fig, name), len(missing)


def plot_class_drift(y3: pd.Series, window: int = 24 * 30, name: str = "eda_class_drift"):
    y = pd.Series(np.asarray(y3), index=pd.DatetimeIndex(y3.index)) if hasattr(y3, "index") else pd.Series(y3)
    df = pd.DataFrame({c: (y == c).astype(float) for c in (-1, 0, 1)})
    roll = df.rolling(window, min_periods=window // 4).mean()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for c, lbl in [(-1, "down"), (0, "flat"), (1, "up")]:
        ax.plot(roll.index, roll[c], label=lbl)
    ax.set_title("Дрейф классов во времени (скользящая доля)")
    ax.legend(loc="upper right"); ax.set_ylabel("доля")
    return save_fig(fig, name)


def plot_acf_returns(r: pd.Series, nlags: int = 72, name: str = "eda_acf_returns"):
    from statsmodels.tsa.stattools import acf
    r = pd.Series(r).dropna().values
    acf_r = acf(r, nlags=nlags, fft=True)
    acf_abs = acf(np.abs(r), nlags=nlags, fft=True)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.stem(range(nlags + 1), acf_r, linefmt="C0-", markerfmt="C0.", basefmt=" ", label="ACF r")
    ax.plot(range(nlags + 1), acf_abs, "C1.-", label="ACF |r|")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("ACF доходностей vs |доходностей|"); ax.legend(); ax.set_xlabel("лаг (ч)")
    return save_fig(fig, name)


def plot_seasonality(r: pd.Series, name: str = "eda_seasonality_hour_dow"):
    idx = pd.DatetimeIndex(r.index)
    df = pd.DataFrame({"r": np.abs(np.asarray(r)), "hour": idx.hour, "dow": idx.dayofweek}).dropna()
    piv = df.pivot_table(index="dow", columns="hour", values="r", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_title("Сезонность |r|: час × день недели")
    ax.set_xlabel("час"); ax.set_ylabel("день недели"); fig.colorbar(im, ax=ax)
    return save_fig(fig, name)


def plot_vol_regime(r: pd.Series, window: int = 24 * 7, name: str = "eda_vol_regime"):
    vol = pd.Series(r).rolling(window, min_periods=window // 4).std() * np.sqrt(24 * 365)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(pd.DatetimeIndex(r.index), vol)
    ax.set_title("Режимы волатильности (annualised rolling σ)"); ax.set_ylabel("ann. σ")
    return save_fig(fig, name)


# ─────────────────────────────── diagnostics ───────────────────────────────
def plot_confusion(cm_norm: np.ndarray, labels=(-1, 0, 1), name: str = "confusion"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("pred"); ax.set_ylabel("true"); ax.set_title("Confusion (row-normalized)")
    fig.colorbar(im, ax=ax)
    return save_fig(fig, name)


def plot_reliability(conf: np.ndarray, acc: np.ndarray, name: str = "reliability"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(conf, acc, "o-", label="model")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title("Reliability")
    ax.legend()
    return save_fig(fig, name)


def plot_equity_drawdown(trade_returns: np.ndarray, name: str = "equity_drawdown"):
    r = np.asarray(trade_returns, dtype=float)
    r = r[np.isfinite(r)]
    eq = np.cumprod(1 + r) if r.size else np.array([1.0])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(eq); a1.set_title("Equity (trade-level)"); a1.set_ylabel("equity")
    a2.fill_between(range(len(dd)), dd, 0, color="C3", alpha=0.5)
    a2.set_ylabel("drawdown"); a2.set_xlabel("сделка")
    return save_fig(fig, name)


def plot_precision_at_coverage(curves: dict, name: str = "precision_at_coverage"):
    """curves: {label: DataFrame[coverage, precision]} — наложить кривые на один график."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for lbl, df in curves.items():
        if df is not None and len(df):
            ax.plot(df["coverage"], df["precision"], "o-", label=lbl, ms=3)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("coverage"); ax.set_ylabel("precision (hit-rate)")
    ax.set_title("Precision-at-coverage"); ax.legend()
    return save_fig(fig, name)


def plot_fold_boxplot(scores_by_setup: dict, metric: str = "MCC", name: str = "fold_boxplot"):
    """scores_by_setup: {setup_name: [per-fold scores]} -> boxplot распределений."""
    labels = list(scores_by_setup.keys())
    data = [np.asarray(scores_by_setup[k], dtype=float) for k in labels]
    data = [d[np.isfinite(d)] for d in data]
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 4))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(metric); ax.set_title(f"{metric} по фолдам (распределение, не одно число)")
    return save_fig(fig, name)


def plot_permutation_importance(names, importances, top: int = 20, name: str = "perm_importance"):
    order = np.argsort(importances)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(order) + 1))
    ax.barh(np.array(names)[order][::-1], np.array(importances)[order][::-1])
    ax.set_title("Permutation importance (top)"); ax.set_xlabel("Δ score")
    return save_fig(fig, name)


# ─────────────────────── quality plots for EVERY variant (W7) ───────────────
_CLS = (-1, 0, 1)
_CLS_LBL = {-1: "down(-1)", 0: "flat(0)", 1: "up(+1)"}


def _ovr(y_true, c):
    return (np.asarray(y_true).astype(int) == c).astype(int)


def plot_roc_ovr(y_true, y_proba, name="roc_ovr", title="ROC (OvR)"):
    """ROC-кривые one-vs-rest по классам + macro-AUC."""
    from sklearn.metrics import roc_curve, roc_auc_score
    proba = np.asarray(y_proba, float)
    fig, ax = plt.subplots(figsize=(5, 4.5)); aucs = []
    for i, c in enumerate(_CLS):
        yb = _ovr(y_true, c)
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        fpr, tpr, _ = roc_curve(yb, proba[:, i]); a = roc_auc_score(yb, proba[:, i]); aucs.append(a)
        ax.plot(fpr, tpr, label=f"{_CLS_LBL[c]} AUC={a:.2f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{title}" + (f"  macroAUC={np.mean(aucs):.2f}" if aucs else "")); ax.legend(fontsize=8)
    return save_fig(fig, name)


def plot_pr_curves(y_true, y_proba, name="pr_curves", title="Precision-Recall (OvR)"):
    """PR-кривые по классам + average-precision (важнее ROC для редких ±1)."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    proba = np.asarray(y_proba, float)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for i, c in enumerate(_CLS):
        yb = _ovr(y_true, c)
        if yb.sum() == 0:
            continue
        pr, rc, _ = precision_recall_curve(yb, proba[:, i]); ap = average_precision_score(yb, proba[:, i])
        ax.plot(rc, pr, label=f"{_CLS_LBL[c]} AP={ap:.2f}")
        ax.axhline(yb.mean(), color="gray", ls=":", lw=0.6)
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title(title); ax.legend(fontsize=8)
    return save_fig(fig, name)


def plot_score_separation(y_true, signal_score, name="score_separation"):
    """Гистограмма signal_score=P(+1)−P(-1), разбитая по ИСТИННОМУ классу — видно
    разводит ли модель классы."""
    y = np.asarray(y_true).astype(int); s = np.asarray(signal_score, float)
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in _CLS:
        m = y == c
        if m.sum():
            ax.hist(s[m], bins=40, alpha=0.5, density=True, label=_CLS_LBL[c])
    ax.set_xlabel("signal_score = P(+1) − P(−1)"); ax.set_ylabel("density")
    ax.set_title("Разделимость score по истинному классу"); ax.legend(fontsize=8)
    return save_fig(fig, name)


def plot_calibration_per_class(y_true, y_proba, name="calibration_per_class", n_bins=10):
    """Per-class reliability (OvR) + Brier в подписи."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    proba = np.asarray(y_proba, float)
    fig, ax = plt.subplots(figsize=(5, 4.5)); ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for i, c in enumerate(_CLS):
        yb = _ovr(y_true, c)
        if yb.sum() == 0:
            continue
        try:
            ft, mp = calibration_curve(yb, proba[:, i], n_bins=n_bins, strategy="quantile")
            b = brier_score_loss(yb, proba[:, i])
            ax.plot(mp, ft, "o-", ms=3, label=f"{_CLS_LBL[c]} Brier={b:.3f}")
        except Exception:  # noqa: BLE001
            pass
    ax.set_xlabel("pred prob"); ax.set_ylabel("observed freq")
    ax.set_title("Калибровка per-class"); ax.legend(fontsize=8)
    return save_fig(fig, name)


def plot_train_oof_gap(gaps: dict, name="train_oof_gap", metric="MCC"):
    """gaps: {variant: (train, oof)} -> сгруппированный bar train vs OOF (переобучение)."""
    labels = list(gaps.keys()); tr = [gaps[k][0] for k in labels]; oo = [gaps[k][1] for k in labels]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2, 4))
    ax.bar(x - w / 2, tr, w, label="train"); ax.bar(x + w / 2, oo, w, label="WF-OOF")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel(metric)
    ax.set_title(f"train vs OOF {metric} (зазор = переобучение)"); ax.legend()
    return save_fig(fig, name)


def plot_roc_overlay(variants: dict, name="roc_overlay"):
    """variants: {label: (y_true, y_proba)} -> наложенные macro-ROC."""
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(5.5, 4.5)); ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for lbl, (yt, yp) in variants.items():
        proba = np.asarray(yp, float); aucs = []
        # macro ROC via mean of per-class TPR on a common FPR grid
        grid = np.linspace(0, 1, 100); tprs = []
        for i, c in enumerate(_CLS):
            yb = _ovr(yt, c)
            if yb.sum() in (0, len(yb)):
                continue
            fpr, tpr, _ = roc_curve(yb, proba[:, i]); tprs.append(np.interp(grid, fpr, tpr))
            aucs.append(roc_auc_score(yb, proba[:, i]))
        if tprs:
            ax.plot(grid, np.mean(tprs, axis=0), label=f"{lbl} AUC={np.mean(aucs):.2f}")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC macro overlay"); ax.legend(fontsize=8)
    return save_fig(fig, name)


def plot_edge_decay(per_fold_mcc, per_fold_psi=None, name="edge_decay"):
    """Скользящий per-fold MCC по walk-forward (распад эджа) + средний PSI дрейфа (W5)."""
    m = np.asarray(per_fold_mcc, float); x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(x, m, "o-", label="per-fold MCC"); ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("walk-forward фолд (время →)"); ax.set_ylabel("MCC")
    if per_fold_psi is not None:
        ax2 = ax.twinx(); ax2.plot(x, np.asarray(per_fold_psi, float), "s--", color="C3", label="PSI drift")
        ax2.set_ylabel("PSI", color="C3"); ax2.axhline(0.2, color="C3", ls=":", lw=0.8)
    ax.set_title("Распад эджа / дрейф по walk-forward")
    return save_fig(fig, name)


def plot_shap_summary(model, X_sample, name="shap_summary", class_idx=2, max_display=15):
    """SHAP beeswarm (TreeExplainer, LightGBM) для класса class_idx (по умолч. +1).
    Тихо пропускается, если shap не установлен. X_sample — сэмпл ~2-5k строк."""
    try:
        import shap
    except Exception:  # noqa: BLE001 — shap optional
        return None
    try:
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(X_sample)
        vals = sv[class_idx] if isinstance(sv, list) else sv
        fig = plt.figure(figsize=(6, 5))
        shap.summary_plot(vals, X_sample, max_display=max_display, show=False)
        return save_fig(plt.gcf(), name)
    except Exception:  # noqa: BLE001
        return None
