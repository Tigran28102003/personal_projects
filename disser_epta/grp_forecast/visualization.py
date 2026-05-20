from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Patch


os.makedirs("figures", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")


CRISIS_PERIODS = [(2008, 2009), (2014, 2015), (2020, 2020), (2022, 2023)]


def plot_metrics_table(
    agg_results: pd.DataFrame,
    highlight_best: bool = True,
) -> Figure:
    display_df = agg_results.copy()
    rows = []
    for _, row in display_df.iterrows():
        rows.append(
            [
                row["model_name"],
                f"{row['RMSE_mean']:.4f} +/- {row['RMSE_std']:.4f}",
                f"{row['MAE_mean']:.4f} +/- {row['MAE_std']:.4f}",
                f"{row['OOS_R2_mean']:.4f} +/- {row['OOS_R2_std']:.4f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(10, 0.7 * len(rows) + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["model", "RMSE", "MAE", "OOS_R2"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for col_idx in range(4):
        table[(0, col_idx)].set_facecolor("#d9e2ec")
        table[(0, col_idx)].set_text_props(weight="bold")

    if highlight_best and not display_df.empty:
        best_positions = {
            1: int(display_df["RMSE_mean"].idxmin()),
            2: int(display_df["MAE_mean"].idxmin()),
            3: int(display_df["OOS_R2_mean"].idxmax()),
        }
        for col_idx, df_idx in best_positions.items():
            table[(display_df.index.get_loc(df_idx) + 1, col_idx)].set_facecolor("#c6f6d5")
    fig.tight_layout()
    return fig


def plot_metrics_over_time(
    cv_results: pd.DataFrame,
    metric: str = "RMSE",
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, group in cv_results.sort_values("test_year").groupby("model_name"):
        ax.plot(group["test_year"], group[metric], marker="o", linewidth=1.8, label=model_name)
    for start, end in CRISIS_PERIODS:
        ax.axvspan(start - 0.5, end + 0.5, color="gray", alpha=0.14)
    ax.set_xlabel("test year")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} over expanding-window folds")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def plot_ablation(ablation_results: pd.DataFrame) -> Figure:
    if "model_name" in ablation_results.columns:
        agg = (
            ablation_results.groupby(["scenario", "model_name"], as_index=False)["RMSE"]
            .mean()
            .rename(columns={"RMSE": "RMSE_mean"})
        )
        hue = "model_name"
    else:
        agg = (
            ablation_results.groupby("scenario", as_index=False)["RMSE"]
            .mean()
            .rename(columns={"RMSE": "RMSE_mean"})
        )
        agg["model_name"] = "model"
        hue = "model_name"
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=agg, x="scenario", y="RMSE_mean", hue=hue, ax=ax)
    baseline = agg.loc[agg["scenario"] == "A0", "RMSE_mean"].mean()
    if pd.notna(baseline):
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("scenario")
    ax.set_ylabel("RMSE mean")
    ax.set_title("Ablation study")
    fig.tight_layout()
    return fig


def plot_robustness_outliers(
    full_results: pd.DataFrame,
    outlier_results: pd.DataFrame,
) -> Figure:
    full = full_results[["model_name", "RMSE_mean"]].copy()
    full["panel"] = "full"
    out = outlier_results[["model_name", "RMSE_mean"]].copy()
    out["panel"] = "without outliers"
    data = pd.concat([full, out], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=data, x="model_name", y="RMSE_mean", hue="panel", ax=ax)
    ax.set_xlabel("model")
    ax.set_ylabel("RMSE mean")
    ax.set_title("Robustness: structural outliers")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def plot_robustness_crisis(
    crisis_results: dict[str, pd.DataFrame],
) -> Figure:
    frames = []
    for period, df in crisis_results.items():
        if df is None or df.empty:
            continue
        tmp = df[["model_name", "RMSE_mean"]].copy()
        tmp["period"] = period
        frames.append(tmp)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["model_name", "RMSE_mean", "period"])

    fig, ax = plt.subplots(figsize=(10, 5))
    if not data.empty:
        sns.barplot(data=data, x="model_name", y="RMSE_mean", hue="period", ax=ax)
    ax.set_xlabel("model")
    ax.set_ylabel("RMSE mean")
    ax.set_title("Robustness: crisis vs normal folds")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def plot_shap_global(shap_dict: dict, top_n: int = 15) -> Figure:
    importance = shap_dict["global_importance"].head(top_n).sort_values()
    feature_groups = shap_dict.get("feature_groups", {})
    feature_to_block = {
        feature: block
        for block, features in feature_groups.items()
        for feature in features
    }
    blocks = [feature_to_block.get(feature, "OTHER") for feature in importance.index]
    unique_blocks = list(dict.fromkeys(blocks))
    palette = dict(zip(unique_blocks, sns.color_palette("tab10", len(unique_blocks))))
    colors = [palette[block] for block in blocks]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance.index, importance.values, color=colors)
    ax.set_xlabel("mean absolute SHAP")
    ax.set_title("Global SHAP importance")
    ax.legend(
        handles=[Patch(color=color, label=block) for block, color in palette.items()],
        loc="lower right",
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def plot_shap_beeswarm(shap_dict: dict, top_n: int = 15) -> Figure:
    import shap

    top_features = shap_dict["global_importance"].head(top_n).index.tolist()
    positions = [shap_dict["feature_names"].index(feature) for feature in top_features]
    values = shap_dict["shap_values"][:, positions]
    X = shap_dict["X"].loc[:, top_features]
    base_values = _base_values_for_plot(shap_dict, len(X))
    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=X.to_numpy(),
        feature_names=top_features,
    )
    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(explanation, max_display=top_n, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_shap_group_heatmap(shap_dict: dict) -> Figure:
    data = shap_dict["group_importance"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(data, cmap="Blues", annot=True, fmt=".2f", ax=ax)
    ax.set_xlabel("region type")
    ax.set_ylabel("feature block")
    ax.set_title("SHAP block importance by region type")
    fig.tight_layout()
    return fig


def plot_shap_local(
    shap_dict: dict,
    region_id: str,
    region_name: str,
) -> Figure:
    import shap

    idx = shap_dict.get("local_indices", {}).get(str(region_id))
    if idx is None:
        meta = shap_dict.get("region_meta")
        if meta is not None:
            matches = meta.index[meta["region_id"].astype(str) == str(region_id)].tolist()
            idx = matches[-1] if matches else 0
        else:
            idx = 0
    base_values = _base_values_for_plot(shap_dict, len(shap_dict["X"]))
    explanation = shap.Explanation(
        values=shap_dict["shap_values"][idx],
        base_values=base_values[idx],
        data=shap_dict["X"].iloc[idx].to_numpy(),
        feature_names=shap_dict["feature_names"],
    )
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(explanation, show=False, max_display=15)
    fig = plt.gcf()
    fig.suptitle(f"{region_name} ({region_id})", y=1.02)
    fig.tight_layout()
    return fig


def plot_error_by_region(
    error_dict: dict,
    top_n: int = 20,
) -> Figure:
    data = error_dict["by_region"].head(top_n).copy()
    data = data.sort_values("MAE", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(data))))
    if "region_type" in data.columns and data["region_type"].notna().any():
        sns.barplot(data=data, x="MAE", y="region_id", hue="region_type", dodge=False, ax=ax)
        ax.legend(loc="lower right", fontsize=8)
    else:
        sns.barplot(data=data, x="MAE", y="region_id", color="#4c78a8", ax=ax)
    ax.set_xlabel("MAE")
    ax.set_ylabel("region")
    ax.set_title(f"Top-{top_n} regions by forecast error")
    fig.tight_layout()
    return fig


def plot_error_heatmap(error_dict: dict) -> Figure:
    matrix = error_dict.get("error_matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    if matrix is not None and not matrix.empty:
        sns.heatmap(matrix, cmap="Reds", ax=ax)
    ax.set_xlabel("year")
    ax.set_ylabel("region")
    ax.set_title("Absolute forecast error heatmap")
    fig.tight_layout()
    return fig


def plot_ensemble_weight_search(
    weights: "np.ndarray",
    rmse_by_w: list[float],
    best_w: float,
) -> Figure:
    """Кривая RMSE ensemble в зависимости от веса CatBoost (sweep 0→1)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(weights, rmse_by_w, "o-", color="#4c78a8", linewidth=2, markersize=5)
    best_rmse = min(rmse_by_w)
    ax.axvline(best_w, color="red", linestyle="--", linewidth=1.5,
               label=f"opt. w={best_w:.2f}  RMSE={best_rmse:.4f}")
    # горизонтальная линия: RMSE чистого CatBoost (w=1.0)
    ax.axhline(rmse_by_w[-1] if len(rmse_by_w) == len(weights) else float("nan"),
               color="orange", linestyle=":", linewidth=1.2, label="CatBoost only (w=1)")
    ax.set_xlabel("Вес CatBoost  (вес LGBM = 1 − w)")
    ax.set_ylabel("Среднее RMSE по фолдам")
    ax.set_title("Поиск оптимального веса ансамбля: CatBoost + LGBM")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_ruble_metrics_table(ruble_df: "pd.DataFrame") -> Figure:
    """
    Таблица метрик в рублях (RMSE млрд руб, MAE млрд руб, MAPE %).
    ruble_df должен содержать колонки: model, RMSE_bln_rub, MAE_bln_rub, MAPE_pct.
    """
    df = ruble_df.sort_values("RMSE_bln_rub").reset_index(drop=True)
    rows = []
    for _, row in df.iterrows():
        rows.append([
            row["model"],
            f"{row['RMSE_bln_rub']:.2f}",
            f"{row['MAE_bln_rub']:.2f}",
            f"{row['MAPE_pct']:.2f}%",
        ])

    fig, ax = plt.subplots(figsize=(10, 0.7 * len(rows) + 1.8))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Модель", "RMSE (млн руб.)", "MAE (млн. руб.)", "MAPE (%)"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for col_idx in range(4):
        table[(0, col_idx)].set_facecolor("#d9e2ec")
        table[(0, col_idx)].set_text_props(weight="bold")
    if not df.empty:
        best_col = {1: int(df["RMSE_bln_rub"].idxmin()),
                    2: int(df["MAE_bln_rub"].idxmin()),
                    3: int(df["MAPE_pct"].idxmin())}
        for col_idx, df_idx in best_col.items():
            table[(df.index.get_loc(df_idx) + 1, col_idx)].set_facecolor("#c6f6d5")
    fig.tight_layout()
    return fig


def plot_mape_by_model(ruble_df: "pd.DataFrame") -> Figure:
    """
    Горизонтальный bar chart: MAPE (%) по моделям.
    Вертикальная линия = MAPE NaiveForecaster.
    """
    df = ruble_df.sort_values("MAPE_pct", ascending=True).reset_index(drop=True)
    naive_mape = df.loc[df["model"].str.contains("Naive", case=False), "MAPE_pct"]
    naive_val = float(naive_mape.iloc[0]) if not naive_mape.empty else None

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(df))))
    colors = ["#c6f6d5" if v <= (naive_val or float("inf")) else "#fed7d7"
              for v in df["MAPE_pct"]]
    ax.barh(df["model"], df["MAPE_pct"], color=colors)
    if naive_val is not None:
        ax.axvline(naive_val, color="black", linestyle="--", linewidth=1.5,
                   label=f"Naive MAPE={naive_val:.2f}%")
    ax.set_xlabel("MAPE (%)")
    ax.set_title("Средняя абсолютная процентная ошибка прогноза ВРП")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def _base_values_for_plot(shap_dict: dict, n_rows: int) -> np.ndarray:
    base_values = np.asarray(shap_dict.get("base_values", np.zeros(n_rows)))
    if base_values.ndim == 0:
        return np.repeat(float(base_values), n_rows)
    if len(base_values) == n_rows:
        return base_values
    return np.repeat(float(np.ravel(base_values)[0]), n_rows)
