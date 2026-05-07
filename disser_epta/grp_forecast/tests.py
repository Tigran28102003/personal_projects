from __future__ import annotations

import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from models import (
    BaseForecaster,
    CatBoostForecaster,
    GPBoostForecaster,
    LGBMForecaster,
    NaiveForecaster,
)
from validation import aggregate_results, compute_metrics


OUTLIER_REGIONS = [
    "москва",
    "ханты-мансийский ао",
    "ямало-ненецкий ао",
    "сахалинская область",
]

CRISIS_YEARS = {
    "crisis_2008": [2008, 2009],
    "crisis_2014": [2014, 2015],
    "crisis_2020": [2020],
    "crisis_2022": [2022, 2023],
}


def run_ablation_study(
    best_model_cls: type[BaseForecaster],
    folds: list[dict],
    feature_groups: dict[str, list[str]],
    model_config: dict | None = None,
) -> pd.DataFrame:
    """
    Ablation study: incrementally add feature blocks and measure RMSE.

    Parameters
    ----------
    best_model_cls : type[BaseForecaster]
        Class of the best model (e.g. CatBoostForecaster).
    folds : list[dict]
        Cross-validation folds from build_folds().
    feature_groups : dict[str, list[str]]
        Mapping block_name -> list of feature column names.
    model_config : dict | None
        Hyperparameters forwarded to best_model_cls(**model_config).
        If None, defaults to {} (model uses its own defaults).
        Pass CONFIG["models"]["catboost"] etc. for reproducible results.
    """
    scenarios = {
        "A0": ["LAG_BLOCK"],
        "A1": ["LAG_BLOCK", "PRODUCTION_BLOCK", "INVESTMENT_BLOCK"],
        "A2": ["LAG_BLOCK", "LABOUR_BLOCK", "BUDGET_BLOCK"],
        "A3": [
            "LAG_BLOCK",
            "PRODUCTION_BLOCK",
            "INVESTMENT_BLOCK",
            "LABOUR_BLOCK",
            "BUDGET_BLOCK",
            "MACRO_BLOCK",
            "PRICE_BLOCK",
            "COLLECTION_BLOCK",
            "CATEGORICAL_BLOCK",
            "SHOCK_BLOCK",
        ],
    }
    cfg = dict(model_config or {})
    rows: list[dict] = []
    for scenario, blocks in scenarios.items():
        selected = _columns_for_blocks(feature_groups, blocks)
        for fold in folds:
            X_train = _filter_features(fold["X_train"], selected)
            X_test = _filter_features(fold["X_test"], selected)
            model = best_model_cls(**cfg)
            model.fit(X_train, fold["y_train"])
            y_pred = model.predict(X_test)
            # Use the same benchmark as run_cross_validation:
            # fold["y_bench"] is mean growth rate (predict_growth=True)
            # or log_grp_lag1 level (predict_growth=False).
            y_bench = np.asarray(
                fold.get("y_bench", X_test["log_grp_lag1"].astype(float).to_numpy()),
                dtype=float,
            )
            metrics = compute_metrics(fold["y_test"].to_numpy(), y_pred, y_bench)
            rows.append(
                {
                    "scenario": scenario,
                    "model_name": model.name,
                    "fold": fold["fold"],
                    "test_year": fold["test_year"],
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def robustness_check_outliers(
    models: list[BaseForecaster],
    folds: list[dict],
    outlier_region_ids: list[str] = OUTLIER_REGIONS,
) -> pd.DataFrame:
    outliers = set(str(region_id) for region_id in outlier_region_ids)
    rows: list[dict] = []
    for fold in folds:
        train_keep = ~fold["X_train"]["region_id"].astype(str).isin(outliers)
        test_keep = ~fold["X_test"]["region_id"].astype(str).isin(outliers)
        X_train = fold["X_train"].loc[train_keep]
        y_train = fold["y_train"].loc[X_train.index]
        X_test = fold["X_test"].loc[test_keep]
        y_test = fold["y_test"].loc[X_test.index]
        if X_test.empty or X_train.empty:
            continue
        y_bench = X_test["log_grp_lag1"].astype(float).to_numpy()
        for model_template in models:
            model = model_template.clone() if hasattr(model_template, "clone") else copy.deepcopy(model_template)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test.to_numpy(), y_pred, y_bench)
            rows.append(
                {
                    "model_name": model.name,
                    "fold": fold["fold"],
                    "test_year": fold["test_year"],
                    "n_obs": len(y_test),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def robustness_check_crisis(
    models: list[BaseForecaster],
    folds: list[dict],
    crisis_years: dict = CRISIS_YEARS,
) -> dict[str, pd.DataFrame]:
    processed_dir = Path("data/processed")
    crisis_set = {int(year) for years in crisis_years.values() for year in years}
    raw_rows = {"crisis": [], "normal": []}
    for fold in folds:
        period = "crisis" if int(fold["test_year"]) in crisis_set else "normal"
        for model in models:
            path = processed_dir / f"predictions_{model.name}_fold{fold['fold']}.parquet"
            if not path.exists():
                warnings.warn(f"Missing saved predictions: {path}", RuntimeWarning)
                continue
            pred_df = pd.read_parquet(path)
            metrics = compute_metrics(
                pred_df["y_true"].to_numpy(),
                pred_df["y_pred"].to_numpy(),
                pred_df["y_bench"].to_numpy(),
            )
            raw_rows[period].append(
                {
                    "model_name": model.name,
                    "fold": fold["fold"],
                    "test_year": fold["test_year"],
                    "n_obs": len(pred_df),
                    **metrics,
                }
            )

    result: dict[str, pd.DataFrame] = {}
    empty_cols = [
        "model_name",
        "RMSE_mean",
        "RMSE_std",
        "MAE_mean",
        "MAE_std",
        "OOS_R2_mean",
        "OOS_R2_std",
    ]
    for period, rows in raw_rows.items():
        result[period] = aggregate_results(pd.DataFrame(rows)) if rows else pd.DataFrame(columns=empty_cols)
    return result


def run_shap_analysis(
    best_model: BaseForecaster,
    X_test_all: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    region_meta: pd.DataFrame,
) -> dict:
    import shap

    estimator, X_model = _shap_estimator_and_matrix(best_model, X_test_all)
    explainer = shap.TreeExplainer(estimator)
    try:
        explanation = explainer(X_model)
        shap_values = np.asarray(explanation.values)
        base_values = np.asarray(explanation.base_values)
    except Exception:
        shap_values = np.asarray(explainer.shap_values(X_model))
        base_values = np.asarray(explainer.expected_value)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    feature_names = X_model.columns.tolist()
    global_importance = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)
        .sort_values(ascending=False)
    )

    region_ids = _region_ids_from_X(X_test_all)
    meta = pd.DataFrame({"region_id": region_ids}).merge(
        region_meta.drop_duplicates("region_id"),
        on="region_id",
        how="left",
    )
    group_importance = _shap_group_importance(
        shap_values=shap_values,
        feature_names=feature_names,
        feature_groups=feature_groups,
        region_types=meta["region_type"].fillna("unknown"),
    )
    local_examples, local_indices = _local_examples(meta)
    _check_economic_signs(shap_values, feature_names)

    return {
        "shap_values": shap_values,
        "global_importance": global_importance,
        "group_importance": group_importance,
        "local_examples": local_examples,
        "local_indices": local_indices,
        "feature_groups": feature_groups,
        "feature_names": feature_names,
        "X": X_model,
        "base_values": base_values,
        "explainer": explainer,
        "region_meta": meta,
    }


def error_analysis(
    cv_results: pd.DataFrame,
    predictions_dir: str = "data/processed/",
) -> dict[str, pd.DataFrame]:
    predictions_path = Path(predictions_dir)
    files = sorted(predictions_path.glob("predictions_*_fold*.parquet"))
    if not files:
        raise FileNotFoundError(f"No prediction parquet files found in {predictions_path}")
    pred_all = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    best_model = aggregate_results(cv_results).iloc[0]["model_name"]
    best_pred = pred_all[pred_all["model_name"] == best_model].copy()

    meta = _load_region_meta(predictions_path)
    by_region = (
        best_pred.groupby("region_id", as_index=False)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "MAE"})
        .sort_values("MAE", ascending=False)
    )
    by_region = by_region.merge(meta, on="region_id", how="left")

    by_year = (
        pred_all.groupby(["year", "model_name"], as_index=False)["abs_error"]
        .mean()
        .pivot(index="year", columns="model_name", values="abs_error")
        .rename(columns=lambda col: f"MAE_{col}")
        .reset_index()
    )

    best_with_meta = best_pred.merge(meta, on="region_id", how="left")
    if "region_type" in best_with_meta.columns and best_with_meta["region_type"].notna().any():
        by_type = (
            best_with_meta.groupby("region_type", as_index=False)["abs_error"]
            .mean()
            .rename(columns={"abs_error": "MAE"})
            .sort_values("MAE", ascending=False)
        )
    else:
        by_type = pd.DataFrame(columns=["region_type", "MAE"])

    top_regions = by_region.head(30)["region_id"]
    error_matrix = (
        best_pred[best_pred["region_id"].isin(top_regions)]
        .pivot_table(index="region_id", columns="year", values="abs_error", aggfunc="mean")
        .reindex(top_regions)
    )
    return {
        "by_region": by_region,
        "by_year": by_year,
        "by_type": by_type,
        "error_matrix": error_matrix,
        "best_model": best_model,
        "predictions": pred_all,
    }


def test_no_leakage(folds: list[dict]) -> None:
    for fold in folds:
        assert max(fold["train_years"]) < fold["test_year"]


def test_scaler_fit_on_train_only(folds: list[dict]) -> None:
    from sklearn.preprocessing import RobustScaler

    for fold in folds:
        scaler = fold.get("scaler")
        X_train = fold["X_train"]
        scaled_features = [col for col in fold.get("scaled_features", []) if col in X_train.columns]
        if not scaled_features:
            continue

        # Scaled X_train must have median ≈ 0 (RobustScaler property when fit on train)
        train_medians = X_train[scaled_features].median().abs()
        assert (train_medians < 1e-6).all(), (
            f"Fold {fold['fold']}: non-zero train medians after scaling "
            f"— scaler likely fit on wrong data: {train_medians[train_medians >= 1e-6].to_dict()}"
        )

        if scaler is None:
            continue

        # Inverse-transform X_train back to raw space, re-fit reference scaler,
        # and verify center_ matches: this proves scaler was fit on X_train only.
        X_train_raw_vals = scaler.inverse_transform(X_train[scaled_features])
        ref_scaler = RobustScaler().fit(X_train_raw_vals)
        np.testing.assert_allclose(
            ref_scaler.center_,
            scaler.center_,
            atol=1e-6,
            err_msg=(
                f"Fold {fold['fold']}: scaler.center_ does not match train-only median "
                "— scaler may have been fit on data outside the train fold."
            ),
        )


def test_naive_benchmark(folds: list[dict]) -> None:
    for fold in folds:
        predict_growth = fold.get("predict_growth", False)
        model = NaiveForecaster(predict_growth=predict_growth)
        model.fit(fold["X_train"], fold["y_train"])
        pred = model.predict(fold["X_test"])
        bench = np.asarray(
            fold.get("y_bench", fold["X_test"]["log_grp_lag1"].astype(float).to_numpy())
        )
        np.testing.assert_allclose(pred, bench, atol=1e-6)


def test_oos_r2_naive_is_zero(folds: list[dict]) -> None:
    for fold in folds:
        bench = np.asarray(
            fold.get("y_bench", fold["X_test"]["log_grp_lag1"].astype(float).to_numpy())
        )
        metrics = compute_metrics(fold["y_test"].to_numpy(), bench, bench)
        if not np.isnan(metrics["OOS_R2"]):
            assert abs(metrics["OOS_R2"]) < 1e-12


def test_all_regions_covered(df: pd.DataFrame, min_regions: int = 80) -> None:
    if "region_id" in df.columns:
        n_regions = df["region_id"].astype(str).nunique()
    elif "region_id" in df.index.names:
        n_regions = pd.Index(df.index.get_level_values("region_id")).astype(str).nunique()
    else:
        raise AssertionError("region_id is absent.")
    assert n_regions >= min_regions, f"Only {n_regions} regions found; expected >= {min_regions}."


def _columns_for_blocks(feature_groups: dict[str, list[str]], blocks: list[str]) -> list[str]:
    columns: list[str] = []
    for block in blocks:
        columns.extend(feature_groups.get(block, []))
    return list(dict.fromkeys(columns))


def _filter_features(X: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    keep = [col for col in selected if col in X.columns]
    for required in ["region_id", "log_grp_lag1"]:
        if required in X.columns and required not in keep:
            keep.insert(0, required)
    return X.loc[:, list(dict.fromkeys(keep))].copy()


def _shap_estimator_and_matrix(
    best_model: BaseForecaster, X: pd.DataFrame
) -> tuple[object, pd.DataFrame]:
    if isinstance(best_model, LGBMForecaster):
        return best_model.model_, _numeric_reindexed(X, best_model.feature_columns_)
    if isinstance(best_model, CatBoostForecaster):
        X_model = X.drop(columns=["region_id", "year"], errors="ignore")
        X_model = X_model.reindex(columns=best_model.feature_columns_, fill_value=0)
        X_model = X_model.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return best_model.model_, X_model
    if isinstance(best_model, GPBoostForecaster):
        return best_model.booster_, _numeric_reindexed(X, best_model.feature_columns_)
    raise TypeError("SHAP analysis supports LGBMForecaster, CatBoostForecaster and GPBoostForecaster.")


def _numeric_reindexed(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    X_num = X.reindex(columns=columns, fill_value=0)
    X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X_num.astype(float)


def _region_ids_from_X(X: pd.DataFrame) -> np.ndarray:
    if "region_id" in X.columns:
        return X["region_id"].astype(str).to_numpy()
    if "region_id" in X.index.names:
        return X.index.get_level_values("region_id").astype(str).to_numpy()
    return np.array(["unknown"] * len(X))


def _shap_group_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_groups: dict[str, list[str]],
    region_types: pd.Series,
) -> pd.DataFrame:
    feature_pos = {feature: idx for idx, feature in enumerate(feature_names)}
    rows: dict[str, dict[str, float]] = {}
    for region_type in sorted(region_types.dropna().unique()):
        mask = region_types.to_numpy() == region_type
        type_values: dict[str, float] = {}
        for block, columns in feature_groups.items():
            positions = [feature_pos[col] for col in columns if col in feature_pos]
            if positions:
                type_values[block] = float(np.abs(shap_values[mask][:, positions]).mean())
            else:
                type_values[block] = 0.0
        total = sum(type_values.values())
        if total > 0:
            type_values = {key: value / total for key, value in type_values.items()}
        rows[str(region_type)] = type_values
    return pd.DataFrame(rows).fillna(0.0)


def _local_examples(meta: pd.DataFrame) -> tuple[dict[str, str], dict[str, int]]:
    examples: dict[str, str] = {}
    indices: dict[str, int] = {}
    for region_type, group in meta.dropna(subset=["region_type"]).groupby("region_type", sort=False):
        row = group.iloc[-1]
        region_id = str(row["region_id"])
        region_name = str(row["region_name"]) if "region_name" in row and pd.notna(row["region_name"]) else region_id
        examples[region_id] = region_name
        indices[region_id] = int(row.name)
        if len(examples) >= 5:
            break
    return examples, indices


def _check_economic_signs(shap_values: np.ndarray, feature_names: list[str]) -> None:
    checks = {
        "log_grp_lag1": 1,
        "invest_total_lag1": 1,
        "unemployment_rate_lag1": -1,
    }
    feature_pos = {feature: idx for idx, feature in enumerate(feature_names)}
    for feature, expected_sign in checks.items():
        if feature not in feature_pos:
            continue
        mean_shap = float(np.nanmean(shap_values[:, feature_pos[feature]]))
        if expected_sign > 0 and mean_shap <= 0:
            warnings.warn(f"Unexpected non-positive mean SHAP for {feature}: {mean_shap:.6f}")
        if expected_sign < 0 and mean_shap >= 0:
            warnings.warn(f"Unexpected non-negative mean SHAP for {feature}: {mean_shap:.6f}")


def _load_region_meta(predictions_path: Path) -> pd.DataFrame:
    panel_path = predictions_path / "panel.parquet"
    if not panel_path.exists():
        return pd.DataFrame(columns=["region_id", "region_name", "region_type"])
    panel = pd.read_parquet(panel_path).reset_index(drop=True)
    cols = [col for col in ["region_id", "region_name", "region_type"] if col in panel.columns]
    if "region_id" not in cols:
        return pd.DataFrame(columns=["region_id", "region_name", "region_type"])
    return panel[cols].drop_duplicates("region_id")
