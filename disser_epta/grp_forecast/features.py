from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


PANEL_KEYS = ["region_id", "year"]


class PanelDataLoader:
    def __init__(
        self,
        raw_dir: str,
        target_col: str = "log_grp_real",
        rebuild_make_data: bool = False,
    ):
        self.raw_dir = Path(raw_dir)
        self.target_col = target_col
        self.rebuild_make_data = rebuild_make_data

    def load(self) -> pd.DataFrame:
        """
        Load panel data. If raw_dir contains make_data/build_dataset.py or
        make_data/panel_dataset.csv, use that builder output. Otherwise load
        CSV/XLSX files from raw_dir and merge them by (region_id, year).
        Returns a DataFrame indexed by (region_id, year), while keeping these
        keys as regular columns for downstream routines.
        """
        if self._is_make_data_dir(self.raw_dir):
            df = self._load_make_data_panel()
            return self._finalize_panel(df)

        files = sorted(
            [
                *self.raw_dir.glob("*.csv"),
                *self.raw_dir.glob("*.xlsx"),
                *self.raw_dir.glob("*.xls"),
            ]
        )
        if not files:
            raise FileNotFoundError(f"No CSV/XLSX files found in {self.raw_dir}")

        frames = [self._read_file(path) for path in files]
        df = self._merge_frames(frames)
        df = self.deflate(df)

        required = ["region_id", "year", self.target_col, "grp_nominal", "cpi_regional"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns after load: {missing}")

        return self._finalize_panel(df)

    def _finalize_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["region_id", "year", self.target_col, "grp_nominal", "cpi_regional"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns after load: {missing}")

        df["region_id"] = df["region_id"].astype(str)
        df["year"] = df["year"].astype(int)
        df = df.sort_values(PANEL_KEYS).set_index(PANEL_KEYS, drop=False)

        processed_dir = Path("data/processed")
        if processed_dir.exists():
            try:
                df.to_parquet(processed_dir / "panel.parquet")
            except Exception as exc:  # pragma: no cover - depends on parquet engine
                warnings.warn(f"Could not save panel.parquet: {exc}", RuntimeWarning)
        return df

    def _load_make_data_panel(self) -> pd.DataFrame:
        csv_path = self.raw_dir / "panel_dataset.csv"
        if csv_path.exists() and not self.rebuild_make_data:
            panel = pd.read_csv(csv_path)
        else:
            panel = self._run_make_data_builder()
            try:
                panel.to_csv(csv_path, index=False, encoding="utf-8-sig")
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"Could not save {csv_path}: {exc}", RuntimeWarning)
        return self._harmonize_make_data_schema(panel)

    def _run_make_data_builder(self) -> pd.DataFrame:
        import importlib.util

        builder_path = self.raw_dir / "build_dataset.py"
        if not builder_path.exists():
            raise FileNotFoundError(f"Missing make_data builder: {builder_path}")
        spec = importlib.util.spec_from_file_location("grp_make_data_build_dataset", builder_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import make_data builder from {builder_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "build_panel"):
            raise AttributeError(f"{builder_path} must define build_panel().")
        panel = module.build_panel()
        if not isinstance(panel, pd.DataFrame):
            raise TypeError("build_panel() must return pandas.DataFrame.")
        return panel

    def _harmonize_make_data_schema(self, panel: pd.DataFrame) -> pd.DataFrame:
        out = panel.copy()
        if "region" not in out.columns:
            raise ValueError("make_data panel must contain region column.")
        if "year" not in out.columns:
            raise ValueError("make_data panel must contain year column.")

        out = self._normalize_make_data_regions(out)
        if "policy_rate" not in out.columns:
            rates = self._load_make_data_policy_rates(out)
            if not rates.empty:
                out = out.merge(rates, on="year", how="left")

        out["region_name"] = out["region"].astype(str)
        out["region_code_numeric"] = out["region_id"] if "region_id" in out.columns else np.nan
        out["region_id"] = out["region"].astype(str)

        if "region_type" in out.columns:
            out["region_type"] = out["region_type"].replace(
                {
                    "resource": "raw_material",
                    "agro": "agricultural",
                }
            )
        else:
            out["region_type"] = "mixed"

        rename_map = {
            "vrp": "grp_nominal",
            "cpi": "cpi_regional",
            "investment": "invest_total",
            "mining_output": "mining",
            "manufacturing_output": "manufacturing",
            "freight_transport": "transport",
            "retail_trade": "retail",
            "avg_wage": "avg_wage_nominal",
            "avg_income_percapita": "income_nominal",
        }
        for src, dst in rename_map.items():
            if src in out.columns and dst not in out.columns:
                out[dst] = out[src]

        if "ipi" not in out.columns:
            components = [col for col in ["mining", "manufacturing"] if col in out.columns]
            out["ipi"] = out[components].mean(axis=1) if components else np.nan
        if "agriculture" not in out.columns:
            if "agriculture_output_collection" in out.columns:
                out["agriculture"] = out["agriculture_output_collection"]
            else:
                out["agriculture"] = np.nan
        if "mining_idx_collection" in out.columns:
            out["mining_index"] = out["mining_idx_collection"]
        if "manufacturing_idx_collection" in out.columns:
            out["manufacturing_index"] = out["manufacturing_idx_collection"]
        if "transfers" not in out.columns:
            out["transfers"] = np.nan
        if "budget_invest" not in out.columns:
            out["budget_invest"] = np.nan
        if "employment_rate" not in out.columns:
            if "employed" in out.columns and "population" in out.columns:
                population = pd.to_numeric(out["population"], errors="coerce").replace(0, np.nan)
                out["employment_rate"] = pd.to_numeric(out["employed"], errors="coerce") / population
            else:
                out["employment_rate"] = np.nan
        if "real_income" not in out.columns:
            if "income_nominal" in out.columns and "cpi_regional" in out.columns:
                cpi = pd.to_numeric(out["cpi_regional"], errors="coerce")
                factor = np.where(cpi.abs() > 10, cpi / 100.0, cpi)
                out["real_income"] = pd.to_numeric(out["income_nominal"], errors="coerce") / factor
            elif "real_income_idx" in out.columns:
                out["real_income"] = out["real_income_idx"]
            else:
                out["real_income"] = np.nan
        if "policy_rate_real" not in out.columns and "policy_rate" in out.columns and "cpi_regional" in out.columns:
            inflation = pd.to_numeric(out["cpi_regional"], errors="coerce") - 100
            if "regional_cpi_yoy" in out.columns:
                inflation = pd.to_numeric(out["regional_cpi_yoy"], errors="coerce").combine_first(inflation)
            out["policy_rate_real"] = pd.to_numeric(out["policy_rate"], errors="coerce") - inflation

        out = self.deflate(out)
        return out

    def _load_make_data_policy_rates(self, panel: pd.DataFrame) -> pd.DataFrame:
        builder_path = self.raw_dir / "build_dataset.py"
        if not builder_path.exists():
            return pd.DataFrame()
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("grp_make_data_build_dataset", builder_path)
            if spec is None or spec.loader is None:
                return pd.DataFrame()
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "load_cbr_policy_rates"):
                return pd.DataFrame()
            year_min = int(pd.to_numeric(panel["year"], errors="coerce").min())
            year_max = int(pd.to_numeric(panel["year"], errors="coerce").max())
            return module.load_cbr_policy_rates(year_min, year_max)
        except Exception as exc:
            warnings.warn(f"Could not load CBR policy rates: {exc}", RuntimeWarning)
            return pd.DataFrame()

    @staticmethod
    def _normalize_make_data_regions(panel: pd.DataFrame) -> pd.DataFrame:
        out = panel.copy()
        region = out["region"].astype(str)
        out = out[~region.str.startswith("в том числе", na=False)].copy()
        aggregate_pairs = [
            ("архангельская область", "архангельская область без автономного округа"),
            ("тюменская область", "тюменская область (без ао)"),
        ]
        for aggregate, residual in aggregate_pairs:
            if (out["region"] == aggregate).any() and (out["region"] == residual).any():
                out = out[out["region"] != aggregate].copy()
        out["region"] = out["region"].replace(
            {
                "архангельская область без автономного округа": "архангельская область",
                "тюменская область (без ао)": "тюменская область",
            }
        )
        out = out.sort_values(["region", "year"])
        return out.drop_duplicates(subset=["region", "year"], keep="first").reset_index(drop=True)

    @staticmethod
    def _is_make_data_dir(path: Path) -> bool:
        return (path / "build_dataset.py").exists() or (path / "panel_dataset.csv").exists()

    def deflate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert nominal GRP into real GRP using available real-volume data or CPI."""
        out = df.copy()
        if "cpi_regional" not in out.columns:
            fallback = next(
                (col for col in ["cpi_russia", "cpi_national", "cpi"] if col in out.columns),
                None,
            )
            if fallback is None:
                warnings.warn(
                    "cpi_regional is absent; using neutral CPI=100. "
                    "Replace this with regional or national CPI for final estimates.",
                    RuntimeWarning,
                )
                out["cpi_regional"] = 100.0
            else:
                out["cpi_regional"] = out[fallback]

        if "grp_nominal" in out.columns:
            if "vrp_idx" in out.columns and "region_id" in out.columns and "year" in out.columns:
                out["grp_real"] = self._chain_real_grp(out)
            else:
                cpi = pd.to_numeric(out["cpi_regional"], errors="coerce")
                cpi_median = cpi.dropna().median()
                cpi_factor = cpi / 100.0 if pd.notna(cpi_median) and abs(cpi_median) > 10 else cpi
                cpi_factor = cpi_factor.replace(0, np.nan)
                out["grp_real"] = pd.to_numeric(out["grp_nominal"], errors="coerce") / cpi_factor
            log_real = np.log(out["grp_real"].where(out["grp_real"] > 0))
            if self.target_col in out.columns:
                out[self.target_col] = out[self.target_col].fillna(log_real)
            else:
                out[self.target_col] = log_real
        elif self.target_col not in out.columns:
            raise ValueError("Either grp_nominal or target column must be present.")

        return out

    @staticmethod
    def _chain_real_grp(df: pd.DataFrame) -> pd.Series:
        work = df.copy()
        if any(key in work.index.names for key in PANEL_KEYS):
            work = work.reset_index(drop=True)
        work = work.sort_values(PANEL_KEYS)
        result = pd.Series(np.nan, index=work.index, dtype=float)
        for _, group in work.groupby(work["region_id"].astype(str), sort=False):
            values = []
            prev_real = np.nan
            for _, row in group.iterrows():
                nominal = pd.to_numeric(row.get("grp_nominal"), errors="coerce")
                idx = pd.to_numeric(row.get("vrp_idx"), errors="coerce")
                if not values:
                    real = nominal
                elif pd.notna(prev_real) and pd.notna(idx) and idx > 0:
                    real = prev_real * idx / 100.0
                else:
                    real = nominal
                values.append(real)
                prev_real = real
            result.loc[group.index] = values
        return result.reindex(df.index)

    def _read_file(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)

        df.columns = [str(col).strip() for col in df.columns]
        rename_map = {}
        if "region_id" not in df.columns:
            for candidate in ["region_code", "oktmo", "okato", "region"]:
                if candidate in df.columns:
                    rename_map[candidate] = "region_id"
                    break
        if "year" not in df.columns:
            for candidate in ["date_year", "period", "god"]:
                if candidate in df.columns:
                    rename_map[candidate] = "year"
                    break
        if rename_map:
            df = df.rename(columns=rename_map)

        missing = [col for col in PANEL_KEYS if col not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: missing key columns {missing}")

        df["region_id"] = df["region_id"].astype(str)
        df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
        return df

    @staticmethod
    def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
        merged = frames[0].copy()
        for frame in frames[1:]:
            overlap = [
                col
                for col in merged.columns.intersection(frame.columns)
                if col not in PANEL_KEYS
            ]
            merged = merged.merge(frame, on=PANEL_KEYS, how="outer", suffixes=("", "__new"))
            for col in overlap:
                new_col = f"{col}__new"
                merged[col] = merged[col].combine_first(merged[new_col])
                merged = merged.drop(columns=[new_col])
        return merged


class FeatureBuilder:
    REGION_TYPES = [
        "raw_material",
        "industrial",
        "agricultural",
        "service",
        "budget_dependent",
    ]

    PRODUCTION_RAW = [
        "ipi",
        "mining",
        "manufacturing",
        "agriculture",
        "construction",
        "transport",
        "retail",
    ]
    LABOUR_RAW = [
        "employment_rate",
        "unemployment_rate",
        "avg_wage_real",
        "real_income",
    ]
    BUDGET_RAW = [
        "budget_revenue",
        "budget_expenditure",
        "transfers",
        "budget_invest",
    ]
    MACRO_RAW = [
        "key_rate",
        "refinancing_rate",
        "policy_rate",
        "policy_rate_eop",
        "policy_rate_change",
        "policy_rate_real",
    ]
    PRICE_RAW = [
        "regional_cpi_yoy",
        "regional_cpi_yoy_eop",
        "regional_cpi_yoy_std",
        "regional_cpi_yoy_change",
        "usd_rub",
        "usd_rub_eop",
        "usd_rub_change",
        "usd_rub_pct_change",
        "usd_rub_volatility",
        "usd_rub_range",
    ]
    COLLECTION_RAW = [
        "working_age_share",
        "migration_growth_per_10k",
        "life_expectancy",
        "rural_population_share",
        "mining_idx_collection",
        "manufacturing_idx_collection",
        "agriculture_output_collection",
        "housing_commissioning_per_1000",
        "public_catering",
        "paid_services",
        "paid_services_idx",
        "road_freight_turnover",
        "road_freight_shipments",
        "household_ruble_credit_debt",
        "mortgage_ruble_credit_debt",
        "corporate_ruble_credit_debt",
        "min_food_basket_cost",
        "agriculture_output_collection_per_capita",
        "public_catering_per_capita",
        "paid_services_per_capita",
        "road_freight_turnover_per_capita",
        "road_freight_shipments_per_capita",
        "household_ruble_credit_debt_per_capita",
        "mortgage_ruble_credit_debt_per_capita",
        "corporate_ruble_credit_debt_per_capita",
        "agriculture_output_collection_to_grp",
        "public_catering_to_grp",
        "paid_services_to_grp",
        "household_ruble_credit_debt_to_grp",
        "mortgage_ruble_credit_debt_to_grp",
        "corporate_ruble_credit_debt_to_grp",
    ]

    def __init__(
        self,
        lag_depth: int = 2,
        target_col: str = "log_grp_real",
        missing_drop_threshold: float = 0.4,
    ):
        self.lag_depth = lag_depth
        self.target_col = target_col
        self.missing_drop_threshold = missing_drop_threshold
        self.scaler: RobustScaler | None = None
        self.numeric_features_: list[str] = []
        self.scaled_features_: list[str] = []
        self.scale_exempt_features_: list[str] = ["log_grp_lag1", "region_trend"]
        self.impute_values_: pd.Series | None = None
        self.dropped_missing_features_: list[str] = []
        self.missing_ratios_: pd.Series | None = None
        self.categorical_levels_: dict[str, list[str]] = {}
        self.feature_groups_: dict[str, list[str]] = self._default_feature_groups()

    def fit_transform(
        self, df: pd.DataFrame, train_mask: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        raw, y = self._build_raw_features(df)
        train_mask = self._align_mask(train_mask, raw.index)

        self.categorical_levels_ = self._fit_categorical_levels(raw, train_mask)
        X = self._encode_categories(raw, fit=True)
        X = X.replace([np.inf, -np.inf], np.nan)

        numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        if train_mask.sum() == 0:
            raise ValueError("train_mask contains no training rows.")

        self.missing_ratios_ = X.loc[train_mask, numeric_cols].isna().mean()
        protected = set(self.scale_exempt_features_ + ["year"])
        self.dropped_missing_features_ = [
            col
            for col, ratio in self.missing_ratios_.items()
            if ratio > self.missing_drop_threshold and col not in protected
        ]
        if self.dropped_missing_features_:
            X = X.drop(columns=self.dropped_missing_features_, errors="ignore")
            numeric_cols = [col for col in numeric_cols if col not in self.dropped_missing_features_]

        self.numeric_features_ = numeric_cols
        self.scaled_features_ = [
            col for col in numeric_cols if col not in self.scale_exempt_features_
        ]

        medians = X.loc[train_mask, numeric_cols].median(numeric_only=True)
        self.impute_values_ = medians.fillna(0.0)
        X.loc[:, numeric_cols] = X.loc[:, numeric_cols].fillna(self.impute_values_)

        self.scaler = RobustScaler()
        if self.scaled_features_:
            self.scaler.fit(X.loc[train_mask, self.scaled_features_])
            X.loc[:, self.scaled_features_] = self.scaler.transform(
                X.loc[:, self.scaled_features_]
            )

        self._refresh_feature_groups(X.columns)
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted encoder, imputer and scaler to a panel DataFrame."""
        if self.scaler is None or self.impute_values_ is None:
            raise RuntimeError("FeatureBuilder must be fitted before transform().")

        raw, _ = self._build_raw_features(df)
        X = self._encode_categories(raw, fit=False)
        X = X.replace([np.inf, -np.inf], np.nan)
        if self.dropped_missing_features_:
            X = X.drop(columns=self.dropped_missing_features_, errors="ignore")
        numeric_cols = [col for col in self.numeric_features_ if col in X.columns]
        X.loc[:, numeric_cols] = X.loc[:, numeric_cols].fillna(self.impute_values_)
        if self.scaled_features_:
            X.loc[:, self.scaled_features_] = self.scaler.transform(
                X.loc[:, self.scaled_features_]
            )
        return X

    def get_feature_groups(self) -> dict[str, list[str]]:
        """Return feature groups for ablation and interpretation."""
        return {key: list(value) for key, value in self.feature_groups_.items()}

    def _build_raw_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        work = self._standardize_panel(df)
        work = self._add_derived_columns(work)
        grouped = work.groupby(work["region_id"], sort=False)

        X = pd.DataFrame(index=work.index)
        X["region_id"] = work["region_id"].astype(str)

        X["log_grp_lag1"] = grouped[self.target_col].shift(1)
        X["log_grp_lag2"] = grouped[self.target_col].shift(2)
        X["grp_growth_lag1"] = X["log_grp_lag1"] - X["log_grp_lag2"]
        X["grp_ma3_lag1"] = grouped[self.target_col].transform(
            lambda s: s.shift(1).rolling(3, min_periods=3).mean()
        )

        for raw_col in self.PRODUCTION_RAW:
            X[f"{raw_col}_lag1"] = self._shift_or_nan(work, grouped, raw_col)

        X["invest_total_lag1"] = self._shift_or_nan(work, grouped, "invest_total")
        X["invest_per_capita_lag1"] = self._shift_or_nan(
            work, grouped, "invest_per_capita"
        )
        if "invest_total" in work.columns:
            invest_lag = grouped["invest_total"].shift(1)
        else:
            invest_lag = pd.Series(np.nan, index=work.index)
        if "grp_nominal" in work.columns:
            grp_lag = grouped["grp_nominal"].shift(1).replace(0, np.nan)
        else:
            grp_lag = np.exp(X["log_grp_lag1"]).replace(0, np.nan)
        X["invest_to_grp_lag1"] = invest_lag / grp_lag

        for raw_col in self.LABOUR_RAW:
            X[f"{raw_col}_lag1"] = self._shift_or_nan(work, grouped, raw_col)

        for raw_col in self.BUDGET_RAW:
            X[f"{raw_col}_lag1"] = self._shift_or_nan(work, grouped, raw_col)

        for raw_col in self.MACRO_RAW:
            X[f"{raw_col}_lag1"] = self._shift_or_nan(work, grouped, raw_col)

        for raw_col in self.PRICE_RAW:
            X[f"{raw_col}_lag1"] = self._shift_or_nan(work, grouped, raw_col)

        for raw_col in self.COLLECTION_RAW:
            X[f"{raw_col}_lag1"] = self._shift_or_nan(work, grouped, raw_col)

        for cat_col in ["federal_district", "region_type"]:
            if cat_col in work.columns:
                X[cat_col] = work[cat_col].astype("string").fillna("unknown")
            else:
                X[cat_col] = "unknown"

        years = work["year"].astype(int)
        X["crisis_2008"] = years.isin([2008, 2009]).astype(float).values
        X["crisis_2014"] = years.isin([2014, 2015]).astype(float).values
        X["crisis_2020"] = (years == 2020).astype(float).values
        X["crisis_2022"] = years.isin([2022, 2023]).astype(float).values

        # Variant 2: region-specific linear trend (years elapsed since first data year)
        first_year_by_region = work.groupby(work["region_id"])["year"].transform("min")
        X["region_trend"] = (work["year"] - first_year_by_region).astype(float).values

        y = pd.to_numeric(work[self.target_col], errors="coerce")
        y.name = self.target_col
        return X, y

    def _fit_categorical_levels(
        self, raw: pd.DataFrame, train_mask: pd.Series
    ) -> dict[str, list[str]]:
        levels: dict[str, list[str]] = {}
        for col in ["federal_district", "region_type"]:
            observed = (
                raw.loc[train_mask, col]
                .astype("string")
                .fillna("unknown")
                .dropna()
                .unique()
                .tolist()
            )
            observed = sorted(str(value) for value in observed if str(value) != "unknown")
            if col == "region_type":
                for value in self.REGION_TYPES:
                    if value not in observed:
                        observed.append(value)
            levels[col] = observed
        return levels

    def _encode_categories(self, raw: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = raw.copy()
        cat_features: dict[str, list[str]] = {}
        for col in ["federal_district", "region_type"]:
            levels = self.categorical_levels_.get(col, [])
            cat_features[col] = [f"{col}_{level}" for level in levels]
            values = X[col].astype("string").fillna("unknown")
            for level in levels:
                X[f"{col}_{level}"] = (values == level).astype(float)
            X = X.drop(columns=[col])

        if fit:
            self.feature_groups_["CATEGORICAL_BLOCK"] = (
                cat_features.get("federal_district", [])
                + cat_features.get("region_type", [])
            )
        return X

    def _refresh_feature_groups(self, columns: Iterable[str]) -> None:
        available = set(columns)
        groups = self._default_feature_groups()
        groups["CATEGORICAL_BLOCK"] = self.feature_groups_.get("CATEGORICAL_BLOCK", [])
        self.feature_groups_ = {
            name: [col for col in cols if col in available]
            for name, cols in groups.items()
        }

    def _default_feature_groups(self) -> dict[str, list[str]]:
        return {
            "LAG_BLOCK": [
                "log_grp_lag1",
                "log_grp_lag2",
                "grp_growth_lag1",
                "grp_ma3_lag1",
                "region_trend",
            ],
            "PRODUCTION_BLOCK": [f"{col}_lag1" for col in self.PRODUCTION_RAW],
            "INVESTMENT_BLOCK": [
                "invest_total_lag1",
                "invest_per_capita_lag1",
                "invest_to_grp_lag1",
            ],
            "LABOUR_BLOCK": [f"{col}_lag1" for col in self.LABOUR_RAW],
            "BUDGET_BLOCK": [f"{col}_lag1" for col in self.BUDGET_RAW],
            "MACRO_BLOCK": [f"{col}_lag1" for col in self.MACRO_RAW],
            "PRICE_BLOCK": [f"{col}_lag1" for col in self.PRICE_RAW],
            "COLLECTION_BLOCK": [f"{col}_lag1" for col in self.COLLECTION_RAW],
            "CATEGORICAL_BLOCK": [f"region_type_{value}" for value in self.REGION_TYPES],
            "SHOCK_BLOCK": [
                "crisis_2008",
                "crisis_2014",
                "crisis_2020",
                "crisis_2022",
            ],
        }

    def _standardize_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        if "region_id" not in work.columns:
            if "region_id" in work.index.names:
                work["region_id"] = work.index.get_level_values("region_id")
            else:
                raise ValueError("DataFrame must contain region_id column or index level.")
        if "year" not in work.columns:
            if "year" in work.index.names:
                work["year"] = work.index.get_level_values("year")
            else:
                raise ValueError("DataFrame must contain year column or index level.")
        if self.target_col not in work.columns:
            raise ValueError(f"DataFrame must contain target column {self.target_col}.")

        if any(key in work.index.names for key in PANEL_KEYS):
            work = work.reset_index(drop=True)
        work["region_id"] = work["region_id"].astype(str)
        work["year"] = pd.to_numeric(work["year"], errors="raise").astype(int)
        work = work.sort_values(PANEL_KEYS).set_index(PANEL_KEYS, drop=False)
        return work

    @staticmethod
    def _shift_or_nan(
        work: pd.DataFrame, grouped: pd.core.groupby.generic.DataFrameGroupBy, col: str
    ) -> pd.Series:
        if col not in work.columns:
            return pd.Series(np.nan, index=work.index)
        return grouped[col].shift(1)

    @staticmethod
    def _align_mask(mask: pd.Series, index: pd.Index) -> pd.Series:
        if not isinstance(mask, pd.Series):
            mask = pd.Series(mask, index=index)
        aligned = mask.reindex(index)
        return aligned.fillna(False).astype(bool)

    @staticmethod
    def _add_derived_columns(work: pd.DataFrame) -> pd.DataFrame:
        out = work.copy()
        if "invest_per_capita" not in out.columns:
            if "invest_total" in out.columns and "population" in out.columns:
                population = pd.to_numeric(out["population"], errors="coerce").replace(0, np.nan)
                out["invest_per_capita"] = pd.to_numeric(
                    out["invest_total"], errors="coerce"
                ) / population

        if "avg_wage_real" not in out.columns:
            wage_col = next(
                (col for col in ["avg_wage_nominal", "avg_wage"] if col in out.columns),
                None,
            )
            if wage_col is not None and "cpi_regional" in out.columns:
                cpi = pd.to_numeric(out["cpi_regional"], errors="coerce")
                cpi_median = cpi.dropna().median()
                factor = cpi / 100.0 if pd.notna(cpi_median) and abs(cpi_median) > 10 else cpi
                out["avg_wage_real"] = pd.to_numeric(out[wage_col], errors="coerce") / factor

        if "real_income" not in out.columns:
            income_col = next(
                (col for col in ["income_nominal", "nominal_income"] if col in out.columns),
                None,
            )
            if income_col is not None and "cpi_regional" in out.columns:
                cpi = pd.to_numeric(out["cpi_regional"], errors="coerce")
                cpi_median = cpi.dropna().median()
                factor = cpi / 100.0 if pd.notna(cpi_median) and abs(cpi_median) > 10 else cpi
                out["real_income"] = pd.to_numeric(out[income_col], errors="coerce") / factor

        if "population" in out.columns:
            population = pd.to_numeric(out["population"], errors="coerce").replace(0, np.nan)
            for col in [
                "agriculture_output_collection",
                "public_catering",
                "paid_services",
                "road_freight_turnover",
                "road_freight_shipments",
                "household_ruble_credit_debt",
                "mortgage_ruble_credit_debt",
                "corporate_ruble_credit_debt",
            ]:
                if col in out.columns and f"{col}_per_capita" not in out.columns:
                    out[f"{col}_per_capita"] = pd.to_numeric(out[col], errors="coerce") / population

        if "grp_nominal" in out.columns:
            grp = pd.to_numeric(out["grp_nominal"], errors="coerce").replace(0, np.nan)
            for col in [
                "agriculture_output_collection",
                "public_catering",
                "paid_services",
                "household_ruble_credit_debt",
                "mortgage_ruble_credit_debt",
                "corporate_ruble_credit_debt",
            ]:
                if col in out.columns and f"{col}_to_grp" not in out.columns:
                    out[f"{col}_to_grp"] = pd.to_numeric(out[col], errors="coerce") / grp
        return out


def build_folds(
    df: pd.DataFrame,
    min_train_years: int = 10,
    target_col: str = "log_grp_real",
    processed_dir: str = "data/processed",
    missing_drop_threshold: float = 0.4,
    predict_growth: bool = False,
) -> list[dict]:
    """
    Build expanding-window folds. FeatureBuilder is fitted separately inside
    every fold, so imputation, encoding and scaling use train rows only.
    """
    processed_path = Path(processed_dir)
    folds_path = processed_path / "folds"
    os.makedirs(folds_path, exist_ok=True)

    df_std = FeatureBuilder(target_col=target_col)._standardize_panel(df)
    try:
        os.makedirs(processed_path, exist_ok=True)
        df_std.to_parquet(processed_path / "panel.parquet")
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Could not save panel.parquet: {exc}", RuntimeWarning)

    # Pre-compute true growth rates from raw data (before any imputation) for Variant 1.
    # diff(1) is NaN for the first year of each region — correct, not imputed.
    _growth_raw = df_std.groupby(df_std["region_id"])[target_col].diff(1)

    # Pre-compute first-year flag: rows excluded from growth train set (Variant 1).
    _region_first_year = df_std.groupby(df_std["region_id"])["year"].transform("min")
    _is_first_year = pd.Series(
        (df_std["year"] == _region_first_year).values, index=df_std.index
    )
    # Pre-compute rows where the level target is not NaN (same filter as y_valid inside folds).
    _target_notna = pd.Series(df_std[target_col].notna().values, index=df_std.index)
    # Pre-compute rows where growth rate is not NaN — covers first years AND internal NaN gaps.
    _growth_notna = pd.Series(_growth_raw.notna().values, index=df_std.index)

    years = sorted(df_std["year"].dropna().astype(int).unique().tolist())
    if len(years) <= min_train_years:
        raise ValueError("Not enough years to create expanding-window folds.")

    folds: list[dict] = []
    for fold_id, test_year in enumerate(years[min_train_years:], start=1):
        train_years = [year for year in years if year < test_year]
        train_mask = pd.Series(df_std["year"].isin(train_years).values, index=df_std.index)
        test_mask = pd.Series((df_std["year"] == test_year).values, index=df_std.index)

        fb = FeatureBuilder(
            lag_depth=2,
            target_col=target_col,
            missing_drop_threshold=missing_drop_threshold,
        )
        # effective_train_mask must exactly equal the rows that end up in X_train
        # after all downstream filters (y_valid and growth), so that the scaler
        # is fit on the same distribution → fold["X_train"][scaled_features].median() == 0.
        if predict_growth:
            # growth filter removes: first years AND rows where prev year had NaN target
            effective_train_mask = train_mask & _target_notna & _growth_notna
        else:
            effective_train_mask = train_mask & _target_notna
        X_all, y_all = fb.fit_transform(df_std, train_mask=effective_train_mask)

        y_valid = y_all.notna()
        X_train = X_all.loc[train_mask & y_valid].copy()
        y_train = y_all.loc[train_mask & y_valid].copy()
        X_test = X_all.loc[test_mask & y_valid].copy()
        y_test = y_all.loc[test_mask & y_valid].copy()

        # Variant 1: transform target to growth rate (Δlog_grp = log_grp_t - log_grp_{t-1})
        # Use _growth_raw (diff on raw panel) so first-year NaN is NOT replaced by median imputation.
        if predict_growth:
            g_train = _growth_raw.reindex(y_train.index).dropna()
            g_test = _growth_raw.reindex(y_test.index).dropna()
            X_train = X_train.loc[g_train.index]
            X_test = X_test.loc[g_test.index]
            y_train = g_train
            y_test = g_test
            # Naive benchmark in growth space = mean historical growth rate from train
            y_bench: np.ndarray = np.full(len(y_test), float(y_train.mean()))
        else:
            y_bench = X_test["log_grp_lag1"].astype(float).to_numpy()

        fold = {
            "fold": fold_id,
            "train_years": train_years,
            "test_year": int(test_year),
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "y_bench": y_bench,
            "predict_growth": predict_growth,
            "train_mask": train_mask,
            "feature_groups": fb.get_feature_groups(),
            "scaled_features": list(fb.scaled_features_),
            "dropped_missing_features": list(fb.dropped_missing_features_),
            "missing_drop_threshold": fb.missing_drop_threshold,
            "scaler": fb.scaler,
        }
        with open(folds_path / f"fold_{fold_id}.pkl", "wb") as file:
            pickle.dump(fold, file)
        folds.append(fold)

    return folds
