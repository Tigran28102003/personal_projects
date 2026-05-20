from __future__ import annotations

import copy
import contextlib
import io
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from models import BaseForecaster

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_bench: np.ndarray,
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_bench = np.asarray(y_bench, dtype=float)

    errors = y_true - y_pred
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    denom = float(np.sum((y_true - y_bench) ** 2))
    if denom == 0:
        oos_r2 = np.nan
    else:
        oos_r2 = float(1.0 - np.sum(errors**2) / denom)
    return {"RMSE": rmse, "MAE": mae, "OOS_R2": oos_r2}


def run_cross_validation(
    models: list[BaseForecaster],
    folds: list[dict],
    tune: bool = True,
    n_trials: int = 50,
    save_predictions: bool = True,
    clean_output: bool = True,
    show_best: bool = True,
    exclude_models: list[str] | tuple[str, ...] | set[str] | None = None,
) -> pd.DataFrame:
    processed_dir = Path("data/processed")
    os.makedirs(processed_dir, exist_ok=True)

    excluded = set(exclude_models or [])
    models = [model for model in models if model.name not in excluded]

    rows: list[dict] = []
    total = len(models) * len(folds)
    best_row: dict | None = None
    progress = tqdm(
        total=total,
        desc="Cross-validation",
        dynamic_ncols=True,
        leave=True,
        position=0,
    )
    for fold in folds:
        X_train = fold["X_train"]
        y_train = fold["y_train"]
        X_test = fold["X_test"]
        y_test = fold["y_test"]
        _y_bench = fold.get("y_bench")
        if _y_bench is not None:
            y_bench = np.asarray(_y_bench, dtype=float)
        else:
            y_bench = X_test["log_grp_lag1"].astype(float).to_numpy()

        for model_template in models:
            model = model_template.clone() if hasattr(model_template, "clone") else copy.deepcopy(model_template)
            progress.set_description(f"{model.name} | fold {fold['fold']}")
            with _quiet_model_output(enabled=clean_output):
                if tune and callable(getattr(model, "tune", None)):
                    model.tune(X_train, y_train, n_trials=n_trials)
                else:
                    model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test.to_numpy(), y_pred, y_bench)
            row = {
                "model_name": model.name,
                "fold": fold["fold"],
                "test_year": fold["test_year"],
                "n_obs": len(y_test),
                **metrics,
            }
            rows.append(row)
            if best_row is None or row["RMSE"] < best_row["RMSE"]:
                best_row = row
            if show_best and best_row is not None:
                progress.set_postfix_str(
                    (
                        f"best={best_row['model_name']} "
                        f"fold={best_row['fold']} "
                        f"RMSE={best_row['RMSE']:.4f}"
                    ),
                    refresh=False,
                )

            if save_predictions:
                pred_df = _prediction_frame(
                    X_test=X_test,
                    y_true=y_test,
                    y_pred=y_pred,
                    y_bench=y_bench,
                    model_name=model.name,
                    fold=fold,
                )
                pred_df.to_parquet(
                    processed_dir / f"predictions_{model.name}_fold{fold['fold']}.parquet",
                    index=False,
                )
            progress.update(1)
    progress.close()
    if show_best and best_row is not None:
        print(
            "Best single fold: "
            f"{best_row['model_name']} | "
            f"fold={best_row['fold']} | "
            f"test_year={best_row['test_year']} | "
            f"RMSE={best_row['RMSE']:.4f} | "
            f"MAE={best_row['MAE']:.4f} | "
            f"OOS_R2={best_row['OOS_R2']:.4f}"
        )
    return pd.DataFrame(rows)


def aggregate_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["RMSE", "MAE", "OOS_R2"]
    agg = (
        cv_results.groupby("model_name", as_index=False)[metrics]
        .agg(["mean", "std"])
    )
    agg.columns = [
        "model_name" if col[0] == "model_name" else f"{col[0]}_{col[1]}"
        for col in agg.columns.to_flat_index()
    ]
    return agg.sort_values("RMSE_mean", ascending=True).reset_index(drop=True)


@contextlib.contextmanager
def _quiet_model_output(enabled: bool = True):
    if not enabled:
        yield
        return
    previous_optuna_verbosity = None
    try:
        import optuna

        previous_optuna_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        pass

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                yield
    finally:
        if previous_optuna_verbosity is not None:
            try:
                import optuna

                optuna.logging.set_verbosity(previous_optuna_verbosity)
            except Exception:
                pass


def _prediction_frame(
    X_test: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_bench: np.ndarray,
    model_name: str,
    fold: dict,
) -> pd.DataFrame:
    if "region_id" in X_test.columns:
        region_id = X_test["region_id"].astype(str).to_numpy()
    elif "region_id" in X_test.index.names:
        region_id = X_test.index.get_level_values("region_id").astype(str)
    else:
        region_id = np.array(["unknown"] * len(X_test))

    if "year" in X_test.columns:
        year = pd.to_numeric(X_test["year"], errors="coerce").fillna(fold["test_year"]).astype(int)
    elif "year" in X_test.index.names:
        year = X_test.index.get_level_values("year").astype(int)
    else:
        year = np.array([fold["test_year"]] * len(X_test))

    # log_grp_lag1 нужен для обратного преобразования в рубли:
    # GRP_t = exp(log_grp_lag1 + y_pred) — в тех же единицах, что и исходный ВРП
    log_grp_lag1 = (
        X_test["log_grp_lag1"].astype(float).to_numpy()
        if "log_grp_lag1" in X_test.columns
        else np.full(len(X_test), np.nan)
    )

    return pd.DataFrame(
        {
            "model_name": model_name,
            "fold": fold["fold"],
            "test_year": fold["test_year"],
            "region_id": region_id,
            "year": year,
            "y_true": y_true.astype(float).to_numpy(),
            "y_pred": np.asarray(y_pred, dtype=float),
            "y_bench": np.asarray(y_bench, dtype=float),
            "log_grp_lag1": log_grp_lag1,   # для back-transform в рубли
        }
    ).assign(abs_error=lambda df: (df["y_true"] - df["y_pred"]).abs())
