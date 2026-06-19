"""The data switch + target selector.

Two entry points:

* :func:`load_or_build_features` — caches **features + OHLC** (never the label) to
  ``data/processed/features_hourly.csv`` (+ parquet). ``rebuild=False`` and the cache
  exists → read; otherwise build (fetch OHLCV → engineer features → attach exogenous →
  drop warmup) and save, plus a raw OHLCV snapshot with a content hash + row count.
* :func:`build_dataset` — reads cached features and builds the {-1,0,+1} label ON THE
  FLY via :func:`labeling.make_target`. Because the label is not cached, switching
  ``TARGET_TYPE`` is instant; a rebuild is only needed when the *features* change.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from . import config, features, labeling
    from .get_data import fetch_binance_ohlcv
except ImportError:  # standalone
    import config, features, labeling
    from get_data import fetch_binance_ohlcv

logger = logging.getLogger(__name__)

OHLC_COLS = ["BTC_Open", "BTC_High", "BTC_Low", "BTC_Close", "BTC_Volume"]


def _start_for_history(months: Optional[int]) -> str:
    if months is None:
        return config.DATA_START
    return (pd.Timestamp(config.DATA_END) - pd.DateOffset(months=months)).strftime("%Y-%m-%d")


def _save_raw_snapshot(ohlc: pd.DataFrame) -> dict:
    """Сохранить сырой OHLCV-снапшот + манифест с хешем содержимого и числом строк."""
    config.RAW.mkdir(parents=True, exist_ok=True)
    ohlc.to_parquet(config.RAW_SNAPSHOT)
    h = hashlib.sha256(pd.util.hash_pandas_object(ohlc, index=True).values.tobytes()).hexdigest()
    manifest = {"rows": int(len(ohlc)), "sha256": h,
                "start": str(ohlc.index.min()), "end": str(ohlc.index.max()),
                "data_end_cfg": config.DATA_END}
    config.RAW_MANIFEST.write_text(json.dumps(manifest, indent=2))
    logger.info("raw snapshot: %d rows, sha256=%s", manifest["rows"], h[:12])
    return manifest


def reindex_full_grid(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Реиндексировать OHLCV на ПОЛНУЮ часовую сетку и ffill (W3): пропущенные бары
    (~128 на полной истории) не должны заставлять `r` считаться через дыру за более
    длинный интервал. Объём на gap-баре — ffill (НЕ 0), чтобы log1p/diff/z остались
    конечными (V2). Добавляет флаг ``gap`` (1 = бар был достроен)."""
    full = pd.date_range(ohlc.index.min(), ohlc.index.max(), freq="h")
    out = ohlc.reindex(full)
    gap = out["BTC_Close"].isna()
    out[OHLC_COLS] = out[OHLC_COLS].ffill()          # цена И объём ffill -> конечные фичи
    out["gap"] = gap.astype(float)
    n_gap = int(gap.sum())
    if n_gap:
        logger.info("reindex full grid: +%d gap bars ffilled (%.2f%%)",
                    n_gap, 100 * n_gap / len(full))
    out.index.name = ohlc.index.name or "Date"
    return out


def _check_finite(frame: pd.DataFrame) -> pd.DataFrame:
    """V2: фичи не должны содержать ±inf. Если нашли — заменить на NaN (деревья кодируют)
    и предупредить (не валим многочасовую сборку); строгий assert — в pytest на фикстуре."""
    num = frame.select_dtypes(include=[np.number])
    inf_mask = np.isinf(num.to_numpy())
    if inf_mask.any():
        bad = list(num.columns[inf_mask.any(axis=0)])
        logger.warning("finite-check: ±inf replaced with NaN in %s", bad)
        frame[num.columns] = num.replace([np.inf, -np.inf], np.nan)
    return frame


def build_frame(ohlc: pd.DataFrame, attach_exo: bool = True) -> pd.DataFrame:
    """OHLC + трейлинговые фичи + (опц.) экзогенка; full-grid reindex; warmup отброшен."""
    ohlc = reindex_full_grid(ohlc)                   # W3: полная сетка + gap-флаг
    gap = ohlc["gap"]
    feats = features.compute_features(ohlc)
    frame = ohlc[OHLC_COLS].join(feats)
    frame["gap"] = gap                               # gap — фича (попадёт в X)
    if attach_exo:
        frame = features.attach_exogenous(frame)
    # отбросить warmup длинных окон (первые WARMUP баров — NaN в BTC-фичах) + лог потерь
    before = len(frame)
    frame = frame.iloc[features.WARMUP:]
    logger.info("warmup drop: %d -> %d rows (-%d, окно %d)", before, len(frame),
                before - len(frame), features.WARMUP)
    return _check_finite(frame)


def load_or_build_features(rebuild: bool = False, history_months: Optional[int] = None,
                           attach_exo: bool = True) -> pd.DataFrame:
    """Загрузить кэш фич (+OHLC) или собрать заново. Метка НЕ кэшируется."""
    if not rebuild and config.FEATURES_PARQUET.exists():
        logger.info("features cache hit: %s", config.FEATURES_PARQUET)
        return pd.read_parquet(config.FEATURES_PARQUET)
    if not rebuild and config.FEATURES_CACHE.exists():
        return pd.read_csv(config.FEATURES_CACHE, index_col=0, parse_dates=True)

    start = _start_for_history(history_months if history_months is not None else config.history_months())
    logger.info("building features: %s -> %s", start, config.DATA_END)
    ohlc = fetch_binance_ohlcv(config.SYMBOL, config.INTERVAL, start=start, end=config.DATA_END)
    if ohlc.empty:
        raise RuntimeError("fetch_binance_ohlcv returned empty (network?) — "
                           "use a synthetic fixture for the smoke test")
    _save_raw_snapshot(ohlc)
    frame = build_frame(ohlc, attach_exo=attach_exo)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.FEATURES_CACHE)
    try:
        frame.to_parquet(config.FEATURES_PARQUET)
    except Exception as e:  # noqa: BLE001
        logger.warning("parquet save failed: %s", e)
    return frame


def split_frame(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разделить кэш на (OHLC, X-фичи). Сырые ценовые уровни (OHLC) в X НЕ попадают
    (нестационарны) — они нужны только для метки и бэктеста."""
    ohlc = frame[[c for c in OHLC_COLS if c in frame.columns]].copy()
    X = frame.drop(columns=[c for c in OHLC_COLS if c in frame.columns]).copy()
    return ohlc, X


def make_labels(ohlc: pd.DataFrame, target_type: str, k: Optional[float] = None,
                threshold: Optional[float] = None):
    return labeling.make_target(ohlc, target_type=target_type, k=k, threshold=threshold)


def build_dataset(rebuild: bool = False, target_type: str = "pointwise",
                  k: Optional[float] = None, threshold: Optional[float] = None,
                  history_months: Optional[int] = None,
                  attach_exo: bool = True, frame: Optional[pd.DataFrame] = None,
                  drop_cols: Optional[list] = None):
    """Собрать ``(X, y3, fwd_ret, ohlc, exit_info)`` для выбранного ``target_type``.

    Метка строится на лету; строки с неопределённой меткой (хвост горизонта, warmup σ)
    отбрасываются. ``frame`` можно передать напрямую (напр. синтетический фикстур для
    smoke-test) — тогда кэш не читается/не пишется."""
    if frame is None:
        frame = load_or_build_features(rebuild=rebuild, history_months=history_months,
                                       attach_exo=attach_exo)
    ohlc, X = split_frame(frame)
    if drop_cols:  # ablation hook (напр. без funding/basis) — НЕ удаление, а аблация
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    y3, fwd_ret, exit_info = make_labels(ohlc, target_type, k, threshold)
    # выровнять и отбросить строки с NaN-меткой
    valid = y3.notna()
    X, y3 = X.loc[valid], y3.loc[valid].astype(int)
    fwd_ret = fwd_ret.loc[valid]
    ohlc = ohlc.loc[valid]
    exit_info = exit_info.loc[valid] if exit_info is not None else None
    logger.info("dataset[%s%s]: %d rows, %d features, classes=%s", target_type,
                f", k={k}" if k else "", len(X), X.shape[1],
                dict(zip(*np.unique(y3, return_counts=True))))
    return X, y3, fwd_ret, ohlc, exit_info
