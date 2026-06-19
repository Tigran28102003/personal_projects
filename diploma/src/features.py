"""Trailing, leakage-safe feature engineering + exogenous merge.

All features are computable at the signal bar ``t`` (close of t): BTC-derived features
are trailing/contemporaneous-at-close (the current return r_t = log(close_t/close_{t-1})
is known at close_t and is a valid predictor of the *forward* label over [t, t+H]).
Exogenous series are lagged at the source so a value in row t reflects state known
strictly before t.

Design notes:
* Indicators enter only in **normalised** form (MACD *histogram*/close, ATR/close, OBV
  change z-score) — raw MACD/OBV drift with the price scale (non-stationary). (R6)
* On-chain daily metrics are lagged ≥ their publication delay (≥1 day) BEFORE the
  hourly forward-fill, so a day-D value never appears inside day D. (R7)
* Cross-asset price *levels* are turned into log-returns (Yahoo ablation group only).
* Exogenous NaNs are forward-filled only (lag already applied), never back-filled.
* STL is NOT used as a feature (look-ahead).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    from . import free_features as ff
    from . import config
except ImportError:  # standalone
    import free_features as ff
    import config

logger = logging.getLogger(__name__)

RETURN_LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 168)
VOL_WINDOWS = (24, 72, 168)
MA_WINDOWS = (24, 72, 168)
REGIME_WINDOW = 720        # ~30d trailing window for regime/σ-rank/drawdown/long-trend (W2)
DD_WINDOWS = (168, 720)    # drawdown-from-trailing-max windows
WARMUP = max(max(RETURN_LAGS), max(VOL_WINDOWS), max(MA_WINDOWS), REGIME_WINDOW)  # 720 (W2)
ONCHAIN_PUB_LAG_DAYS = 1   # R7: ≥ publication delay before hourly ffill
FNG_LAG_DAYS = 1           # Fear&Greed strict lag

# era flags (deterministic by timestamp; redundant for trees via NaN, needed by NN/logreg)
PERP_ERA_START = pd.Timestamp("2019-09-08")   # Binance perp / funding inception
FNG_ERA_START = pd.Timestamp("2018-02-01")    # Fear&Greed history start


# ──────────────────────── indicator helpers (ported, normalised) ───────────
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + ag / al))


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram = MACD-line − signal-EMA (R6). Returned RAW; normalised by close
    in :func:`compute_features`."""
    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - sig


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _obv_change(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Изменение OBV за бар = знак движения · объём (стационарно; сырой OBV кумулятивен)."""
    return np.sign(close.diff()).fillna(0) * volume


def _zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / sd.replace(0, np.nan)


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """sin/cos календарь по timestamp прогноза (детерминирован -> НЕ лагируется)."""
    idx = pd.DatetimeIndex(df.index)
    two_pi = 2.0 * np.pi
    df["hour_sin"] = np.sin(two_pi * idx.hour / 24.0)
    df["hour_cos"] = np.cos(two_pi * idx.hour / 24.0)
    df["dow_sin"] = np.sin(two_pi * idx.dayofweek / 7.0)
    df["dow_cos"] = np.cos(two_pi * idx.dayofweek / 7.0)
    df["month_sin"] = np.sin(two_pi * idx.month / 12.0)
    df["month_cos"] = np.cos(two_pi * idx.month / 12.0)
    return df


# ───────────────────────────── price features ──────────────────────────────
def compute_features(ohlc: pd.DataFrame, prefix: str = "BTC_") -> pd.DataFrame:
    """Трейлинговые фичи из OHLCV (без экзогенки). Возвращает DataFrame фич на индексе
    ``ohlc`` (БЕЗ самих OHLC-колонок — их хранит/джойнит data_io)."""
    close = ohlc[f"{prefix}Close"].astype(float)
    high = ohlc.get(f"{prefix}High", close).astype(float)
    low = ohlc.get(f"{prefix}Low", close).astype(float)
    volume = ohlc[f"{prefix}Volume"].astype(float)

    r = np.log(close).diff()
    f = pd.DataFrame(index=ohlc.index)
    f["ret"] = r
    for lag in RETURN_LAGS:
        f[f"ret_lag{lag}"] = r.shift(lag)
    for w in VOL_WINDOWS:
        f[f"vol{w}"] = r.rolling(w, min_periods=w // 2).std()
    f["vol_ratio_24_168"] = f["vol24"] / f["vol168"].replace(0, np.nan)
    f["vol_ratio_24_72"] = f["vol24"] / f["vol72"].replace(0, np.nan)
    # нормированные индикаторы
    f["rsi"] = _rsi(close) / 100.0
    f["macd_hist"] = _macd_hist(close) / close            # нормировка на цену
    f["atr_close"] = _atr(high, low, close) / close
    obv_chg = _obv_change(close, volume)
    f["obv_chg_z"] = _zscore(obv_chg, 168)
    # объём
    logv = np.log1p(volume)
    f["log_volume"] = logv
    f["log_volume_chg"] = logv.diff()
    f["log_volume_z"] = _zscore(logv, 168)
    # MA как отношения
    for w in MA_WINDOWS:
        f[f"ma_ratio_{w}"] = close / close.rolling(w, min_periods=w // 2).mean() - 1.0

    # ── РЕЖИМ/СЕЗОННОСТЬ (W2; всё СТРОГО каузально — только прошлое) ──
    idx = pd.DatetimeIndex(ohlc.index)
    absr = r.abs()
    # 1. Диурнальная климатология волатильности: трейлинг-среднее |r| по бакету hour×dow.
    #    shift(1).expanding().mean() ВНУТРИ бакета -> только прошлые бары того же слота,
    #    себя исключает (НЕ groupby.mean по всей выборке — то была бы утечка будущего).
    f["vol_clim_hd"] = absr.groupby([idx.hour, idx.dayofweek]).transform(
        lambda s: s.shift(1).expanding().mean())
    # 2. Режим σ: трейлинг z-score текущей vol24 за ~720ч (насколько экстремальна σ).
    f["vol24_z720"] = _zscore(f["vol24"], REGIME_WINDOW)
    # 3. Просадка от трейлинг-максимума (стресс/медведь).
    for w in DD_WINDOWS:
        f[f"dd_{w}"] = close / close.rolling(w, min_periods=w // 2).max() - 1.0
    # 4. Длинный тренд/σ (медленный режим, секулярный сдвиг волатильности).
    f["ma_ratio_720"] = close / close.rolling(REGIME_WINDOW, min_periods=REGIME_WINDOW // 2).mean() - 1.0
    f["vol720"] = r.rolling(REGIME_WINDOW, min_periods=REGIME_WINDOW // 2).std()

    # era-флаги (детерминированы по timestamp; деревьям избыточны, но нужны NN/логрегу)
    f["perp_era"] = (idx >= PERP_ERA_START).astype(float)
    f["fng_era"] = (idx >= FNG_ERA_START).astype(float)

    # календарь (детерминирован)
    f = add_calendar(f)
    return f


# ───────────────────────────── exogenous merge ─────────────────────────────
def _fetch_fear_greed(session=None) -> pd.Series:
    """Дневной Fear&Greed индекс (alternative.me). Series по дате (UTC, normalize).
    Пустой Series при ошибке (graceful)."""
    import requests
    url = "https://api.alternative.me/fng/?limit=0"
    try:
        sess = session or ff._session()
        data = sess.get(url, timeout=15).json()["data"]
        df = pd.DataFrame(data)
        idx = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
        return pd.Series(df["value"].astype(float).values, index=idx, name="fng").sort_index()
    except Exception as e:  # noqa: BLE001
        logger.warning("fetch_fear_greed failed: %s", e)
        return pd.Series(dtype=float, name="fng")


def _daily_to_hourly_lagged(daily: pd.Series, hourly_index: pd.Index, lag_days: int) -> pd.Series:
    """Лагировать дневной ряд на `lag_days` дней, затем ffill на часовую сетку.

    Сдвиг ВЫПОЛНЯЕТСЯ ДО ffill (R7): значение дня D становится доступно только начиная
    со дня D+lag_days, поэтому внутрь дня D протекает лишь прошлое."""
    if daily.empty:
        return pd.Series(np.nan, index=hourly_index, name=daily.name)
    d = daily.sort_index().shift(lag_days)                  # сдвиг по дням
    union = pd.DatetimeIndex(hourly_index).union(d.index).sort_values()
    return d.reindex(union).ffill().reindex(hourly_index)


def attach_exogenous(frame: pd.DataFrame, symbol: str = "BTCUSDT", use_cache: bool = True,
                     include_orderflow_basis_funding: bool = True,
                     include_onchain: bool = True, include_fng: bool = True) -> pd.DataFrame:
    """Добавить крипто-нативную экзогенку (дефолтный длинноисторичный набор).

    * orderflow (tfi/cvd/vpin) + basis + funding — через
      :func:`free_features.attach_free_features` (лаг 1 бар у источника, анти-утечка);
    * on-chain (CoinMetrics daily) — лаг ≥1 день ДО hourly ffill (R7);
    * Fear&Greed — дневной, strict-lag 1 день.

    Любая сетевая ошибка -> соответствующая группа просто не добавляется (graceful).
    История order-flow (tfi/cvd) уходит в 2017 (taker-buy klines); vpin — rolling от tfi
    (тоже длинная). OI как длинная фича НЕ используется (~30 дней бесплатной истории).
    """
    out = frame.copy()
    idx = out.index
    sess = ff._session()
    if include_orderflow_basis_funding:
        try:
            out = ff.attach_free_features(out, freq="hourly", symbol=symbol,
                                          use_cache=use_cache, session=sess, lag=1)
        except Exception as e:  # noqa: BLE001
            logger.warning("attach_exogenous: orderflow/basis/funding failed: %s", e)
    if include_onchain:
        try:
            cm = ff.fetch_coinmetrics(ff.DEFAULT_CM_METRICS, idx.min(),
                                      idx.max() + pd.Timedelta(days=2),
                                      session=sess, use_cache=use_cache)
            onc = ff.onchain_features(cm)
            for col in onc.columns:
                out[col] = _daily_to_hourly_lagged(onc[col], idx, ONCHAIN_PUB_LAG_DAYS)
        except Exception as e:  # noqa: BLE001
            logger.warning("attach_exogenous: on-chain failed: %s", e)
    if include_fng:
        try:
            fng = _fetch_fear_greed(sess)
            out["fng"] = _daily_to_hourly_lagged(fng, idx, FNG_LAG_DAYS)
            out["fng_chg"] = _daily_to_hourly_lagged(fng.diff(), idx, FNG_LAG_DAYS)
        except Exception as e:  # noqa: BLE001
            logger.warning("attach_exogenous: fng failed: %s", e)
    # 5. (опц., W2) трейлинг z-score стресс-прокси funding/basis; NaN в pre-perp -> ок (деревья)
    for col in ("funding", "basis"):
        if col in out.columns:
            out[f"{col}_z720"] = _zscore(out[col], REGIME_WINDOW)
    return out


def attach_yahoo_crossassets(frame: pd.DataFrame) -> pd.DataFrame:
    """ОПЦИОНАЛЬНАЯ группа Yahoo кросс-активов (для ablation на ~729-дн. окне).

    НЕ входит в дефолтную модель: Yahoo 1h ограничен ~729 днями и схлопнул бы обучающее
    окно. Уровни цен -> лог-доходности; лаг 1 бар (экзогенка). Импортирует тяжёлый
    get_data лениво. На ошибке возвращает frame без изменений."""
    try:
        try:
            from . import get_data as gd
        except ImportError:
            import get_data as gd
        dm = gd.DataMaker(end_date=config.DATA_END)
        yh = dm.get_yahoo_data("1h")
        close_cols = [c for c in yh.columns if c.endswith("_Close")]
        lr = np.log(yh[close_cols].replace(0, np.nan)).diff().shift(1)  # лог-доходности, лаг 1
        lr.columns = [f"yh_{c.replace('_Close','')}_lr" for c in close_cols]
        return frame.join(lr.reindex(frame.index))
    except Exception as e:  # noqa: BLE001
        logger.warning("attach_yahoo_crossassets failed (%s) — frame unchanged", e)
        return frame
