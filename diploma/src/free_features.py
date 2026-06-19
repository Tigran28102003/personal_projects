"""Free, backfillable exogenous features (EPIC 7 / T7.4).

Подключает БЕСПЛАТНЫЕ источники, которые реально бэкфиллятся на историю и закрывают
часть Tier-1/Tier-2 признаков из аудита, не требуя платных LOB/деривативных лент:

* **Order-flow imbalance** из Binance klines (поле `taker_buy_base_volume`) — не нужны
  тиковые данные: TFI = (2·taker_buy − volume)/volume ∈ [−1,1], кумулятивная дельта
  (CVD) и прокси «токсичности» потока (rolling |TFI|, грубый VPIN). Доступно на годы
  и на любом таймфрейме.
* **Funding rate** (Binance Futures `/fapi/v1/fundingRate`) — 8-часовой сеттлмент,
  годы истории.
* **Basis (perp − spot)** — из futures-klines и spot-klines, годы истории.
* **On-chain (daily)** из Coin Metrics Community API (free): сетевая активность —
  активные адреса, число транзакций, адреса с балансом, хешрейт, эмиссия предложения.
  (Realized cap / MVRV / SOPR / NVT в community-тарифе недоступны — это PRO.)

Каждый коннектор: retry-session + timeout + try/except + CSV-кэш + **graceful
degradation** (нет сети/данных → пустой фрейм + warning, пайплайн не падает — как
`DataMaker.fetch_*`).

ВАЖНО (анти-утечка): билдеры считают КОНТЕМПОРАННЫЕ значения и НЕ лагируют сами.
Единый источник правды о зазоре — `prep_returns`/`lag_exogenous` ниже по конвейеру:
имена этих фич нужно включить в exog-набор, чтобы они получили `shift(1)`. Funding
выравнивается на сетку бара через ffill от времени сеттлмента (будущее не подтекает).
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_SPOT_BASE = "https://api.binance.com"
_FUT_BASE = "https://fapi.binance.com"
_CM_BASE = "https://community-api.coinmetrics.io/v4"
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".free_features_cache")

# минуты на бар для пагинации klines
_INTERVAL_MIN = {'5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
FREQ_TO_INTERVAL = {'5min': '5m', 'hourly': '1h', 'daily': '1d'}

# имена сгенерированных фич (для расширения TOP_FEATURES и exog-лагирования)
ORDERFLOW_FEATURES = ('tfi', 'cvd', 'vpin')
DERIV_FEATURES = ('basis', 'basis_chg', 'funding', 'funding_chg')
ONCHAIN_FEATURES = ('active_addr_chg', 'tx_cnt_chg', 'addr_bal_chg', 'hashrate_chg', 'supply_chg')
FREE_FEATURE_NAMES = ORDERFLOW_FEATURES + DERIV_FEATURES + ONCHAIN_FEATURES


def _session(retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff_factor,
                  status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


# ---------------------------------------------------------------------------
# Лёгкий CSV-кэш сырья (pyarrow не требуется)
# ---------------------------------------------------------------------------

def _cache_load(key: str) -> Optional[pd.DataFrame]:
    path = os.path.join(_CACHE_DIR, f"{key}.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception:
            return None
    return None


def _cache_save(key: str, df: pd.DataFrame) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        df.to_csv(os.path.join(_CACHE_DIR, f"{key}.csv"))
    except Exception as e:  # noqa: BLE001 — кэш необязателен
        logger.warning("free_features: не удалось записать кэш %s: %s", key, e)


def _to_ms(ts) -> int:
    return int(pd.Timestamp(ts).tz_localize(None).value // 1_000_000)


# ---------------------------------------------------------------------------
# Низкоуровневые коннекторы (Binance, Coin Metrics)
# ---------------------------------------------------------------------------

def fetch_klines(symbol: str, interval: str, start, end, market: str = 'spot',
                 session: Optional[requests.Session] = None, use_cache: bool = True) -> pd.DataFrame:
    """Klines Binance (пагинация по 1000). Возвращает df, индексированный по UTC
    open_time (tz-naive), с колонками close/volume/quote_volume/n_trades/
    taker_buy_base/taker_buy_quote. На ошибке — пустой df с этими колонками.

    `market`: 'spot' (api.binance.com) или 'futures' (fapi.binance.com).

    NB (диплом): помимо `close` возвращаются также `open`/`high`/`low` — нужны для
    бэктеста со входом по `open[t+1]` и для range-based фич. Добавление колонок НЕ
    ломает orderflow/basis-билдеры (они читают только volume/taker/close).
    """
    cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'n_trades',
            'taker_buy_base', 'taker_buy_quote']
    key = f"klines_{market}_{symbol}_{interval}_{_to_ms(start)}_{_to_ms(end)}"
    if use_cache:
        cached = _cache_load(key)
        if cached is not None:
            return cached
    sess = session or _session()
    base = _FUT_BASE if market == 'futures' else _SPOT_BASE
    path = "/fapi/v1/klines" if market == 'futures' else "/api/v3/klines"
    step = _INTERVAL_MIN.get(interval, 60) * 60_000
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    rows = []
    try:
        cursor = start_ms
        for _ in range(10_000):  # жёсткий предел итераций
            if cursor >= end_ms:
                break
            r = sess.get(base + path, params={'symbol': symbol, 'interval': interval,
                         'startTime': cursor, 'endTime': end_ms, 'limit': 1000}, timeout=15)
            batch = r.json()
            if not isinstance(batch, list) or not batch:
                break
            rows.extend(batch)
            last_open = batch[-1][0]
            nxt = last_open + step
            if nxt <= cursor:
                break
            cursor = nxt
            if len(batch) < 1000:
                break
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning("free_features.fetch_klines(%s/%s) failed: %s", market, symbol, e)
        return pd.DataFrame(columns=cols)
    if not rows:
        return pd.DataFrame(columns=cols)
    arr = pd.DataFrame(rows)
    out = pd.DataFrame({
        'open': arr[1].astype(float),
        'high': arr[2].astype(float),
        'low': arr[3].astype(float),
        'close': arr[4].astype(float),
        'volume': arr[5].astype(float),
        'quote_volume': arr[7].astype(float),
        'n_trades': arr[8].astype(float),
        'taker_buy_base': arr[9].astype(float),
        'taker_buy_quote': arr[10].astype(float),
    })
    out.index = pd.to_datetime(arr[0].astype('int64'), unit='ms')  # open_time, UTC tz-naive
    out = out[~out.index.duplicated(keep='last')].sort_index()
    if use_cache:
        _cache_save(key, out)
    return out


def fetch_funding(symbol: str, start, end, session: Optional[requests.Session] = None,
                  use_cache: bool = True) -> pd.DataFrame:
    """Историческая ставка финансирования (Binance Futures). df индексирован по
    fundingTime (UTC tz-naive), колонка `funding`. На ошибке — пустой df."""
    key = f"funding_{symbol}_{_to_ms(start)}_{_to_ms(end)}"
    if use_cache:
        cached = _cache_load(key)
        if cached is not None:
            return cached
    sess = session or _session()
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    rows = []
    try:
        cursor = start_ms
        for _ in range(10_000):
            if cursor >= end_ms:
                break
            r = sess.get(_FUT_BASE + "/fapi/v1/fundingRate",
                         params={'symbol': symbol, 'startTime': cursor, 'endTime': end_ms, 'limit': 1000},
                         timeout=15)
            batch = r.json()
            if not isinstance(batch, list) or not batch:
                break
            rows.extend(batch)
            nxt = batch[-1]['fundingTime'] + 1
            if nxt <= cursor:
                break
            cursor = nxt
            if len(batch) < 1000:
                break
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning("free_features.fetch_funding(%s) failed: %s", symbol, e)
        return pd.DataFrame(columns=['funding'])
    if not rows:
        return pd.DataFrame(columns=['funding'])
    df = pd.DataFrame(rows)
    out = pd.DataFrame({'funding': df['fundingRate'].astype(float)})
    out.index = pd.to_datetime(df['fundingTime'].astype('int64'), unit='ms')
    out = out[~out.index.duplicated(keep='last')].sort_index()
    if use_cache:
        _cache_save(key, out)
    return out


def fetch_coinmetrics(metrics: Sequence[str], start, end, asset: str = 'btc',
                      session: Optional[requests.Session] = None, use_cache: bool = True) -> pd.DataFrame:
    """Дневные on-chain метрики Coin Metrics Community (free). df индексирован по дате,
    колонки = запрошенные метрики (float). На ошибке — пустой df."""
    key = f"cm_{asset}_{'-'.join(metrics)}_{pd.Timestamp(start).date()}_{pd.Timestamp(end).date()}"
    if use_cache:
        cached = _cache_load(key)
        if cached is not None:
            return cached
    sess = session or _session()
    try:
        r = sess.get(_CM_BASE + "/timeseries/asset-metrics", params={
            'assets': asset, 'metrics': ','.join(metrics), 'frequency': '1d',
            'start_time': pd.Timestamp(start).strftime('%Y-%m-%d'),
            'end_time': pd.Timestamp(end).strftime('%Y-%m-%d'), 'page_size': 10000,
        }, timeout=30)
        data = r.json().get('data', [])
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning("free_features.fetch_coinmetrics failed: %s", e)
        return pd.DataFrame(columns=list(metrics))
    if not data:
        return pd.DataFrame(columns=list(metrics))
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df['time']).dt.tz_localize(None).dt.normalize()
    keep = [m for m in metrics if m in df.columns]
    out = df[keep].apply(pd.to_numeric, errors='coerce').sort_index()
    if use_cache:
        _cache_save(key, out)
    return out


# ---------------------------------------------------------------------------
# Feature builders (детерминированы; НЕ лагируют — лаг ниже в prep_returns)
# ---------------------------------------------------------------------------

def orderflow_features(klines: pd.DataFrame, vpin_window: int = 50) -> pd.DataFrame:
    """Order-flow из klines (`taker_buy_base` = объём по тейкер-покупкам за бар).

    tfi  — trade-flow imbalance ``(2·buy − vol)/vol`` ∈ [−1, 1];
    cvd  — нормированная кумулятивная дельта объёма;
    vpin — прокси токсичности потока: rolling-среднее |tfi| (грубый VPIN; истинный
           VPIN считается по volume-бакетам — оставлено упрощение для дешёвой версии).
    """
    out = pd.DataFrame(index=klines.index)
    if klines.empty:
        return pd.DataFrame(columns=list(ORDERFLOW_FEATURES))
    vol = klines['volume'].replace(0, np.nan)
    delta = 2.0 * klines['taker_buy_base'] - klines['volume']  # buy − sell объёма
    out['tfi'] = (delta / vol).clip(-1, 1)
    out['cvd'] = delta.cumsum() / klines['volume'].cumsum().replace(0, np.nan)
    mp = min(vpin_window, max(1, vpin_window // 5))   # min_periods не больше окна
    out['vpin'] = out['tfi'].abs().rolling(vpin_window, min_periods=mp).mean()
    return out


def basis_features(perp_close: pd.Series, spot_close: pd.Series) -> pd.DataFrame:
    """Базис perp−spot: ``perp/spot − 1`` и его изменение (на общем индексе)."""
    idx = perp_close.index.intersection(spot_close.index)
    out = pd.DataFrame(index=idx)
    if len(idx) == 0:
        return pd.DataFrame(columns=['basis', 'basis_chg'])
    basis = perp_close.reindex(idx) / spot_close.reindex(idx).replace(0, np.nan) - 1.0
    out['basis'] = basis
    out['basis_chg'] = basis.diff()
    return out


def funding_on_grid(funding: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Выравнивание funding на сетку бара БЕЗ утечки: значение известно с момента
    `fundingTime`, поэтому ffill вперёд по объединённому индексу, затем reindex на
    `index`. `funding_chg` — изменение относительно прошлого сеттлмента."""
    out = pd.DataFrame(index=index)
    if funding.empty:
        out['funding'] = np.nan
        out['funding_chg'] = np.nan
        return out
    f = funding['funding']
    union = index.union(f.index).sort_values()
    aligned = f.reindex(union).ffill().reindex(index)
    out['funding'] = aligned
    # изменение от предыдущего значения сеттлмента (на сырых точках, затем на грид)
    chg = f.sort_index().diff()
    out['funding_chg'] = chg.reindex(union).ffill().reindex(index)
    return out


def onchain_features(cm: pd.DataFrame) -> pd.DataFrame:
    """On-chain фичи сетевой активности из Coin Metrics Community (free): относительные
    изменения активных адресов / числа транзакций / адресов с балансом / хешрейта /
    предложения. Берётся то, что вернул провайдер."""
    out = pd.DataFrame(index=cm.index)
    if cm.empty:
        return pd.DataFrame(columns=list(ONCHAIN_FEATURES))
    if 'AdrActCnt' in cm:
        out['active_addr_chg'] = cm['AdrActCnt'].pct_change()
    if 'TxCnt' in cm:
        out['tx_cnt_chg'] = cm['TxCnt'].pct_change()
    if 'AdrBalCnt' in cm:
        out['addr_bal_chg'] = cm['AdrBalCnt'].pct_change()
    if 'HashRate' in cm:
        out['hashrate_chg'] = cm['HashRate'].pct_change()
    if 'SplyCur' in cm:
        out['supply_chg'] = cm['SplyCur'].pct_change()
    return out


# ---------------------------------------------------------------------------
# Интеграция: добавить бесплатные фичи к частотному фрейму
# ---------------------------------------------------------------------------

# дефолтный набор community-метрик Coin Metrics (проверено: доступны бесплатно)
DEFAULT_CM_METRICS = ('AdrActCnt', 'TxCnt', 'AdrBalCnt', 'HashRate', 'SplyCur')


def attach_free_features(frame: pd.DataFrame, freq: str, symbol: str = 'BTCUSDT',
                         use_cache: bool = True, session: Optional[requests.Session] = None,
                         lag: int = 1) -> pd.DataFrame:
    """Возвращает копию `frame` с добавленными бесплатными фичами, выровненными по
    `frame.index` (DatetimeIndex, UTC tz-naive). При любой ошибке/отсутствии сети
    возвращает `frame` без изменений (graceful degradation).

    `lag`: сдвиг добавляемых фич на `lag` баров (по умолчанию 1) — лагируем **у
    источника**, как exog в `get_data`: значение в строке t = состояние на t−lag, то
    есть известно ДО t (анти-утечка). Поэтому `prep_returns` их повторно НЕ лагирует.

    `freq`: 'daily' | 'hourly' | '5min'. On-chain добавляется только для 'daily'.
    """
    if freq not in FREQ_TO_INTERVAL:
        logger.warning("attach_free_features: неизвестная частота %r — без изменений", freq)
        return frame
    if not isinstance(frame.index, pd.DatetimeIndex) or len(frame) == 0:
        return frame

    interval = FREQ_TO_INTERVAL[freq]
    start = frame.index.min()
    end = frame.index.max() + pd.Timedelta(days=1)
    sess = session or _session()
    try:
        spot = fetch_klines(symbol, interval, start, end, 'spot', sess, use_cache)
        perp = fetch_klines(symbol, interval, start, end, 'futures', sess, use_cache)
        feats = orderflow_features(spot).reindex(frame.index)
        if not perp.empty and not spot.empty:
            feats = feats.join(basis_features(perp['close'], spot['close']).reindex(frame.index))
        funding = fetch_funding(symbol, start, end, sess, use_cache)
        feats = feats.join(funding_on_grid(funding, frame.index))
        if freq == 'daily':
            cm = fetch_coinmetrics(DEFAULT_CM_METRICS, start, end, session=sess, use_cache=use_cache)
            feats = feats.join(onchain_features(cm).reindex(frame.index))
        # добавляем только РЕАЛЬНО заполненные колонки: иначе all-NaN фича, попав в
        # TOP_FEATURES, обнулит датасет через dropna в prep_returns.
        add = [c for c in FREE_FEATURE_NAMES if c in feats.columns and feats[c].notna().any()]
        if not add:
            logger.warning("attach_free_features: ни одной фичи не собрано — frame без изменений")
            return frame
        feats_add = feats[add]
        if lag:
            feats_add = feats_add.shift(lag)   # лаг у источника -> значение в t известно до t
        return frame.join(feats_add)
    except Exception as e:  # noqa: BLE001 — никогда не валим пайплайн из-за внешних данных
        logger.warning("attach_free_features(%s) failed (%s) — frame без изменений", freq, e)
        return frame
