import pandas as pd
import numpy as np

from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import holidays

from statsmodels.tsa.seasonal import STL

import contextlib
import io
import logging

import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from curl_cffi import requests as curl_requests
from fredapi import Fred
from tqdm.auto import tqdm

import os
from typing import Optional, Literal

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def _make_session(retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """MACD line: EMA(fast) - EMA(slow)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


class DataMaker:
    def __init__(self, start_date: str = "2018-01-01", end_date: Optional[str] = None, fred_api_key: Optional[str] = None) -> None:
        """
        библиотека для сбора данных для предсказания BTC
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        self.fred_api_key = fred_api_key or os.getenv("FRED_API_KEY", "YOUR_API_KEY_HERE")
        self.fred = Fred(api_key=self.fred_api_key)
        self._session = _make_session()
        # curl_cffi-сессия с имитацией TLS-отпечатка Chrome — обязательна для
        # yf.download(): обычная requests.Session приводит к тому, что Yahoo
        # молча возвращает пустые данные по любому тикеру.
        self._yahoo_session = curl_requests.Session(impersonate="chrome")

        self.yahoo_series = {
            # Криптовалюты и стейблкоины
            "BTC-USD": "BTC",
            "ETH-USD": "ETH",
            "XRP-USD": "XRP",
            "DOGE-USD": "Dogecoin",
            "ADA-USD": "Cardano",
            "LTC-USD": "Litecoin",
            "LINK-USD": "Chainlink",
            "BCH-USD": "Bitcoin_Cash",
            "XLM-USD": "Stellar",
            "TRX-USD": "TRON",
            "ETC-USD": "Ethereum_Classic",
            "USDT-USD": "Tether",

            # Металлы и сырье
            "GC=F": "Gold_Future",
            "SI=F": "Silver_Future",
            "PL=F": "Platinum_Future",
            "HG=F": "Copper_Future",
            "CL=F": "WTI_Future",
            "BZ=F": "Brent_Oil_Future",
            "NG=F": "Natural_Gas_Future",

            # Индексы и ETF
            "^GSPC": "S&P_500",
            "^NDX": "NASDAQ_100",
            "^DJI": "Dow_Jones",
            "^VIX": "VIX",
            "QQQ": "Invesco_QQQ",
            "ARKK": "ARKK_ETF",
            "ARKW": "ARK_NextGen_Internet_ETF",
            "SOXX": "iShares_Semiconductor_ETF",
            "SMH": "VanEck_Semiconductor_ETF",
            "XLF": "SPDR_Financials",
            "XLE": "SPDR_Energy",
            "XLY": "SPDR_Consumer_Discretionary",
            "XLK": "SPDR_Tech",
            "XLI": "SPDR_Industrials",
            "XLV": "SPDR_HealthCare",
            "XLC": "SPDR_Communications",
            "XLB": "SPDR_Materials",
            "XLRE": "SPDR_RealEstate",

            # Валютные пары
            "EURUSD=X": "EUR/USD",
            "GBPUSD=X": "GBP/USD",
            "JPYUSD=X": "USD/JPY",
            "AUDUSD=X": "AUD/USD",
            "CHFUSD=X": "USD/CHF",
            "USDCAD=X": "USD/CAD",

            # Акции и компании, чувствительные к BTC/технорынку
            "MSTR": "MicroStrategy",
            "RIOT": "Riot_Blockchain",
            "MARA": "Marathon_Digital",
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Google",
            "AMZN": "Amazon",
            "META": "Meta",
            "TSLA": "Tesla",
            "NVDA": "NVIDIA",
            "AMD": "AMD",
            "INTC": "Intel",
            "ADBE": "Adobe",

        }

        self.fredapi_series = {
            "Unemployment_Rate": "UNRATE",
            "Non_Farm_Payrolls": "PAYEMS",
            "Initial_Jobless_Claims": "ICSA",
            "CPI": "CPIAUCSL",
            "PCE_Price_Index": "PCEPILFE",
            "Consumer_Confidence_Index": "UMCSENT",
            "Retail_Sales": "RSAFS",
            "Housing_Starts": "HOUST",
            "Building_Permits": "PERMIT",
            "M2_Money_Supply": "M2SL",
            "Real_Interest_Rate": "DFII10",
            "Fed_Funds_Rate": "FEDFUNDS",
            "Reverse_Repo_Volume": "RRPONTSYD",
            "Debt": "GFDEBTN",
            "Real_GDP": "GDPC1"
        }

    def download_fred_data(self) -> pd.DataFrame:
        """
            Скачивает и объединяет данные с FRED по заданным сериям.
        """
        data = pd.DataFrame()
        for name, series_id in self.fredapi_series.items():
            try:
                series = self.fred.get_series(series_id, observation_start=self.start_date, observation_end=self.end_date)
                df = pd.DataFrame({name: series})
                data = pd.concat([data, df], axis=1)
            except Exception as e:
                self.log(f"Ошибка с {name} ({series_id}): {e}")
        data.index.name = 'Date'
        return data.ffill()

    def get_yahoo_data(self, interval: Literal['5m', '1h', '60m', '1d', '1wk'] = '1d') -> pd.DataFrame:
        """
            Получает данные с Yahoo Finance по заданной частоте.

            Все тикеры запрашиваются одним батч-вызовом `yf.download` с
            curl_cffi-сессией (имитация TLS-отпечатка Chrome,
            `self._yahoo_session`) и `threads=True`. Такая сессия не
            блокируется анти-бот защитой Yahoo, поэтому батч-загрузка не
            теряет тикеры (в отличие от обычной requests.Session) и при этом
            выполняется на 1-2 порядка быстрее поштучной загрузки.
        """
        start_map = {
            '5m':   date.today() - relativedelta(days=59),
            '60m':  date.today() - relativedelta(days=729),
            '1h':   date.today() - relativedelta(days=729),
        }

        end_map = {
            '5m':   date.today(),
            '60m':  date.today(),
            '1h':   date.today(),
        }

        start = start_map.get(interval, self.start_date)
        end = end_map.get(interval, self.end_date)

        tickers = list(self.yahoo_series.keys())
        frames, failed = [], []

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            raw_all = yf.download(
                tickers=tickers,
                start=start, end=end,
                interval=interval, ignore_tz=True,
                progress=False, threads=True,
                session=self._yahoo_session,
                group_by='ticker',
            )

        for ticker in tqdm(tickers, desc=f'Yahoo {interval}', unit='ticker', leave=False):
            name = self.yahoo_series[ticker]

            try:
                raw = raw_all[ticker]
            except (KeyError, IndexError):
                raw = pd.DataFrame()

            if raw.empty or raw.isna().all().all():
                failed.append(ticker)
                continue

            raw = raw.dropna(how='all')
            frames.append(raw.rename(columns=lambda c: f"{name}_{c}"))

        if failed:
            self.log(
                f"Не удалось загрузить {len(failed)}/{len(tickers)} тикеров "
                f"({interval}): {', '.join(failed)}"
            )

        data = pd.concat(frames, axis=1)

        return data.ffill()

    def add_technical_indicators(self, df: pd.DataFrame, close: str, high: str, low: str, volume: str) -> pd.DataFrame:
        """
            Добавляет следующие индикаторы в датафрейм:
            * `RSI` - Relative Strength Index, индикатор технического анализа, определяющий силу тренда и вероятность его смены.
            * `MACD` - Moving Average Convergence Divergence, индикатор используют для проверки силы и направления тренда, а также определения разворотных точек
            * `ATR` - Average True Range, показывает изменение волатильности, что немаловажно для определения силы тренда, а также «запаса хода» инструмента
            * `OBV` - On-Balance Volume, рассчитывает соотношение изменения цены и сопровождающих его объемов
        """
        df['RSI'] = _rsi(df[close])
        df['MACD'] = _macd(df[close])
        df['ATR'] = _atr(df[high], df[low], df[close])
        df['OBV'] = _obv(df[close], df[volume])

        return df

    def fetch_fear_greed_index(self) -> pd.DataFrame:
        """
            Загружает индекс страха и жадности.
        """
        url = "https://api.alternative.me/fng/?limit=0"
        try:
            resp = self._session.get(url, timeout=10).json()['data']
            df = pd.DataFrame(resp)
            df['Date'] = (
                pd.to_datetime(df['timestamp'].astype(int), unit='s', utc=True)
                .dt.tz_localize(None)
                .dt.normalize()
            )
            df.set_index('Date', inplace=True)
            df['value'] = df['value'].astype(int)
            return df[['value']].resample('D').ffill()
        except requests.RequestException as e:
            self.log(f"Ошибка при загрузке Fear & Greed Index: {e}")
            return pd.DataFrame(columns=['value'])
        except (KeyError, ValueError) as e:
            self.log(f"Ошибка при обработке Fear & Greed Index: {e}")
            return pd.DataFrame(columns=['value'])

    def fetch_btc_supply(self) -> pd.DataFrame:
        """
            Получает данные об общем объеме BTC в обращении
        """
        url = "https://api.blockchain.info/charts/total-bitcoins"
        params = {"timespan": "all", "format": "json", "sampling": "daily"}
        try:
            data = self._session.get(url, params=params, timeout=10).json()
            df = pd.DataFrame({
                'Date': pd.to_datetime([datetime.utcfromtimestamp(p['x']).date() for p in data['values']]),
                'BTC_Supply': [p['y'] for p in data['values']]
            })
            return df.set_index('Date').resample('D').ffill()
        except requests.RequestException as e:
            self.log(f"Ошибка при загрузке BTC Supply: {e}")
            return pd.DataFrame(columns=['BTC_Supply'])
        except (KeyError, ValueError) as e:
            self.log(f"Ошибка при обработке BTC Supply: {e}")
            return pd.DataFrame(columns=['BTC_Supply'])

    def _get_right_price(self, data: pd.DataFrame) -> pd.DataFrame:
        """
            приводит цены к следующему виду:

            цена = (цена закрытия * 2 + наивысшая цена + наименьшая цена) / 4
        """
        target_columns = ['RSI', 'MACD', 'ATR', 'OBV']
        columns = data.columns[~data.columns.isin(target_columns)]

        close_price = columns[columns.str.contains('_Close')]
        high_price  = columns[columns.str.contains('_High')]
        low_price   = columns[columns.str.contains('_Low')]

        price_columns = high_price.str.replace('_High', '', regex=False)

        cols =  pd.DataFrame(
            (2 * data[close_price].values
                + data[high_price].values
                + data[low_price].values) / 4,
            columns=price_columns,
            index=data.index
        )

        return pd.concat(
                    [cols, data.iloc[:, data.columns.isin(target_columns)]], axis=1
                )

    def _clean_dataframe(self, df: pd.DataFrame, name: str, nan_threshold: float = 15.0) -> pd.DataFrame:
        """
            Финальная очистка датасета:

            1. колонки, в которых после `.ffill()` доля пропусков превышает
               `nan_threshold` % (как правило тикеры, которые `get_yahoo_data`
               не смог загрузить и которые остались полностью пустыми), -
               удаляются;
            2. оставшиеся "хвостовые" пропуски в начале ряда (`.ffill()`
               не может заполнить значения до первого наблюдения, например
               для активов/индикаторов с более поздним стартом истории) -
               обрезаются по первой полностью валидной строке.
        """
        nan_pct = df.isna().mean() * 100
        drop_cols = nan_pct[nan_pct > nan_threshold].index.tolist()
        if drop_cols:
            self.log(
                f"[{name}] Удалены колонки с >{nan_threshold:.0f}% пропусков "
                f"({len(drop_cols)}): {', '.join(drop_cols)}"
            )
            df = df.drop(columns=drop_cols)

        valid = df.dropna()
        if valid.empty:
            raise ValueError(f"[{name}] После очистки не осталось ни одной валидной строки")

        first_valid = valid.index[0]
        if first_valid != df.index[0]:
            n_dropped = df.index.get_loc(first_valid)
            self.log(
                f"[{name}] Обрезаны первые {n_dropped} строк с пропусками "
                f"({df.index[0].date()} -> {first_valid.date()})"
            )
            df = df.loc[first_valid:]

        df = df.dropna()
        assert df.isna().sum().sum() == 0, f"[{name}] остались NaN после очистки"
        self.log(f"[{name}] Итоговый размер: {df.shape[0]} строк x {df.shape[1]} колонок")

        return df

    def build_minute_dataset(self) -> pd.DataFrame:
        """
            Создает 5-минутный датасет с техническими индикаторами и временными признаками.
        """
        self.log('Минутные данные начали грузиться\n')

        df = self.get_yahoo_data('5m')

        not_target_cols = df.columns[~df.columns.isin(['BTC_Close', 'BTC_High', 'BTC_Low', 'BTC_Volume'])]
        df = pd.concat([
            df[['BTC_Close', 'BTC_High', 'BTC_Low', 'BTC_Volume']],
            df[not_target_cols].shift(1)
        ], axis=1).iloc[1:]
        df = self.add_technical_indicators(df, 'BTC_Close', 'BTC_High', 'BTC_Low', 'BTC_Volume')
        df = self._get_right_price(df)

        df['minute'] =  df.index.minute.astype(int)
        df['hour'] =  df.index.hour.astype(int)

        lags_list = [1, 2, 5, 10]

        for lag in lags_list:
            df[f'BTC_lag{lag}'] = df['BTC'].shift(lag)

        df['BTC_mean5'] = df['BTC'].rolling(window=5).mean()
        df['BTC_std10'] = df['BTC'].rolling(window=10).std()

        df = df[~df['BTC'].isna()]

        self.log('Данные успешно загружены\n')

        return self._clean_dataframe(df.ffill(), 'Minute')

    def build_hourly_dataset(self) -> pd.DataFrame:
        """
            Создает почасовой датасет с техническими индикаторами и временными признаками.
        """
        self.log('Часовые данные начали грузиться\n')

        df = self.get_yahoo_data('60m')
        not_target_cols = df.columns[~df.columns.isin(['BTC_Close', 'BTC_High', 'BTC_Low', 'BTC_Volume'])]
        df = pd.concat([
            df[['BTC_Close', 'BTC_High', 'BTC_Low', 'BTC_Volume']],
            df[not_target_cols].shift(1)
        ], axis=1).iloc[1:]
        df = self.add_technical_indicators(df, 'BTC_Close', 'BTC_High', 'BTC_Low', 'BTC_Volume')
        df = self._get_right_price(df)

        df['day'] = df.index.day.astype(int)
        df['hour'] =  df.index.hour.astype(int)

        lags_list = [1, 2, 5, 10]

        for lag in lags_list:
            df[f'BTC_lag{lag}'] = df['BTC'].shift(lag)

        df['BTC_mean5'] = df['BTC'].rolling(window=5).mean()
        df['BTC_std10'] = df['BTC'].rolling(window=10).std()


        df = df[~df.MACD.isna()]

        self.log('Данные успешно загружены\n')

        return self._clean_dataframe(df.ffill(), 'Hourly')

    def build_daily_dataset(self) -> pd.DataFrame:
        """
            Создает дневной датасет с техническими индикаторами, макроэкономикой, индексом FNG и supply.
        """
        self.log('Дневные данные начали грузиться\n')

        df = self.get_yahoo_data('1d')

        not_target_cols = df.columns[~df.columns.isin(['BTC_Close', 'BTC_High', 'BTC_Low', 'BTC_Volume'])]
        df = pd.concat([
            df[['BTC_Close', 'BTC_High', 'BTC_Low', 'BTC_Volume']],
            df[not_target_cols].shift(1)
        ], axis=1).iloc[1:]
        df = self.add_technical_indicators(df, 'BTC_Close', 'BTC_High', 'BTC_Low', 'BTC_Volume')
        df = self._get_right_price(df)

        df = (
            df
            .merge(
                self.download_fred_data(),
                left_index=True, right_index=True,
                how='left')
            .merge(
                self.fetch_btc_supply(),
                left_index=True, right_index=True,
                how='left')
            .merge(
                self.fetch_fear_greed_index().rename(
                    columns={'value': 'Feer_Greed_value'}
                ),
                left_index=True, right_index=True,
                how='left')
            )

        df['day'] = df.index.day.astype(int)
        df['month'] = df.index.month.astype(int)
        df['year'] = df.index.year.astype(int)
        df['day_of_week'] = df.index.dayofweek.astype(int)
        df['is_holiday'] = [1 if date in holidays.US() else 0 for date in df.index]

        lags_list = [1, 2, 5, 10]

        for lag in lags_list:
            df[f'BTC_lag{lag}'] = df['BTC'].shift(lag)

        df['BTC_mean5'] = df['BTC'].rolling(window=5).mean()
        df['BTC_std10'] = df['BTC'].rolling(window=10).std()

        df = df[~df.Feer_Greed_value.isna()]

        self.log('Данные успешно загружены\n')

        return self._clean_dataframe(df.ffill(), 'Daily')

    def build_weekly_dataset(self) -> pd.DataFrame:
        """
            Создает недельный датасет с агрегацией макроэкономических показателей и технических индикаторов.
        """
        self.log('Недельные данные начали грузиться\n')

        df = self.get_yahoo_data('1wk')

        not_target_cols = df.columns[~df.columns.isin(['BTC_Close', 'BTC_High', 'BTC_Low', 'BTC_Volume'])]
        df = pd.concat([
            df[['BTC_Close', 'BTC_High', 'BTC_Low', 'BTC_Volume']],
            df[not_target_cols].shift(1)
        ], axis=1).iloc[1:]
        df = self.add_technical_indicators(df, 'BTC_Close', 'BTC_High', 'BTC_Low', 'BTC_Volume')
        df = self._get_right_price(df)

        fred_w = self.download_fred_data().resample('W-MON').ffill()
        fg_w = self.fetch_fear_greed_index().resample('W-MON').mean().rename(columns={'value': 'Feer_Greed_value'})
        supply_w = self.fetch_btc_supply().resample('W-MON').last()

        df = (
            df
            .merge(
                fred_w,
                left_index=True, right_index=True,
                how='left')
            .merge(
                fg_w,
                left_index=True, right_index=True,
                how='left')
            .merge(
                supply_w,
                left_index=True, right_index=True,
                how='left')
            )

        df['month'] = df.index.month.astype(int)
        df['quarter'] = df.index.quarter.astype(int)
        df['year'] = df.index.year.astype(int)

        df = df[~df.Feer_Greed_value.isna()]

        self.log('Данные успешно загружены\n')

        return self._clean_dataframe(df.ffill(), 'Weekly')

    def log(self, message: str) -> None:
        """
            Выводит сообщение в консоль с временной меткой.
        """

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n[94m[{timestamp}][0m {message}")

    def save_dataset(self, df: pd.DataFrame, name: str = "dataset", format: Literal['csv', 'parquet'] = "csv") -> None:
        """
            Сохраняет датасет в указанный формат.
        """
        path = f"{name}.{format}"

        if format == "csv":
            df.to_csv(path)
            self.log(f"Сохранено в {path}")

        elif format == "parquet":
            df.to_parquet(path)
            self.log(f"Сохранено в {path}")

        else:
            self.log('Некорректный формат данных. Доступны `parquet`, `csv')

    def add_stl_components(self, df: pd.DataFrame, target_col: str = 'BTC_Close', period: int = 20) -> pd.DataFrame:
        """
            Добавляет STL-компоненты (тренд, сезонность, остатки) к выбранной колонке.
        """

        stl = STL(df[target_col], period=period, robust=True)
        result = stl.fit()

        df['Trend'] = result.trend
        df['Seasonal'] = result.seasonal
        df['Residuals'] = result.resid

        return df
