import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import List, Optional


class MOEXCandlesFetcher:
    """
    Получение свеч определнных тикеров с Московской биржи.
    """
    BASE_URL = "https://iss.moex.com/iss/history/engines/stock/markets/shares/securities"

    TICKER_MAPPING = {
        "YNDX": "YDEX",
        "FIVE": "X5"
    }

    def __init__(
        self,
        candle_interval: int = 24,
    ):
        """
        Args:
            candle_interval: Интервал свеч, 24 - свечи с дневной гранулярностью.
        """
        self.candle_interval = candle_interval

    def fetch_candles(
        self,
        tickers: List[str],
        from_date: str,
        till_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Основной метод для получения свеч по всем тикерам.

        Args:
            tickers: Список тикеров, можно взять с Мосбиржи.
            from_date: Дата начала периода.
            till_date: Дата окончания периода,
                        если не указана, то будет выбран вчерашний день.

        Returns:
            Датафрейм со свечами.
        """
        if till_date is None:
            till_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

        all_data = []
        for ticker in tickers:
            df = self._fetch_single_ticker(ticker, from_date, till_date)
            if not df.empty:
                all_data.append(df)
            else:
                print(f"[Предупреждение]: Тикер '{ticker}' не найден в период с {from_date} по {till_date}.")

        if not all_data:
            return pd.DataFrame(columns=["ticker", "date", "open", "low", "high", "close", "volume"])

        result_df = pd.concat(all_data, ignore_index=True)
        result_df = self._normalize_tickers(result_df)
        return result_df

    def _fetch_single_ticker(
        self,
        ticker: str,
        from_date: str,
        till_date: str,
        board_id: str = "TQBR"
    ) -> pd.DataFrame:
        """
        Метод для получения свечей по одному тикеру.

        Args:
            ticker: Тикер с Мосбиржи.
            from_date: Дата начала периода.
            till_date: Дата окончания периода,
                        если не указана, то будет выбран вчерашний день.
            board_id: Идентификатор площадки/режима торгов.
                        TQBR – основная площадка для акций и облигаций.

        Returns:
            Датафрейм со свечами.
        """
        url = f"{self.BASE_URL}/{ticker}/candles.json"
        params = {
            "from": from_date,
            "till": till_date,
            "interval": self.candle_interval,
            "start": 0,
            "limit": 100
        }

        all_rows = []
        while True:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data_json = response.json()

            rows = data_json["history"]["data"]
            columns = data_json["history"]["columns"]

            if not rows:
                break

            df_page = pd.DataFrame(rows, columns=columns)
            df_page = df_page[df_page["BOARDID"] == board_id]

            if df_page.empty:
                break

            all_rows.append(df_page)

            cursor_data = data_json["history.cursor"]["data"][0]
            current_index, total, page_size = cursor_data
            params["start"] += page_size

            if current_index + page_size >= total:
                break

        if not all_rows:
            return pd.DataFrame()

        result_df = pd.concat(all_rows, ignore_index=True)
        result_df = result_df[["SECID", "TRADEDATE", "OPEN", "LOW", "HIGH", "CLOSE", "VOLUME"]].copy()
        result_df.rename(columns={
            "SECID": "ticker",
            "TRADEDATE": "date",
            "OPEN": "open",
            "LOW": "low",
            "HIGH": "high",
            "CLOSE": "close",
            "VOLUME": "volume"
        }, inplace=True)

        return result_df

    def run(
        self,
        data_path: str = "./data/moex_data.csv",
        tickers: Optional[List[str]] = None,
        from_date: str = "2024-01-01",
        till_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Запускает получение свечей и сохраняет данные в .csv файл.

        Args:
            data_path: Папка где лежит .csv файл с данными,
                        если файла нет, то сохранит его в эту папку.
            tickers: Список тикеров.
            from_date: Дата начала периода.
            till_date: Дата окончания периода,
                        если не указана, то будет выбран вчерашний день.

        Returns:
            Датафрейм со свечами.
        """
        if tickers is None or len(tickers) == 0:
            raise ValueError("Список тикеров не должен быть пустым.")

        try:
            df = pd.read_csv(data_path)
            df = df.drop_duplicates(subset=["ticker", "date"], keep="first")
            df["date"] = pd.to_datetime(df["date"], yearfirst=True)
            df = df.sort_values(by=["ticker", "date"], ignore_index=True)
            print(f"Загружены ранее скачанные данные {data_path}.")
        except FileNotFoundError:
            print("Данные отсутствуют. Загружаем данные с начала периода.")

            dfs = []
            pbar = tqdm(tickers, desc="Первичная загрузка данных")
            for ticker in pbar:
                pbar.set_description(f"Загрузка тикера {ticker}")
                df_ticker = self.fetch_candles([ticker], from_date=from_date, till_date=till_date)
                dfs.append(df_ticker)

            df = pd.concat(dfs, ignore_index=True)
            df = df.drop_duplicates(subset=["ticker", "date"], keep="first")
            df["date"] = pd.to_datetime(df["date"], yearfirst=True)
            df = df.sort_values(by=["ticker", "date"], ignore_index=True)
            df.to_csv(data_path, index=False)
        return df

    def _normalize_tickers(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Переименовывает тикеры. Например, когда у компании поменялся тикер,
                                но нужна полная история.

        Args:
            df: Датафрейм со свечами.

        Returns:
            Датафрейм с переименованными тикерами.
        """
        df["ticker"] = df["ticker"].replace(self.TICKER_MAPPING)
        return df
