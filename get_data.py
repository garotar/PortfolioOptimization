import requests
import pandas as pd
import os
from typing import List

TOKEN = os.environ["INVEST_TOKEN"]


class GetCandles:
    def __init__(self, instrument_ids: List[str], date_from: str, date_to: str):
        self.instrument_ids = instrument_ids
        self.date_from = date_from
        self.date_to = date_to
        self.url = "https://invest-public-api.tinkoff.ru/rest/tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles"
        self.headers = {
            "Authorization": f"Bearer {TOKEN}",
            "accept": "application/json",
            "Content-Type": "application/json"
        }

    def fetch_candles(self, instrument_id: str) -> pd.DataFrame:
        params = {
            "from": self.date_from,
            "to": self.date_to,
            "interval": "CANDLE_INTERVAL_1_MIN",
            "instrumentId": instrument_id,
            "candleSourceType": "CANDLE_SOURCE_EXCHANGE",
            "limit": 1_000_000_000
        }

        response = requests.post(self.url, json=params, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            candles = data.get("candles", [])

            if candles:
                df = pd.json_normalize(candles)
                df["UID"] = instrument_id
                df["UTC"] = pd.to_datetime(df["time"], yearfirst=True, utc=True)

                df["open"] = df["open.units"].astype(int) + df["open.nano"] / 1_000_000_000
                df["low"] = df["low.units"].astype(int) + df["low.nano"] / 1_000_000_000
                df["high"] = df["high.units"].astype(int) + df["high.nano"] / 1_000_000_000
                df["close"] = df["close.units"].astype(int) + df["close.nano"] / 1_000_000_000
                df["volume"] = df["volume"].astype(int)

                return df[["UID", "UTC", "open", "close", "high", "low", "volume"]]
            else:
                print(f"Свечей для {instrument_id} за указанный период нет.")
                return pd.DataFrame()

        else:
            print(f"Ошибка для {instrument_id}: {response.status_code}")
            print(response.text)
            return pd.DataFrame()

    def run(self) -> pd.DataFrame:
        dfs = []

        for uid in self.instrument_ids:
            df = self.fetch_candles(uid)

            if not df.empty:
                dfs.append(df)

        if dfs:
            result_df = pd.concat(dfs).sort_values(by=["UID", "UTC"]).reset_index(drop=True)
            return result_df
        else:
            print("Нет доступных данных для объединения.")
            return pd.DataFrame()