import pandas as pd
import vectorbt as vbt
from typing import List


# Минутные свечи и 252 торговых дня
vbt.settings.array_wrapper["freq"] = "minutes"
vbt.settings.returns["year_freq"] = "252 days"

class BaselineBacktest:
    def __init__(self, df: pd.DataFrame, close_price: str = "close"):
        """
        Бэктест бейзлайн стратегии "купить и держать".

        Args:
            df: Датафрейм со свечами, содержащий колонки "UID", "UTC" и "close".
            close_price: Колонка с ценой закрытия.
        """
        self.df = df.copy()
        self.close_price = close_price

    def prepare_backtest_data(self) -> pd.DataFrame:
        df_pivot = self.df.pivot(index="UTC", columns="UID", values=self.close_price)
        df_pivot = df_pivot.sort_index()
        df_pivot = df_pivot.ffill().bfill()
        return df_pivot

    def run_backtest(self, init_cash: float = 100_000, freq: str = "min") -> vbt.Portfolio:
        """
        Бэктест с использованием библиотеки vectorbt.

        Args:
            init_cash: начальный капитал.
            freq: гранулярность свечей, в моем случае 1 минута.

        Return:
            Объект Portfolio, содержащий статистику стратегии.
        """
        close_prices = self.prepare_backtest_data()
        self.baseline_portfolio = vbt.Portfolio.from_holding(close_prices, init_cash=init_cash, freq=freq)

        return self.baseline_portfolio

    def get_metrics(self, tickers: List[str]) -> dict:
        """
        Рассчитывает финансовые метрики: Sharpe Ratio, Sortino Ratio и Max Drawdown.

        Args:
            tickers: Список тикеров (уникальные UID из датафрейма).

        Return:
            Словарь с метриками.
        """  
        metrics = {}

        for ticker in tickers:
            stats_ticker = self.baseline_portfolio.stats(column=ticker)
            metrics[ticker] = {
                "Sharpe Ratio": stats_ticker.get("Sharpe Ratio", None),
                "Sortino Ratio": stats_ticker.get("Sortino Ratio", None),
                "Max Drawdown": stats_ticker.get("Max Drawdown [%]", None)
            }
        return metrics