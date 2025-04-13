import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import List, Tuple, Dict

from sklearn.metrics import mean_absolute_error, mean_squared_error

vbt.settings.array_wrapper["freq"] = "days"
vbt.settings.returns["year_freq"] = "252 days"


class BaselineBacktest:
    """
    Бэктест бейзлайн стратегии "купить и держать".
    """
    def __init__(
        self,
        df: pd.DataFrame,
        close_price: str = "close"
    ):
        """
        Args:
            df: Датафрейм со свечами, содержащий колонки "ticker", "date" и "close".
            close_price: Колонка с ценой закрытия.
        """
        self.df = df.copy()
        self.close_price = close_price

    def prepare_backtest_data(self) -> pd.DataFrame:
        df_pivot = self.df.pivot(
            index="date",
            columns="ticker",
            values=self.close_price)
        df_pivot = df_pivot.sort_index()
        df_pivot = df_pivot.ffill().bfill()
        return df_pivot

    def run_backtest(
        self,
        init_cash: float = 100_000,
        freq: str = "day"
    ) -> vbt.Portfolio:
        """
        Бэктест с использованием библиотеки vectorbt.

        Args:
            init_cash: начальный капитал.
            freq: гранулярность свечей, в моем случае 1 день.

        Return:
            Объект Portfolio, содержащий статистику стратегии.
        """
        close_prices = self.prepare_backtest_data()
        self.baseline_portfolio = vbt.Portfolio.from_holding(close_prices, init_cash=init_cash, freq=freq)

        return self.baseline_portfolio

    def get_metrics(
        self,
        tickers: List[str]
    ) -> dict:
        """
        Рассчитывает финансовые метрики: Sharpe Ratio, Sortino Ratio и Max Drawdown.

        Args:
            tickers: Список тикеров (уникальные ticker из датафрейма).

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

    @staticmethod
    def seasonal_naive_forecast(
        returns: pd.DataFrame,
        train_ratio: float = 0.9,
        season_length: int = 7
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        """
        Наивный прогноз. Взял Seasonal Naive Method.

        Args:
            returns: Pivot датафрейм доходностей.
            train_ratio: Доля данных для обучения.
            season_length: Число точек, используемых для прогноза.

        Returns:
            forecast: Датафрейм с прогнозами.
            test: Датафрейм с таргетами.
            combined: Объединённый датафрейм с таргетами и прогнозами.
            metrics: Словарь с метриками.
        """
        n = len(returns)
        n_train = int(n * train_ratio)
        train = returns.iloc[:n_train]
        test = returns.iloc[n_train:]

        seasonal_values = train.iloc[-season_length:]

        reps = int(np.ceil(len(test) / season_length))
        forecast = pd.concat([seasonal_values] * reps, axis=0).iloc[:len(test)]

        forecast.index = test.index

        mae = mean_absolute_error(test.values, forecast.values)
        rmse = np.sqrt(mean_squared_error(test.values, forecast.values))
        metrics = {"MAE": mae, "RMSE": rmse}

        return (forecast, metrics, train, test)
