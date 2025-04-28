import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Tuple, Dict

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
        """
        Создает Pivot датафрейм на основе исходного датафрейма.

        Returns:
            Pivot датафрейм.
        """
        df_pivot = self.df.pivot(
            index="date",
            columns="ticker",
            values=self.close_price)
        df_pivot = df_pivot.sort_index().ffill().bfill()
        return df_pivot

    def get_test_period_dates(
        self,
        test_split_ratio: float = 0.1
    ) -> Tuple[str, str]:
        """
        Находит первую и последнюю дату тестовой выборки.

        Args:
            test_split_ratio: Доля тестовой выборки.

        Returns:
            Кортеж с первой и последней датой.
        """
        close_prices = self.prepare_backtest_data()

        n = len(close_prices)
        test_len = int(n * test_split_ratio)

        test_start = close_prices.index[-test_len]
        test_end = close_prices.index[-1]

        start = test_start.strftime("%Y-%m-%d")
        end = test_end.strftime("%Y-%m-%d")

        return (start, end)

    def buy_and_hold_performance(
        self,
        test_split_ratio: float = 0.1,
        init_cash: float = 100_000
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Ручная реализация стратегии "купить и держать".

        Args:
            test_split_ratio: Доля тестовой выборки.
            init_cash: Начальный капитал.

        Returns:
            total_portfolio_df: Датафрейм со стоимостями лотов по каждому тикеру.
            total_portfolio_value: Общая стоимость портфеля.
            portfolio_return: Доходность портфеля.
        """
        test_period_dates = self.get_test_period_dates(test_split_ratio)
        close_prices = self.prepare_backtest_data()

        test_close_prices = close_prices.loc[test_period_dates[0]: test_period_dates[1]]

        # равновесный портфель
        tickers = test_close_prices.columns
        N = len(tickers)
        weights = 1 / N

        first_prices = test_close_prices.iloc[0]
        cash_per_ticker = init_cash * weights

        # количество бумаг
        lots = (cash_per_ticker // first_prices).astype(int)
        invested_per_ticker = lots * first_prices
        total_invested = invested_per_ticker.sum()

        # остаток денег
        cash_resid = init_cash - total_invested

        total_portfolio_df = test_close_prices * lots
        total_portfolio_value = total_portfolio_df.sum(axis=1)
        if cash_resid != 0:
            total_portfolio_value += cash_resid

        portfolio_return = total_portfolio_value / total_portfolio_value.iloc[0] - 1
        self._lots = lots
        self.test_close_prices = test_close_prices

        return (total_portfolio_df, total_portfolio_value, portfolio_return)

    def run_backtest(
        self,
        test_split_ratio: float = 0.1,
        init_cash: float = 100_000,
        freq: str = "days"
    ) -> Tuple[vbt.Portfolio, Dict[str, float]]:
        """
        Бэктест с помощью vectorbt.

        Args:
            test_split_ratio: Доля тестовой выборки.
            init_cash: Начальный капитал.
            freq: Гранулярность свечей. В моем случае - "день".

        Returns:
            Объект Portfolio и основные метрики.
        """
        test_period_dates = self.get_test_period_dates(test_split_ratio)
        close_prices = self.prepare_backtest_data().loc[test_period_dates[0]: test_period_dates[1]]

        size_df = pd.DataFrame(0, index=close_prices.index, columns=close_prices.columns)
        size_df.iloc[0] = self._lots

        self.baseline_portfolio = vbt.Portfolio.from_orders(
            close=close_prices,
            size=size_df,
            size_type="amount",
            direction="longonly",
            init_cash=init_cash,
            cash_sharing=True,
            group_by=True,
            freq=freq
        )

        metrics = self.baseline_portfolio.stats(metrics=["sharpe_ratio", "sortino_ratio", "max_dd", "total_return"])

        return (self.baseline_portfolio, dict(metrics))
