import pandas as pd
import numpy as np
from scipy.optimize import minimize


class MeanVarianceOptimizer:
    def __init__(self, rebalance: str = "m"):
        """
        rebalance: частота ребалансировки портфеля.
                   "d" - ежедневно, "w" - еженедельно, "m" - ежемесячно.
        """
        self.rebalance = rebalance
        if rebalance not in {"d", "w", "m"}:
            raise ValueError("Доступная ребалансировка: d - день, w - неделя, m - месяц.")

    def _get_rebalance_dates(
        self,
        dates: pd.DatetimeIndex
    ) -> pd.DatetimeIndex:

        if self.rebalance == "d":
            return dates
        elif self.rebalance == "w":
            week_groups = dates.to_series().groupby(dates.to_period("W")).last()
            return pd.DatetimeIndex(week_groups.values)
        elif self.rebalance == "m":
            month_groups = dates.to_series().groupby(dates.to_period("M")).last()
            return pd.DatetimeIndex(month_groups.values)

    def _max_sharpe_weights(
        self,
        mu: np.ndarray,
        sigma: np.ndarray
    ) -> np.ndarray:

        N = len(mu)

        def neg_sharpe(w):
            ret = w @ mu
            vol = (w @ sigma @ w)**0.5
            return -ret/vol if vol != 0 else 0

        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
        bounds = [(0, 1)] * N

        # начальное приближение - равновесный портфель
        init_w = np.full(N, 1/N)
        result = minimize(neg_sharpe,
                          init_w,
                          method="SLSQP",
                          bounds=bounds,
                          constraints=constraints)

        # если оптимизация успешна, то берем result.x
        if result.success:
            w_opt = result.x
        else:
            w_opt = result.x if result.x is not None else init_w

        w_opt = np.clip(w_opt, 0, 1)
        w_opt = w_opt / np.sum(w_opt)
        return w_opt

    def optimize(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        close_prices = df.pivot(index="date", columns="ticker", values="close").sort_index()
        returns = close_prices.pct_change().fillna(0)
        dates = returns.index
        rebalance_dates = self._get_rebalance_dates(dates)

        tickers = list(close_prices.columns)
        weights_df = pd.DataFrame(index=rebalance_dates, columns=tickers, dtype=float)

        first = True

        # проходим по каждой дате ребалансировки и оптимизируем веса
        for date in rebalance_dates:
            if first:

                # в начале равные веса
                w = np.full(len(tickers), 1.0/len(tickers))
                first = False
            else:

                # используем данные доходностей ДО текущей даты включительно
                past_returns = returns.loc[:date]
                mu = past_returns.mean().values
                sigma = past_returns.cov().values
                w = self._max_sharpe_weights(mu, sigma)
            weights_df.loc[date] = w

        weights_df.index = pd.DatetimeIndex(weights_df.index)
        return weights_df
