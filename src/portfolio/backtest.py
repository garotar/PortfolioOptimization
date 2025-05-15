import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objs as go
from typing import Optional, Union

vbt.settings.array_wrapper["freq"] = "days"
vbt.settings.returns["year_freq"] = "252 days"
vbt.settings.portfolio["seed"] = 42
vbt.settings.portfolio.stats["incl_unrealized"] = True


class Backtester:
    def __init__(
        self,
        df: pd.DataFrame,
        init_cash: int = 100_000,
        fees: float = 0.01,
        test_start_date: str = None
    ):
        """
        Бэктест стратегий.

        Args:
            df: Исходный датафрейм.
            init_cash: Начальный капитал.
            fees: Комиссия брокеру.
            test_start_date: Дата начала тестового периода.
        """
        prices = df.pivot(index="date", columns="ticker", values="close").sort_index().ffill().bfill()

        if test_start_date:
            self.close_prices = prices.loc[test_start_date:]
        else:
            self.close_prices = prices

        self.returns = self.close_prices.pct_change().fillna(0)
        self.init_cash = init_cash
        self.fees = fees
        self.tickers = self.close_prices.columns
        self.num_tickers = len(self.tickers)
        self.test_start_date = test_start_date

    def check_stat_measures(
        self,
        mode: str = "returns"
    ):
        """
        Вывод статистических характеристик: среднее, стандартное отклонение, матрица корреляции.

        Args:
            mode: Расчет характеристик по ценам закрытия "close" или по доходностям "returns".

        Returns:
            stat_measures: Словарь с характеристиками.
        """
        if mode == "returns":
            mode_type = self.returns
        elif mode == "close":
            mode_type = self.close_prices
        else:
            raise ValueError("Доступный mode: 'returns' или 'close'.")

        mean = mode_type.mean()
        std = mode_type.std()
        corr = mode_type.corr()

        stat_measures = {
            "mean": mean,
            "std": std,
            "corr": corr
        }

        return stat_measures

    def run_buy_and_hold(self):
        """
        Стратегия "Купи и держи" с равновесным портфелем без ребалансировки.

        Returns:
            portfolio: Объект vbt.Portfolio.
        """
        equal_weights = [np.full(self.num_tickers, 1 / self.num_tickers)]
        num_tests = 1
        multind_price = self.close_prices.vbt.tile(num_tests, keys=pd.Index(np.arange(num_tests), name="symbol_group"))
        multind_price = multind_price.vbt.stack_index(pd.Index(np.concatenate(equal_weights), name="weights"))

        equal_size = np.full_like(multind_price, np.nan)
        equal_size[0, :] = np.concatenate(equal_weights)

        portfolio = vbt.Portfolio.from_orders(
            close=multind_price,
            size=equal_size,
            size_type="targetpercent",
            group_by="symbol_group",
            cash_sharing=True,
            fees=self.fees,
            init_cash=self.init_cash,
            freq="1D",
            min_size=1,
            size_granularity=1
            )
        return portfolio

    def run_buy_and_hold_rebalanced(
        self,
        rebalance_freq: str = "Q",
        weights: Optional[np.ndarray] = None
    ):
        """
        Стратегия "Купи и держи" с ребалансировкой и указанием конкретных весов активов.

        Args:
            rebalance_freq: Частота ребалансировки.
            weights: Веса активов.

        Returns:
            rebalanced_portfolio: Объект vbt.Portfolio.
        """
        if weights is None:
            weights = [np.full(self.num_tickers, 1 / self.num_tickers)]
        else:
            weights = [np.array([weights[t] for t in self.tickers])]
        num_tests = 1
        multind_price = self.close_prices.vbt.tile(num_tests, keys=pd.Index(np.arange(num_tests), name="symbol_group"))
        multind_price = multind_price.vbt.stack_index(pd.Index(np.concatenate(weights), name="weights"))

        rebalance_mask = ~multind_price.index.to_period(rebalance_freq).duplicated()

        rebalance_size = np.full_like(multind_price, np.nan)
        rebalance_size[rebalance_mask, :] = np.concatenate(weights)

        rebalanced_portfolio = vbt.Portfolio.from_orders(
            close=multind_price,
            size=rebalance_size,
            size_type="targetpercent",
            group_by="symbol_group",
            cash_sharing=True,
            call_seq="auto",
            fees=self.fees,
            init_cash=self.init_cash,
            freq="1D",
            min_size=1,
            size_granularity=1
        )

        return rebalanced_portfolio

    def run_signal_backtest(
        self,
        signals_df: pd.DataFrame,
        size: Union[int, float] = 10_000,
        size_type: str = "value"
    ):
        """
        Сигнальная стратегия.

        Args:
            signals_df: Датафрейм с сигналами -1, 0, 1.
            size: Размер позиции, зависящий от size_type.
            size_type: Тип выделения капитала на сделку.

        Returns:
            signal_portfolio: Объект vbt.Portfolio.
        """
        signals_df = signals_df[signals_df["date"] >= self.test_start_date]
        pivot_signals = signals_df.pivot(index="date", columns="ticker", values="signal").sort_index().fillna(0)

        entries = pivot_signals == 1
        exits = pivot_signals == -1

        signal_portfolio = vbt.Portfolio.from_signals(
            close=self.close_prices,
            entries=entries,
            exits=exits,
            size=size,
            size_type=size_type,
            cash_sharing=True,
            fees=self.fees,
            init_cash=self.init_cash,
            min_size=1,
            size_granularity=1,
            freq="1D",
            direction="longonly"
        )
        return signal_portfolio

    def run_ema_crossover(
        self,
        short_window: int = 10,
        long_window: int = 20,
        size: Union[int, float] = 10_000,
        size_type: str = "value",
    ):
        """
        Стратегия пересечения двух экспоненциальных скользящих средних.

        Args:
            short_window: Период короткой экспоненциальной скользящей средней.
            long_window: Период длинной экспоненциальной скользящей средней.
            size: Размер позиции, зависящий от size_type.
            size_type: Тип выделения капитала на сделку.

        Returns:
            ema_portfolio: Объект vbt.Portfolio.
        """
        short_ema = vbt.MA.run(self.close_prices, short_window, ewm=True)
        long_ema = vbt.MA.run(self.close_prices, long_window, ewm=True)

        entries = short_ema.ma_crossed_above(long_ema)
        exits = short_ema.ma_crossed_below(long_ema)

        ema_portfolio = vbt.Portfolio.from_signals(
            close=self.close_prices,
            entries=entries,
            exits=exits,
            size=size,
            size_type=size_type,
            cash_sharing=True,
            fees=self.fees,
            init_cash=self.init_cash,
            min_size=1,
            size_granularity=1,
            freq="1D",
            direction="longonly"
        )

        return ema_portfolio

    def portfolio_performance(
        self,
        portfolio: vbt.Portfolio
    ):
        """
        Расчет основных финансовых метрик портфеля.

        Args:
            portfolio: Объект vbt.Portfolio.
        """
        sharpe_ratio = portfolio.sharpe_ratio()
        sortino_ratio = portfolio.sortino_ratio()
        drawdowns = portfolio.drawdowns.max_drawdown()
        performance = portfolio.total_return()

        print(f"Sharpe ratio: {sharpe_ratio:.2f}")
        print(f"Sortino ratio: {sortino_ratio:.2f}")
        print(f"Max drawdown: {drawdowns * 100:.2f}%")
        print(f"Total return: {performance * 100:.2f}%")

    def individual_stock_returns(
        self,
        portfolio: vbt.Portfolio
    ):
        """
        Расчет дохода/убытка по каждому активу.

        Args:
            portfolio: Объект vbt.Portfolio.
        """
        performance = portfolio.total_return(group_by=False) * 100
        print("Доходности каждого актива, %")
        print(performance)

    def trade_details(
        self,
        portfolio: vbt.Portfolio
    ):
        """
        Детали по сделкам.

        Args:
            portfolio: Объект vbt.Portfolio.

        Returns:
            trades: Датафрейм с информацией по сделкам.
        """
        trades = portfolio.trades.records_readable
        return trades

    def plot_equity_and_drawdown(
        self,
        portfolio: vbt.Portfolio,
        mode: str = "plt"
    ):
        equity = portfolio.value()
        drawdown = portfolio.drawdown() * 100

        if mode == "plt":
            fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

            axes[0].plot(equity.index, equity, label="Кривая капитал", color="blue")
            axes[0].set_title("Кривая капитала")
            axes[0].set_ylabel("Капитал")
            axes[0].grid(alpha=0.3)

            axes[1].fill_between(drawdown.index, drawdown, color="red", alpha=0.3)
            axes[1].set_title("Кривая просадки")
            axes[1].set_ylabel("Просадка, %")
            axes[1].set_xlabel("Дата")
            axes[1].grid(alpha=0.3)

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        elif mode == "plotly":
            equity_trace = go.Scatter(
                x=equity.index, y=equity, mode="lines", name="Кривая капитала"
            )
            equity_layout = go.Layout(
                title="Кривая капитала", xaxis_title="Дата", yaxis_title="Капитал"
            )
            equity_fig = go.Figure(data=[equity_trace], layout=equity_layout)

            drawdown_trace = go.Scatter(
                x=drawdown.index, y=drawdown, mode="lines",
                name="Кривая просадки", fill="tozeroy", line=dict(color="red")
            )
            drawdown_layout = go.Layout(
                title="Кривая просадки", xaxis_title="Дата",
                yaxis_title="Просадка, %", template="plotly_white"
            )
            drawdown_fig = go.Figure(data=[drawdown_trace], layout=drawdown_layout)
            return (equity_fig, drawdown_fig)
        else:
            raise ValueError("Доступный mode: 'plt' или 'plotly'.")

    def plot_cash_and_value(
        self,
        portfolio: vbt.Portfolio,
        mode: str = "plt"
    ):
        cash_data = portfolio.cash()
        value_data = portfolio.value()
        init_cash = self.init_cash

        if mode == "plt":
            fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

            axes[0].plot(cash_data.index, cash_data, color="green", label="Наличные")
            axes[0].fill_between(
                cash_data.index, cash_data, init_cash,
                color="red", alpha=0.3, interpolate=True, label="Капитал в обороте"
            )
            axes[0].set_title("Наличные за период")
            axes[0].set_ylabel("Наличные")
            axes[0].axhline(init_cash, color="gray", linestyle="--", linewidth=1)
            axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
            axes[0].grid(alpha=0.3)

            axes[1].plot(value_data.index, value_data, color="blue", label="Кривая капитала")
            axes[1].axhline(init_cash, color="gray", linestyle="--", linewidth=1, label="Начальный капитал")
            axes[1].fill_between(
                value_data.index, value_data, init_cash,
                where=(value_data > init_cash), interpolate=True, color="green", alpha=0.3, label="Прибыль"
            )
            axes[1].set_title("Капитал за период")
            axes[1].set_ylabel("Капитал")
            axes[1].set_xlabel("Дата")
            axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
            axes[1].grid(alpha=0.3)

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        elif mode == "plotly":
            cash_fig = go.Figure()

            cash_fig.add_trace(go.Scatter(
                x=cash_data.index, y=cash_data, mode="lines", name="Наличные", line=dict(color="green")
            ))

            cash_fig.add_trace(go.Scatter(
                x=cash_data.index, y=[init_cash]*len(cash_data), mode="lines",
                line=dict(color="gray", dash="dash"), name="Начальный капитал"
            ))

            cash_fig.add_trace(go.Scatter(
                x=cash_data.index, y=cash_data, fill="tonexty",
                fillcolor="rgba(255,0,0,0.3)", mode="none", showlegend=True, name="Капитал в обороте"
            ))

            cash_fig.update_layout(
                title="Наличные за период",
                xaxis_title="Дата",
                yaxis_title="Наличные",
                template="plotly_white"
            )

            value_fig = go.Figure()

            value_fig.add_trace(go.Scatter(
                x=value_data.index,
                y=value_data,
                mode="lines",
                name="Капитал",
                line=dict(color="blue")
            ))

            value_fig.add_trace(go.Scatter(
                x=value_data.index,
                y=[init_cash]*len(value_data),
                mode="lines",
                line=dict(color="gray", dash="dash"),
                name="Начальный капитал"
            ))

            profit_mask = value_data > init_cash
            if profit_mask.any():
                value_fig.add_trace(go.Scatter(
                    x=value_data.index,
                    y=np.where(profit_mask, value_data, init_cash),
                    mode="none",
                    fill="tonexty",
                    fillcolor="rgba(26,150,65,0.5)",
                    name="Прибыль"
                ))

            value_fig.update_layout(
                title="Капитал за период",
                xaxis_title="Дата",
                yaxis_title="Капитал, руб.",
                template="plotly_white"
            )
            return cash_fig, value_fig
        else:
            raise ValueError("Доступный mode: 'plt' или 'plotly'.")

    def plot_weights(
        self,
        portfolio: vbt.Portfolio,
        mode: str = "plt"
    ):
        asset_value = portfolio.asset_value(group_by=False)
        total_value = portfolio.value()

        weights = asset_value.divide(total_value, axis=0)

        if mode == "plt":
            weights.plot.area(figsize=(14, 6), cmap="tab20")
            plt.title("Веса активов за период")
            plt.ylabel("Веса")
            plt.xlabel("Дата")
            plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

            plt.xticks(rotation=45)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        elif mode == "plotly":
            fig = weights.vbt.plot(
                trace_names=self.tickers,
                trace_kwargs=dict(stackgroup="one")
            )
            fig.update_layout(
                title="Веса активов за период",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1,
                    xanchor="center",
                    x=0.5
                ),
                width=1450,
                height=500,
                margin=dict(l=60, r=60, t=100, b=60),
                template="plotly_white"
            )
            return fig

        else:
            raise ValueError("Доступный mode: 'plt' или 'plotly'.")
