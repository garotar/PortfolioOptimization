import numpy as np
import pandas as pd
from pypfopt import risk_models, BlackLittermanModel, EfficientFrontier, objective_functions, black_litterman
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Tuple, Optional


class BlackLittermanOptimizer:
    def __init__(
        self,
        historical_prices: pd.DataFrame,
        market_prices: pd.Series,
        market_caps: Dict[str, float] = None,
        test_start_date: Optional[str] = None,
        delta: Optional[float] = None
    ):
        """
        Оптимизация весов портфеля с помощью модели Блэка-Литтермана.

        Args:
            historical_prices: Исторические цены закрытия активов.
            market_prices: Исторические цены рыночного индекса (IMOEX).
            market_caps: Рыночные капитализации тикеров.
            test_start_date: Дата начала тестового периода.
            delta: Коэффициент неприятия риска.
        """
        self.historical_prices = historical_prices.pivot(index="date", columns="ticker", values="close").sort_index().ffill().bfill()
        self.market_prices = market_prices.pivot(index="date", columns="ticker", values="close").sort_index()

        if market_caps:
            self.market_caps = market_caps
        else:
            # данные на 08.05.2025
            self.market_caps = {
                "YDEX": 1561021463169,
                "LKOH": 4558710281079,
                "MGNT": 440766610375,
                "ROSN": 4718308764128,
                "GAZP": 3405908300923,
                "SBER": 6527461336240,
                "CHMF": 855310751860,
                "NVTK": 3551870758800
            }

        if test_start_date:
            self.historical_prices = self.historical_prices.loc[:test_start_date].iloc[:-1]
            self.market_prices = self.market_prices.loc[:test_start_date]

        self.delta = delta or self._calculate_delta()

    def _calculate_delta(self) -> float:
        """
        Расчет коэффициента непринятия риска.

        Returns:
            delta: Коэффициент непринятия риска.
        """
        delta = black_litterman.market_implied_risk_aversion(self.market_prices)
        return delta

    def calculate_market_prior(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Расчет рыночных равновесных доходностей и ковариационной матрицы доходностей.

        Returns:
            market_prior: Рыночные равновесные доходности (априорные доходности).
            S: Ковариационная матрица доходностей.
        """
        S = risk_models.CovarianceShrinkage(self.historical_prices).ledoit_wolf()
        market_prior = black_litterman.market_implied_prior_returns(self.market_caps, self.delta, S)
        return (market_prior, S)

    @staticmethod
    def prepare_predictions(
        df: pd.DataFrame,
        test_start_date: str,
        predictions: np.ndarray,
        targets: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Вспомогательная функция для подготовки датафреймов с предиктами и таргетами.

        Args:
            df: Исходный датафрейм.
            test_start_date: Дата начала тестовой части.
            predictions: Предикты DL моделей.
            targets: Таргеты.

        Returns:
            df: Подготовленный датафрейм.
        """
        close_prices = df.pivot(index="date", columns="ticker", values="close").sort_index().ffill().bfill()

        test_data = close_prices.loc[test_start_date:]
        preds_df = pd.DataFrame(predictions, index=test_data.index, columns=test_data.columns)

        if targets:
            targets_df = pd.DataFrame(targets, index=test_data.index, columns=test_data.columns)
            return (preds_df, targets_df)
        else:
            return preds_df

    def prepare_views_and_confidences(
        self,
        preds_df: pd.DataFrame,
        lower: np.ndarray,
        upper: np.ndarray,
        confidence_lower: float = 0.1,
        confidence_upper: float = 0.5,
        scaling_factor: float = 0.6
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Подготовка cубъективных прогнозов доходностей и уверенностей по методу Idzorek.

        Args:
            preds_df: Прогнозы доходностей.
            lower: Нижняя граница доверительного интервала для прогнозов.
            upper: Верхняя граница доверительного интервала для прогнозов.
            confidence_lower: Нижняя граница уверенности.
            confidence_upper: Верхняя граница уверенности.
            scaling_factor: Коэффициент для нормализации.

        Returns:
            views: Cубъективный прогноз доходностей.
            confidences: Уверенности.
        """
        # views - средний прогноз доходностей
        views = preds_df.mean(axis=0).to_dict()

        # confidences - уверенности (чем меньше интервал, тем выше уверенность)
        lb_df = pd.DataFrame(lower, index=preds_df.index, columns=preds_df.columns)
        ub_df = pd.DataFrame(upper, index=preds_df.index, columns=preds_df.columns)

        interval_width = (ub_df - lb_df).mean(axis=0)
        min_width = interval_width.min()
        max_width = interval_width.max()

        confidences = 1 - ((interval_width - min_width) / (max_width - min_width)) * scaling_factor
        confidences = confidences.clip(confidence_lower, confidence_upper).tolist()

        # print("Confidences по Idzorek для тикеров:")
        # for ticker, confidence in zip(preds_df.columns, confidences):
        #     print(f"{ticker}: {confidence:.2f}")

        # print("\nViews (средние прогнозы доходностей):")
        # for ticker, view in views.items():
        #     print(f"{ticker}: {view:.4%}")

        return (views, confidences)

    def optimize_weights(
        self,
        views: Dict[str, float],
        confidences: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Оптимизация весов с использованием модели Блэка-Литтермана.

        Args:
            views: Cубъективный прогноз доходностей.
            confidences: Уверенности.

        Returns:
            cleaned_weights: Оптимизированные веса портфеля.
            S: Ковариационная матрица доходностей.
            market_prior: Рыночные равновесные доходности (априорные доходности).
            bl_returns: Доходности, полученные с учетом субъективных прогнозов и уверенности (апостериорные доходности).
        """
        market_prior, S = self.calculate_market_prior()

        if confidences is not None:
            bl = BlackLittermanModel(cov_matrix=S, pi=market_prior, absolute_views=views, omega="idzorek", view_confidences=confidences)
        else:
            bl = BlackLittermanModel(cov_matrix=S, pi=market_prior, absolute_views=views, omega="default")

        bl_returns = bl.bl_returns()
        S_bl = bl.bl_cov()

        ef = EfficientFrontier(bl_returns, S_bl)
        ef.add_objective(objective_functions.L2_reg)
        ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        return (cleaned_weights, S, market_prior, bl_returns)

    def plot_market_prior(
        self,
        market_prior: pd.Series,
        mode: str = "plt",
        figsize: tuple = (10, 5)
    ):
        if mode == "plt":
            ax = market_prior.plot.barh(figsize=figsize, color="skyblue")
            ax.set_title("Предполагаемая рыночная доходность")
            ax.set_xlabel("Ожидаемая доходность")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        elif mode == "plotly":
            fig = go.Figure(go.Bar(
                x=market_prior.values,
                y=market_prior.index,
                orientation="h",
                marker_color="skyblue"
            ))
            fig.update_layout(
                title="Предполагаемая рыночная доходность",
                xaxis_title="Ожидаемая доходность",
                yaxis_title="Активы",
                template="plotly_white"
            )
            return fig
        else:
            raise ValueError("Доступный mode: 'plt' или 'plotly'.")

    def plot_prior_posterior_views(
        self,
        market_prior: pd.Series,
        posterior_returns: pd.Series,
        views: Dict[str, float],
        mode: str = "plt",
        figsize: tuple = (10, 5)
    ):
        rets_df = pd.DataFrame({
            "Prior": market_prior,
            "Posterior": posterior_returns,
            "Views": pd.Series(views)
        })

        if mode == "plt":
            ax = rets_df.plot.bar(figsize=figsize)
            ax.set_title("Сравнение априорных, апостериорных и прогнозных доходностей")
            ax.set_ylabel("Доходность")
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        elif mode == "plotly":
            fig = go.Figure()
            for col in rets_df.columns:
                fig.add_trace(go.Bar(
                    x=rets_df.index,
                    y=rets_df[col],
                    name=col
                ))
            fig.update_layout(
                title="Сравнение априорных, апостериорных и прогнозных доходностей",
                xaxis_title="Активы",
                yaxis_title="Доходность",
                barmode="group",
                template="plotly_white",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1,
                    xanchor="center",
                    x=0.5
                )
            )
            return fig
        else:
            raise ValueError("Доступный mode: 'plt' или 'plotly'.")

    def plot_optimal_weights(
        self,
        weights: Dict[str, float],
        mode: str = "plt",
        figsize: tuple = (10, 5)
    ):
        weights_series = pd.Series(weights)

        if mode == "plt":
            fig, ax = plt.subplots(figsize=figsize)
            ax.pie(
                weights_series,
                labels=weights_series.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=plt.cm.tab20.colors
            )
            ax.set_title("Оптимальные веса портфеля")
            plt.tight_layout()
            plt.show()

        elif mode == "plotly":
            fig = go.Figure(go.Pie(
                labels=weights_series.index,
                values=weights_series.values,
                hoverinfo="label+percent",
                textinfo="percent",
                textfont_size=14,
                marker=dict(colors=px.colors.qualitative.Pastel)
            ))
            fig.update_layout(
                title="Оптимальные веса портфеля",
                template="plotly_white",
            )
            return fig
        else:
            raise ValueError("Доступный mode: 'plt' или 'plotly'.")
