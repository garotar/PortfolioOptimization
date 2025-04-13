import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from typing import List, Dict, Any, Optional

import vectorbt as vbt

from src.prepare_features import prepare_features

vbt.settings.array_wrapper["freq"] = "days"
vbt.settings.returns["year_freq"] = "252 days"


class SignalClassifier:
    """
    Классификатор bullish/bearish сигналов.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        lookahead: int = 3,
        model_params: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            df: Датафрейм с аномалиями, полученный с помощью AnomalyDetector.
            features: Список признаков для генерации.
            lookahead: Число свеч, на которое будем смотреть вперед.
            model_params: Параметры для классификатора Catboost.
        """
        self.df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        self.features = features
        self.lookahead = lookahead

        default_params = {
            "iterations": 500,
            "depth": 7,
            "learning_rate": 0.05,
            "random_seed": 42,
            "verbose": 100,
            "early_stopping_rounds": 50,
        }

        if model_params:
            default_params.update(model_params)

        self.model = CatBoostClassifier(**default_params)

        self._generate_features(self.features)
        self._prepare_target()

    def _generate_features(
        self,
        features: List[str]
    ):
        """
        Генерирует необходимые признаки, используя функцию prepare_features.

        Args:
            features: Список признаков для генерации.
        """
        self.df, self.features = prepare_features(self.df, features)
        self.df["ticker"] = self.df["ticker"].astype("category")

    def _prepare_target(self):
        """
        Размечает таргет для датафрейма с аномалиями. Оставляет только "аномальные" свечи.
        """
        self.df["future_return"] = (
            self.df.groupby("ticker", observed=True)["close"]
            .shift(-self.lookahead) / self.df["close"] - 1)
        self.df["target"] = np.where(self.df["future_return"] > 0, 1, 0)
        self.labeled_df = self.df[self.df["anomaly"] == 1].copy()

    # TODO: переделать разбивку на train/val/test с использованием ratio
    def train(
        self,
        train_period_end: str,
        eval_period_end: str
    ):
        """
        Обучение модели.

        Args:
            train_period_end: Конец train_df.
            eval_period_end: Конец eval_df.
        """
        train_df = self.labeled_df[self.labeled_df["date"] <= train_period_end].copy()
        eval_df = self.labeled_df[(self.labeled_df["date"] > train_period_end) & (self.labeled_df["date"] <= eval_period_end)].copy()
        test_df = self.labeled_df[self.labeled_df["date"] > eval_period_end].copy()

        X_train, y_train = train_df[["ticker"] + self.features], train_df["target"]
        X_eval, y_eval = eval_df[["ticker"] + self.features], eval_df["target"]
        X_test, y_test = test_df[["ticker"] + self.features], test_df["target"]

        cat_features = ["ticker"]

        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_eval, y_eval),
            cat_features=cat_features
        )

        print("Модель успешно обучена!")

        self.X_test, self.y_test = X_test, y_test

    def evaluate(self) -> pd.DataFrame:
        full_df = self.df.copy()
        full_df["signal"] = 0

        test_indices = self.X_test.index
        preds = self.model.predict(self.X_test)

        full_df.loc[test_indices, "signal"] = preds

        return full_df[["date", "ticker", "close", "signal"]].reset_index(drop=True)

    def prepare_backtest_data(self):
        df_with_signals = self.evaluate()

        close_prices = df_with_signals.pivot(index="date", columns="ticker", values="close").sort_index().ffill().bfill()
        signals = df_with_signals.pivot(index="date", columns="ticker", values="signal").reindex(close_prices.index).fillna(0)

        entries = signals.astype(bool)
        exits = (signals.shift(1).astype(bool)) & (~signals.astype(bool))

        return close_prices, entries, exits

    def run_backtest(
        self,
        init_cash: float = 100_000,
        freq: str = "day"
    ):
        """
        Бэктест с использованием библиотеки vectorbt.

        Args:
            init_cash: начальный капитал.
            freq: гранулярность свечей, в моем случае 1 день.

        Returns:
            Объект Portfolio, содержащий статистику стратегии.
        """
        close_prices, entries, exits = self.prepare_backtest_data()
        self.portfolio_signal = vbt.Portfolio.from_signals(close_prices, entries, exits, init_cash=init_cash, freq=freq)

        return self.portfolio_signal

    def get_metrics(
        self,
        tickers: List[str]
    ) -> dict:
        """
        Рассчитывает финансовые метрики: Sharpe Ratio, Sortino Ratio и Max Drawdown.

        Args:
            tickers: Список тикеров (уникальные ticker из датафрейма).

        Returns:
            Словарь с метриками.
        """
        metrics = {}

        for ticker in tickers:
            stats_ticker = self.portfolio_signal.stats(column=ticker)
            metrics[ticker] = {
                "Sharpe Ratio": stats_ticker.get("Sharpe Ratio", None),
                "Sortino Ratio": stats_ticker.get("Sortino Ratio", None),
                "Max Drawdown": stats_ticker.get("Max Drawdown [%]", None)
            }
        return metrics

    def feature_importance(self) -> pd.DataFrame:
        """
        Важность признаков.
        """
        importance = self.model.get_feature_importance(prettified=True)
        return importance
