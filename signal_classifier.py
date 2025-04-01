import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from typing import List, Dict, Tuple, Any, Optional

import vectorbt as vbt

from prepare_data import prepare_features


# Минутные свечи и 252 торговых дня
vbt.settings.array_wrapper["freq"] = "minutes"
vbt.settings.returns["year_freq"] = f"{830*252} minutes"

class SignalClassifier:
    def __init__(self, df: pd.DataFrame, features: List[str], lookahead: int = 5, model_params: Optional[Dict[str, Any]] = None):
        """
        CatBoost классификатор bullish/bearish сигналов.
        """
        self.df = df.sort_values(["UID", "UTC"]).reset_index(drop=True)
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

    def _generate_features(self, features):
        self.df, self.features = prepare_features(self.df, features)
        self.df["UID"] = self.df["UID"].astype("category")

    def _prepare_target(self):
        self.df["future_return"] = (self.df.groupby("UID", observed=True)["close"].shift(-self.lookahead) / self.df["close"] - 1)
        self.df["target"] = np.where(self.df["future_return"] > 0, 1, 0)
        self.labeled_df = self.df[self.df["anomaly"] == 1].copy()

    def train(self, train_period_end: str, eval_period_end: str):
        """
        Обучение модели.

        Args:
            train_period_end: Конец train_df.
            eval_period_end: Конец eval_df.
        """
        train_df = self.labeled_df[self.labeled_df["UTC"] <= train_period_end].copy()
        eval_df = self.labeled_df[(self.labeled_df["UTC"] > train_period_end) & (self.labeled_df["UTC"] <= eval_period_end)].copy()
        test_df = self.labeled_df[self.labeled_df["UTC"] > eval_period_end].copy()

        X_train, y_train = train_df[["UID"] + self.features], train_df["target"]
        X_eval, y_eval = eval_df[["UID"] + self.features], eval_df["target"]
        X_test, y_test = test_df[["UID"] + self.features], test_df["target"]
        
        cat_features = ["UID"]

        self.model.fit(X_train, y_train, eval_set=(X_eval, y_eval), cat_features=cat_features)

        print("Модель успешно обучена!")

        self.X_test, self.y_test = X_test, y_test

    def evaluate(self) -> pd.DataFrame:
        full_df = self.df.copy()
        full_df["signal"] = 0

        test_indices = self.X_test.index
        preds = self.model.predict(self.X_test)

        full_df.loc[test_indices, "signal"] = preds

        return full_df[["UTC", "UID", "close", "signal"]].reset_index(drop=True)

    def prepare_backtest_data(self):
        df_with_signals = self.evaluate()

        close_prices = df_with_signals.pivot(index="UTC", columns="UID", values="close").sort_index().ffill().bfill()
        signals = df_with_signals.pivot(index="UTC", columns="UID", values="signal").reindex(close_prices.index).fillna(0)

        entries = signals.astype(bool)
        exits = (signals.shift(1).astype(bool)) & (~signals.astype(bool))

        return close_prices, entries, exits

    def run_backtest(self, init_cash: float = 100_000, freq: str = "min"):
        """
        Бэктест с использованием библиотеки vectorbt.

        Args:
            init_cash: начальный капитал.
            freq: гранулярность свечей, в моем случае 1 минута.

        Return:
            Объект Portfolio, содержащий статистику стратегии.
        """
        close_prices, entries, exits = self.prepare_backtest_data()
        self.portfolio_signal = vbt.Portfolio.from_signals(close_prices, entries,exits, init_cash=init_cash, freq=freq)
        
        return self.portfolio_signal
    
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