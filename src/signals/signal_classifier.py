import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from typing import List, Dict, Any, Optional

from src.signals.prepare_features import prepare_features


class SignalClassifier:
    """
    Классификатор bullish/bearish/neutral сигналов.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        lookahead: int = 7,
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
            "iterations": 1000,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 2.0,
            "bagging_temperature": 1.0,
            "verbose": 100,
            "early_stopping_rounds": 100,
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1"
        }
        if model_params:
            default_params.update(model_params)

        self.model_params = default_params
        self._generate_features()
        self._prepare_target()

    def _generate_features(self):
        self.df, self.features = prepare_features(self.df, self.features)
        self.df["ticker"] = self.df["ticker"].astype("category")

    def _prepare_target(
        self,
        threshold: float = 1,
        window: int = 20
    ):
        self.df["future_return"] = self.df.groupby("ticker")["close"].pct_change(self.lookahead).shift(-self.lookahead)
        self.df["rolling_sigma"] = self.df.groupby("ticker")["return"].transform(lambda x: x.rolling(window).std())

        up_thr = threshold * self.df["rolling_sigma"]
        down_thr = -threshold * self.df["rolling_sigma"]

        self.df["target"] = 0
        self.df.loc[self.df["future_return"] > up_thr, "target"] = 1
        self.df.loc[self.df["future_return"] < down_thr, "target"] = -1

        self.df.ffill(inplace=True)
        self.labeled_df = self.df.copy()

    def check_class_balance(self):
        print("Баланс классов по тикерам:")
        for ticker, group in self.labeled_df.groupby("ticker"):
            print(f"{ticker}:\n{group['target'].value_counts()}\n")

    def train(
        self,
        test_split_ratio: float = 0.1,
        n_folds: int = 5,
        gap: int = 5
    ):
        close_price = self.df.pivot(index="date", columns="ticker", values="close").sort_index()
        dates = close_price.index
        cutoff_date = dates[int(len(dates) * (1 - test_split_ratio))]

        print(f"Дата разделения train/test: {cutoff_date.date()}")

        train_df = self.labeled_df[self.labeled_df["date"] <= cutoff_date]
        test_df = self.df[self.df["date"] > cutoff_date]

        X_train = train_df[["ticker"] + self.features]
        y_train = train_df["target"]
        cat_features = [0]

        dates = train_df["date"].unique()
        fold_sizes = np.linspace(0.6, 0.9, n_folds, endpoint=False)
        date_cutoffs = dates[(fold_sizes * len(dates)).astype(int)]

        best_iterations = []
        for i, cut_date in enumerate(date_cutoffs):
            train_mask = train_df["date"] <= cut_date
            val_mask = (train_df["date"] > cut_date + pd.Timedelta(days=gap))
            if i < len(date_cutoffs) - 1:
                val_mask &= (train_df["date"] <= date_cutoffs[i + 1])

            X_tr = X_train[train_mask]
            y_tr = y_train[train_mask]
            X_val = X_train[val_mask]
            y_val = y_train[val_mask]

            fold_model = CatBoostClassifier(**self.model_params, random_seed=42+i)
            fold_model.fit(
                X_tr,
                y_tr,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                use_best_model=True,
                verbose=False
            )
            best_iterations.append(fold_model.get_best_iteration())

        optimal_iters = int(np.mean(best_iterations))
        print(f"Оптимальное количество итераций: {optimal_iters}")

        self.model_params["iterations"] = optimal_iters
        self.model = CatBoostClassifier(**self.model_params)
        self.model.fit(X_train, y_train, cat_features=cat_features)

        self.test_df = test_df

    def evaluate(self, bull_threshold=0.7, bear_threshold=0.7):
        X_test = self.test_df[["ticker"] + self.features]
        proba = self.model.predict_proba(X_test)
        idx_bear = np.where(self.model.classes_ == -1)[0][0]
        idx_bull = np.where(self.model.classes_ == 1)[0][0]

        bear_p = proba[:, idx_bear]
        bull_p = proba[:, idx_bull]

        preds = np.where(
            bull_p >= bull_threshold, 1,
            np.where(bear_p >= bear_threshold, -1, 0)
        )

        result = self.test_df[["date", "ticker", "close"]].copy()
        result["signal"] = preds
        result = result.sort_values(["date", "ticker"]).reset_index(drop=True)

        return (proba, result)

    def feature_importance(self) -> pd.DataFrame:
        """
        Важность признаков.
        """
        importance = self.model.get_feature_importance(prettified=True)
        return importance
