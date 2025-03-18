import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from typing import List, Dict, Any, Optional

from prepare_data import prepare_features


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
        self.df["future_return"] = (self.df.groupby("UID")["close"].shift(-self.lookahead) / self.df["close"] - 1)
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

    def evaluate(self):
        """
        Оценка качества модели.
        """
        preds = self.model.predict(self.X_test)
        report = classification_report(self.y_test, preds, output_dict=True)
        return report

    def feature_importance(self) -> pd.DataFrame:
        """
        Важность признаков.
        """
        importance = self.model.get_feature_importance(prettified=True)
        return importance