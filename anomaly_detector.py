import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Optional

from prepare_data import prepare_features


class AnomalyDetector:
    def __init__(self, df: pd.DataFrame, model_params: Optional[Dict[str, Any]] = None):
        """
        Поиск аномалий с IsolationForest.
        """
        self.df = df
        self.features = []

        default_params: Dict[str, Any] = {
            "n_estimators": 100,
            "max_samples": "auto",
            "contamination": 0.01,
            "max_features": 1.0,
            "bootstrap": False,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": 0,
            "warm_start": False
        }

        if model_params:
            default_params.update(model_params)

        self.model = IsolationForest(**default_params)

    def generate_features(self):
        """
        Генерирует необходимые признаки, используя функцию prepare_features.
        """
        self.df, self.features = prepare_features(self.df)
        return self.df, self.features

    def detect_anomalies(self) -> pd.DataFrame:
        """
        Обучает IsolationForest на сгенерированных признаках и добавляет результаты в исходный датафрейм.

        Return:
            Датафрейм с колонками anomaly_score и anomaly (1 - аномалия, 0 - нормальная точка).
        """
        self.df["anomaly_score"] = self.model.fit_predict(self.df[self.features])
        self.df["anomaly"] = self.df["anomaly_score"].map(lambda x: 1 if x == -1 else 0)
        return self.df

    def get_anomalies(self) -> pd.DataFrame:
        """
        Возвращает только те строки, которые модель определила как аномалии.
        """
        return self.df[self.df["anomaly"] == 1]

    def visualize_anomalies(self, instrument_id: str):
        """
        Строит интерактивный график с аномалиями.
        """
        df_asset = self.df[self.df["UID"] == instrument_id]
        fig = px.line(df_asset, x="UTC", y="return", title=f"Anomaly Detection for {instrument_id}", labels={"UTC": "Month-Year", "return": "Return"})
        anomalies = df_asset[df_asset["anomaly"] == 1]
        fig.add_scatter(x=anomalies["UTC"], y=anomalies["return"], mode="markers", marker=dict(color="red", size=8), name="Anomalies")
        fig.update_layout(xaxis_title="Month-Year", yaxis_title="Return", xaxis=dict(tickformat="%m-%Y"))

        fig.show()