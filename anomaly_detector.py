import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
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

        self.model_params = default_params

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
        result_df = pd.DataFrame()

        for uid, group in self.df.groupby("UID"):
            model = IsolationForest(**self.model_params)

            group["anomaly_score"] = model.fit_predict(group[self.features])
            group["anomaly"] = group["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

            result_df = pd.concat([result_df, group])
        
        self.df = result_df.reset_index(drop=True)
        return self.df

    def get_anomalies(self) -> pd.DataFrame:
        """
        Возвращает только те строки, которые модель определила как аномалии.
        """
        return self.df[self.df["anomaly"] == 1]

    def visualize_anomalies(self, instrument_id: str, is_interactive: bool = True, date_from: str = None, date_to: str = None):
        """
        Строит график с аномалиями.
        """
        df_asset = self.df[self.df["UID"] == instrument_id]

        if date_from:
            df_asset = df_asset[df_asset["UTC"] >= pd.to_datetime(date_from, utc=True)]
        if date_to:
            df_asset = df_asset[df_asset["UTC"] <= pd.to_datetime(date_to, utc=True)]

        anomalies = df_asset[df_asset["anomaly"] == 1]

        if is_interactive:
            fig = px.line(df_asset, x="UTC", y="return", title=f"Anomaly Detection for {instrument_id}", labels={"UTC": "Month-Year", "return": "Return"})
            fig.add_scatter(x=anomalies["UTC"], y=anomalies["return"], mode="markers", marker=dict(color="red", size=8), name="Anomalies")
            fig.update_layout(xaxis_title="Month-Year", yaxis_title="Return", xaxis=dict(tickformat="%m-%Y"))

            fig.show()

        else:
            plt.figure(figsize=(14, 6))
            sns.lineplot(data=df_asset, x="UTC", y="return", label="Return")
            sns.scatterplot(data=anomalies, x="UTC", y="return", color="red", label="Anomalies", s=50)

            plt.title(f"Anomaly Detection for {instrument_id}", fontsize=16)
            plt.xlabel("Month-Year")
            plt.ylabel("Return")

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

            plt.xticks(rotation=45)
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()