import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Optional

from src.prepare_features import prepare_features


class AnomalyDetector:
    """
    Поиск аномалий с IsolationForest.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        model_params: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            df: Исходный датафрейм.
            model_params: Параметры для Isolation Forest.
        """
        self.df = df
        self.features = []

        default_params = {
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
        Обучает IsolationForest на сгенерированных признаках
                    и добавляет результаты в исходный датафрейм.

        Returns:
            Датафрейм с колонкой anomaly (1 - аномалия, 0 - нормальная точка).
        """
        result_df = pd.DataFrame()

        for ticker, group in self.df.groupby("ticker"):
            model = IsolationForest(**self.model_params)

            group["anomaly_score"] = model.fit_predict(group[self.features])
            group["anomaly"] = group["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

            result_df = pd.concat([result_df, group])
            result_df = result_df.drop(columns=["anomaly_score"], axis=1)
        self.df = result_df.reset_index(drop=True)
        return self.df

    def get_anomalies(self) -> pd.DataFrame:
        """
        Возвращает только те строки, которые модель определила как аномалии.
        """
        return self.df[self.df["anomaly"] == 1]

    def visualize_anomalies(
        self,
        ticker: str,
        is_interactive: bool = True,
        date_from: str = None,
        date_to: str = None
    ):
        """
        Строит график с аномалиями в двух режимах:
            - интерактивный с использованием plotly
            - статический с использованием seaborn
        """
        df_asset = self.df[self.df["ticker"] == ticker]

        if date_from:
            df_asset = df_asset[df_asset["date"] >= pd.to_datedate(date_from, date=True)]
        if date_to:
            df_asset = df_asset[df_asset["date"] <= pd.to_datedate(date_to, date=True)]

        anomalies = df_asset[df_asset["anomaly"] == 1]

        if is_interactive:
            fig = px.line(df_asset, x="date", y="return", title=f"Детекция аномалий для тикера {ticker}", labels={"date": "Дата", "return": "Доходность"})
            fig.add_scatter(x=anomalies["date"], y=anomalies["return"], mode="markers", marker=dict(color="red", size=8), name="Аномалии")
            fig.update_layout(xaxis_title="Дата", yaxis_title="Доходность")

            fig.show()

        else:
            plt.figure(figsize=(14, 6))
            sns.lineplot(data=df_asset, x="date", y="return", label="Доходность")
            sns.scatterplot(data=anomalies, x="date", y="return", color="red", label="Аномалии", s=50)

            plt.title(f"Детекция аномалий для тикера {ticker}", fontsize=16)
            plt.xlabel("Дата")
            plt.ylabel("Доходность")

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))

            plt.xticks(rotation=45)
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
