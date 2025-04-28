import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Optional, Tuple

from src.prepare_features import prepare_features


class AnomalyDetector:
    """
    Поиск аномалий с IsolationForest.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        model_params: Optional[Dict[str, Any]] = None,
        contamination: float = 0.01
    ):
        """
        Args:
            df: Исходный датафрейм.
            model_params: Параметры для Isolation Forest.
            contamination: Доля аномалий.
        """
        self.df = df
        self.features = []

        default_params = {
            "n_estimators": 100,
            "max_samples": "auto",
            "contamination": contamination,
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

        Returns:
            Датафрейм и список признаков.
        """
        self.df, self.features = prepare_features(self.df)
        return self.df, self.features

    def detect_anomalies(
        self,
        train_period_end: str = "2024-12-31"
    ) -> pd.DataFrame:
        """
        Обучает IsolationForest на сгенерированных признаках
                    и добавляет результаты в исходный датафрейм.

        Returns:
            Датафрейм с колонками "anomaly" и "anomaly_score".
        """
        result_list = []
        train_period_end = pd.to_datetime(train_period_end)

        for ticker, group in self.df.groupby("ticker"):
            group = group.sort_values("date").reset_index(drop=True)

            if train_period_end < group["date"].min():
                raise ValueError("train_end раньше первой даты.")
            if train_period_end > group["date"].max():
                raise ValueError("train_end позже последней даты.")

            train_mask = group["date"] <= train_period_end
            X_train = group.loc[train_mask, self.features]

            scaler = StandardScaler()
            scaler.fit(X_train)

            model = IsolationForest(**self.model_params)
            model.fit(scaler.transform(X_train))

            scores = model.decision_function(
                scaler.transform(group[self.features])
            )
            preds = model.predict(
                scaler.transform(group[self.features])
            )

            group["anomaly_score"] = -scores
            group["anomaly"] = (preds == -1).astype(int)

            result_list.append(group)

        self.df = pd.concat(result_list, ignore_index=True)
        return self.df

    def get_anomalies(self) -> pd.DataFrame:
        """
        Возвращает только строки с аномалиями.
        """
        return self.df[self.df["anomaly"] == 1]

    def visualize_anomalies(
        self,
        ticker: str,
        is_interactive: bool = True,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Строит график с аномалиями в двух режимах:
            - интерактивный с использованием plotly
            - статический с использованием matplotlib
        Args:
            ticker: Тикер актива.
            is_interactive: Использовать интерактивный график plotly.
            date_from: Начало периода для графика (YYYY-MM-DD).
            date_to: Конец периода для графика (YYYY-MM-DD).
            figsize: Размер статического графика (ширина, высота).
        """
        df_asset = self.df[self.df["ticker"] == ticker]

        if date_from:
            df_asset = df_asset[df_asset["date"] >= pd.to_datetime(date_from)]
        if date_to:
            df_asset = df_asset[df_asset["date"] <= pd.to_datetime(date_to)]

        anomalies = df_asset[df_asset["anomaly"] == 1]

        if is_interactive:
            fig = px.line(
                df_asset,
                x="date",
                y="return",
                title=f"Детекция аномалий для тикера {ticker}",
                labels={"date": "Дата", "return": "Доходность"}
            )
            fig.add_scatter(
                x=anomalies["date"], y=anomalies["return"],
                mode="markers", marker=dict(color="red", size=8),
                name="Аномалии"
            )
            fig.update_layout(xaxis_title="Дата", yaxis_title="Доходность")

            fig.show()

        else:
            plt.figure(figsize=figsize)
            plt.plot(df_asset["date"], df_asset["return"], color="blue", linewidth=1, label="Доходность")
            plt.plot(anomalies["date"], anomalies["return"], "ro", markersize=3, label="Аномалии")

            plt.title(f"Детекция аномалий для тикера {ticker}", fontsize=16)
            plt.xlabel("Дата")
            plt.ylabel("Доходность")

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))

            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
