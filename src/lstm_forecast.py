import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Dict, Optional, Any


class CNNEmbedding(nn.Module):
    """
    Свертка ковариационных матриц, позволяет преобразовать матрицу в вектор длины cov_embed_dim.
    """
    def __init__(
        self,
        num_tickers: int,
        cov_embed_dim: int = 8
    ):
        """
        Args:
            num_tickers: Количество тикеров (после применения
                                        TICKER_MAPPING, если он использовался).
            cov_embed_dim: Желаемая длина выходного вектора.
        """
        super().__init__()

        self.num_tickers = num_tickers
        self.cov_embed_dim = cov_embed_dim
        self.cnn = nn.Sequential(

            # входной канал - одна ковариационная матрица
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(8*2*2, self.cov_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        batch_size, seq_len, self.num_tickers, _ = x.shape
        x = x.view(batch_size*seq_len, 1, self.num_tickers, self.num_tickers)
        emb = self.cnn(x)
        emb = emb.view(batch_size, seq_len, -1)

        return emb


class LSTMForecaster(nn.Module):
    """
    Модель для прогнозирования будущих доходностей с учетом ковариационных матриц.
    """
    def __init__(
        self,
        num_tickers: int,
        cov_embed_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 1,
        forecast_horizon: int = 7,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            num_tickers: Количество тикеров (после применения
                                        TICKER_MAPPING, если он использовался).
            cov_embed_dim: Желаемая длина выходного вектора.
            hidden_dim: Размер скрытого слоя LSTM.
            num_layers: Количество слоев LSTM.
            forecast_horizon: Горизонт прогнозирования.
            device: GPU или CPU.
        """
        super().__init__()

        self.num_tickers = num_tickers
        self.cov_embed_dim = cov_embed_dim
        self.forecast_horizon = forecast_horizon
        self.output_dim = num_tickers * forecast_horizon
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cov_embedding = CNNEmbedding(num_tickers, cov_embed_dim)

        self.lstm = nn.LSTM(
            input_size=num_tickers + cov_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.output_dim)
        )

    def forward(
        self,
        X_ret: torch.Tensor,
        X_cov: torch.Tensor
    ) -> torch.Tensor:

        # (batch_size, seq_len, cov_embed_dim)
        X_cov_emb = self.cov_embedding(X_cov)

        # (batch_size, seq_len, num_tickers + cov_embed_dim)
        x = torch.cat([X_ret, X_cov_emb], dim=-1)

        # (batch_size, seq_len, lstm_hidden_dim)
        lstm_out, _ = self.lstm(x)

        # последний временной шаг (batch_size, lstm_hidden_dim)
        last_output = lstm_out[:, -1, :]

        # прогноз (batch_size, num_tickers * forecast_horizon)
        y_pred = self.fc(last_output)

        return y_pred

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        patience: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Запускает обучение LSTM.

        Args:
            train_loader: DataLoader для обучения.
            val_loader: DataLoader для валидации.
            num_epochs: Число эпох обучения.
            optimizer: Оптимизатор, по умолчанию Adam с learning_rate = 0.001.
            criterion: Функция потерь, по умолчанию MSELoss.
            patience: Количество эпох без улучшения val loss для ранней остановки.
            verbose: Информация об эпохе - train и val loss + ранняя остановка.

        Returns:
            Словарь с train и val loss, лучшим val loss,
                номером последней эпохи (на случай ранней остановки).
        """
        self.to(self.device)

        if criterion is None:
            criterion = nn.MSELoss()
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=0.001)

        best_val_loss = np.inf
        epochs_no_improve = 0
        best_state = None

        losses = {"train_loss": [], "val_loss": []}

        for epoch in range(1, num_epochs + 1):

            self.train()

            train_loss = 0
            for X_ret, X_cov, y in train_loader:
                X_ret = X_ret.to(self.device)
                X_cov = X_cov.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                y_pred = self(X_ret, X_cov)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            self.eval()

            val_loss = 0
            with torch.no_grad():
                for X_ret, X_cov, y in val_loader:
                    X_ret = X_ret.to(self.device)
                    X_cov = X_cov.to(self.device)
                    y = y.to(self.device)
                    y_pred = self(X_ret, X_cov)
                    loss = criterion(y_pred, y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            losses["train_loss"].append(train_loss)
            losses["val_loss"].append(val_loss)

            if verbose:
                print(f"Эпоха {epoch}: [train_loss] = {train_loss:.6f} | [val_loss] = {val_loss:.6f}")

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = self.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print("Сработала ранняя остановка.")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        model_info = {
            "losses": losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch
            }

        return model_info

    def predict_dataset(
        self,
        test_dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготавливает 2 массива: предикты и таргеты.

        Args:
            test_dataloader: Тестовый DataLoader.

        Returns:
            Массивы с предиктами и таргетами.
        """
        self.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for X_ret, X_cov, y in test_dataloader:
                X_ret = X_ret.to(self.device)
                X_cov = X_cov.to(self.device)
                y = y.to(self.device)
                y_pred = self(X_ret, X_cov)
                preds.append(y_pred.cpu().numpy())
                targets.append(y.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

        return preds, targets

    def evaluate_model(
        self,
        test_dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Вычисляет метрики.

        Args:
            test_dataloader: Тестовый DataLoader.

        Returns:
            Словарь с метриками.
        """
        preds, targets = self.predict_dataset(test_dataloader)
        mae = mean_absolute_error(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))

        metrics = {
            "MAE": mae,
            "RMSE": rmse
            }

        return metrics

    @staticmethod
    def create_forecast_dfs(
        pivot_df: pd.DataFrame,
        forecast_array: np.ndarray,
        last_n: int = 20
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Создает 3 датафрейма:
            a. Датафрейм с последними N доходностями по каждому тикеру;
            b. Датафрейм с прогнозами;
            c. Объединенный датафрейм из пунктов a и b.

        Args:
            pivot_df: Pivot датафрейм с доходностями.
            forecast_array: Массив прогнозов
                        размерности (горизонт прогнозирования, число тикеров).
            last_n: Количество последних доходностей.

        Returns:
            df_last: Последние last_n строк pivot_df.
            df_forecast: Датафрейм с прогнозами.
            df_combined: Объединенный датафрейм с историей и прогнозами.
        """
        df_last = pivot_df.tail(last_n).copy()

        if forecast_array.ndim == 2 and forecast_array.shape[0] > 1:
            forecast_array = forecast_array[-1]

        num_tickers = pivot_df.shape[1]
        total_length = forecast_array.shape[0]
        if total_length % num_tickers != 0:
            raise ValueError(f"Длина forecast_array ({total_length}) не кратна числу тикеров ({num_tickers}).")

        forecast_horizon = total_length // num_tickers
        forecast_array = forecast_array.reshape(forecast_horizon, num_tickers)

        last_date = pivot_df.index.max()

        # бизнес дни для реалистичности
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                      periods=forecast_horizon)

        df_forecast = pd.DataFrame(data=forecast_array, index=future_dates,
                                   columns=pivot_df.columns)

        df_combined = pd.concat([df_last, df_forecast])

        return (df_last, df_forecast, df_combined)
