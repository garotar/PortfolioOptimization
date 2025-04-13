import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict


def prepare_dataloaders(
    returns: pd.DataFrame,
    rolling_covs: pd.DataFrame,
    seq_len: int = 60,
    forecast_horizon: int = 7,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Dict[str, DataLoader]:
    """
    Подготавливает DataLoader'ы для обучения LSTM.

    Args:
        returns: Pivot датафрейм (индекс-дата, колонки-тикеры) с доходностями.
        rolling_covs: Pivot датафрейм с ковариационными матрицами доходностей.
                        Используется мульти-индекс: [дата, тикер].

        seq_len: Длина входной последовательности.
        forecast_horizon: Горизонт прогнозирования.
        batch_size: Размер батча для DataLoader.
        train_ratio: Доля данных для обучения.
        val_ratio: Доля данных для валидации.

    Returns:
        Словарь train/val/test DataLoader'ами.
    """
    sequences_X_ret = []
    sequences_X_cov = []
    sequences_y = []

    returns_values = returns.values
    dates = returns.index

    total_samples = len(dates) - seq_len - forecast_horizon + 1

    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    indices = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total_samples)
    }

    for i in range(total_samples):
        X_ret = returns_values[i:i+seq_len]

        cov_matrices = []
        for j in range(i, i+seq_len):
            date = dates[j]
            cov_matrix = rolling_covs.loc[date].values
            cov_matrices.append(cov_matrix)

        X_cov = np.stack(cov_matrices)

        y = returns_values[i+seq_len:i+seq_len+forecast_horizon].flatten()

        sequences_X_ret.append(X_ret)
        sequences_X_cov.append(X_cov)
        sequences_y.append(y)

    X_ret_tensor = torch.tensor(np.array(sequences_X_ret), dtype=torch.float32)
    X_cov_tensor = torch.tensor(np.array(sequences_X_cov), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(sequences_y), dtype=torch.float32)

    data = {
        split: TensorDataset(
            X_ret_tensor[start:end],
            X_cov_tensor[start:end],
            y_tensor[start:end]
        )
        for split, (start, end) in indices.items()
    }

    dataloaders = {
        split: DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for split, dataset in data.items()
    }

    return dataloaders
