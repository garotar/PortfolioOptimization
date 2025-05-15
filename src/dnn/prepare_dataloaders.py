import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


class FinTSDataset(Dataset):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        corr_matrix: np.array,
        window: int
    ):
        self.X = X.values
        self.y = y.values
        self.corr_matrix = corr_matrix
        self.window = window
        self.dates = y.index

    def __len__(self):
        return len(self.X) - self.window

    def __getitem__(self, idx: int):
        X_window = torch.tensor(self.X[idx:idx+self.window],
                                dtype=torch.float32)
        corr_window = torch.tensor(self.corr_matrix[idx:idx+self.window],
                                   dtype=torch.float32)
        y_window = torch.tensor(self.y[idx+self.window],
                                dtype=torch.float32)
        return (X_window, corr_window, y_window)


class FinTSDataloaders:
    def __init__(
        self,
        df: pd.DataFrame,
        window: int,
        forecast_horizon: int,
        batch_size: int
    ):
        self.df = df
        self.window = window
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size

        self.close_prices = self.df.pivot(index="date", columns="ticker", values="close").sort_index().ffill().bfill()
        self.volumes = self.df.pivot(index="date", columns="ticker", values="volume").sort_index().ffill().bfill()

    def add_features(self):
        features_dict = {}
        for ticker in self.close_prices.columns:
            ticker_features = pd.DataFrame(index=self.close_prices.index)

            # доходности за разные периоды
            for period in [1, 5, 10, 20]:
                ticker_features[f"return_{period}d"] = self.close_prices[ticker].pct_change(periods=period)

            # скользящие средние доходности
            for ma_window in [5, 10, 20]:
                ma = self.close_prices[ticker].rolling(window=ma_window).mean()
                ticker_features[f"ma_return_{ma_window}d"] = self.close_prices[ticker] / ma - 1

            # волатильность
            for vol_window in [5, 10, 20]:
                vol = self.close_prices[ticker].pct_change().rolling(window=vol_window).std()
                ticker_features[f"volatility_{vol_window}d"] = vol

            # объем торгов
            ticker_features["volume"] = self.volumes[ticker]
            ticker_features["volume_ma5"] = self.volumes[ticker].rolling(window=5).mean()
            ticker_features["volume_ma20"] = self.volumes[ticker].rolling(window=20).mean()
            ticker_features["volume_ratio"] = ticker_features["volume_ma5"] / ticker_features["volume_ma20"]

            # таргет
            ticker_features[f"target_return_{self.forecast_horizon}d"] = (
                self.close_prices[ticker].pct_change(periods=-self.forecast_horizon).shift(self.forecast_horizon)
            )

            features_dict[ticker] = ticker_features.ffill().fillna(0)

        combined_features = pd.concat(features_dict, axis=1)

        features_df = combined_features.drop(columns=f"target_return_{self.forecast_horizon}d", level=1)
        target_df = combined_features.xs(f"target_return_{self.forecast_horizon}d", level=1, axis=1)

        return (features_df, target_df)

    def create_corr_matrices(self):
        returns_df = self.close_prices.pct_change().fillna(0)
        corr_matrices = []

        for i in range(len(returns_df)):
            if i < self.window:
                window_returns = returns_df.iloc[:i+1]
            else:
                window_returns = returns_df.iloc[i-self.window+1:i+1]
            corr_matrix = window_returns.corr().fillna(0).values
            corr_matrices.append(corr_matrix)

        corr_matrices = np.array(corr_matrices)

        return corr_matrices

    def split_and_scale_data(
        self,
        train_split_ratio: float = 0.7,
        val_split_ratio: float = 0.2
    ):
        features_df, target_df = self.add_features()
        corr_matrices = self.create_corr_matrices()

        train_size = int(len(features_df) * train_split_ratio)
        val_size = int(len(features_df) * val_split_ratio)

        X_train = features_df.iloc[:train_size]
        X_val = features_df.iloc[train_size + self.window:train_size + val_size + self.window]
        X_test = features_df.iloc[train_size + val_size + self.window:]
        # X_val = features_df.iloc[train_size:train_size+val_size]
        # X_test = features_df.iloc[train_size+val_size:]

        y_train = target_df.iloc[:train_size]
        y_val = target_df.iloc[train_size + self.window:train_size + val_size + self.window]
        y_test = target_df.iloc[train_size + val_size + self.window:]
        # y_val = target_df.iloc[train_size:train_size+val_size]
        # y_test = target_df.iloc[train_size+val_size:]

        corr_train = corr_matrices[:train_size]
        corr_val = corr_matrices[train_size + self.window:train_size + val_size + self.window]
        corr_test = corr_matrices[train_size + val_size + self.window:]
        # corr_val = corr_matrices[train_size:train_size+val_size]
        # corr_test = corr_matrices[train_size+val_size:]

        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        X_train_scaled = pd.DataFrame(
            feature_scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            feature_scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            feature_scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        y_train_scaled = pd.DataFrame(
            target_scaler.fit_transform(y_train),
            columns=y_train.columns,
            index=y_train.index
        )
        y_val_scaled = pd.DataFrame(
            target_scaler.transform(y_val),
            columns=y_val.columns,
            index=y_val.index
        )
        y_test_scaled = pd.DataFrame(
            target_scaler.transform(y_test),
            columns=y_test.columns,
            index=y_test.index
        )

        return (
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            corr_train, corr_val, corr_test,
            feature_scaler, target_scaler
        )

    def get_loaders(
        self,
        train_split_ratio: float = 0.7,
        val_split_ratio: float = 0.2
    ):
        (
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            corr_train, corr_val, corr_test,
            feature_scaler, target_scaler
        ) = self.split_and_scale_data(train_split_ratio, val_split_ratio)

        train_dataset = FinTSDataset(X_train_scaled,
                                     y_train_scaled,
                                     corr_train,
                                     self.window)
        val_dataset = FinTSDataset(X_val_scaled,
                                   y_val_scaled,
                                   corr_val,
                                   self.window)
        test_dataset = FinTSDataset(X_test_scaled,
                                    y_test_scaled,
                                    corr_test,
                                    self.window)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False)

        return (
            train_loader, val_loader, test_loader,
            feature_scaler, target_scaler
            )
