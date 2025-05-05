import pandas as pd
import numpy as np
import vectorbt as vbt
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, Dict


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler=None,
        device="cpu"
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(
        self,
        train_loader,
        val_loader,
        epochs=100,
        patience=20
    ):
        if self.scheduler is None:
            self.scheduler = ReduceLROnPlateau(self.optimizer,
                                               patience=10,
                                               factor=0.5)

        best_val_loss = np.inf
        counter = 0

        train_losses = []
        val_losses = []

        for epoch in range(epochs):

            self.model.train()

            train_loss = 0

            for X_batch, corr_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                corr_batch = corr_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(X_batch, corr_batch)
                loss = self.criterion(preds, y_batch)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            self.model.eval()

            val_loss = 0

            with torch.no_grad():

                for X_batch, corr_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    corr_batch = corr_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    preds = self.model(X_batch, corr_batch)
                    loss = self.criterion(preds, y_batch)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            self.scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), "best_model.pth")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Ранняя остановка на эпохе {epoch+1}")
                    break

            if (epoch != 0) & (epoch % 10 == 0):
                print(f"Эпоха [{epoch}/{epochs}] | Train loss: {avg_train_loss:.6f} | Val loss: {avg_val_loss:.6f}")

        self.model.load_state_dict(torch.load("best_model.pth"))

        return (train_losses, val_losses)

    def evaluate(
        self,
        dataloader
    ):
        self.model.eval()

        total_loss = 0

        with torch.no_grad():

            for X_batch, corr_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                corr_batch = corr_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                preds = self.model(X_batch, corr_batch)
                loss = self.criterion(preds, y_batch)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Test loss: {avg_loss:.6f}")

        return avg_loss

    def predict(
        self,
        dataloader,
        target_scaler,
        train_loader,
        ci_coef=2
    ):
        self.model.eval()

        preds = []
        targets = []

        with torch.no_grad():

            for X_batch, corr_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                corr_batch = corr_batch.to(self.device)
                outputs = self.model(X_batch, corr_batch)
                preds.append(outputs.cpu().numpy())
                targets.append(y_batch.cpu().numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        preds_orig = target_scaler.inverse_transform(preds)
        targets_orig = target_scaler.inverse_transform(targets)

        train_preds = []
        train_targets = []

        with torch.no_grad():

            for X_batch, corr_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                corr_batch = corr_batch.to(self.device)
                output = self.model(X_batch, corr_batch)
                train_preds.append(output.cpu().numpy())
                train_targets.append(y_batch.numpy())

        train_preds = np.concatenate(train_preds)
        train_targets = np.concatenate(train_targets)

        tr_preds_orig = target_scaler.inverse_transform(train_preds)
        tr_targets_orig = target_scaler.inverse_transform(train_targets)

        train_errors = tr_preds_orig - tr_targets_orig
        error_std = np.std(train_errors, axis=0)

        lower_bound = preds_orig - ci_coef * error_std
        upper_bound = preds_orig + ci_coef * error_std

        return (preds_orig, targets_orig, lower_bound, upper_bound)

    @staticmethod
    def run_backtest(
        df: pd.DataFrame,
        test_preds: pd.DataFrame,
        signal_threshold: float = 0,
        init_cash: float = 100_000,
        freq: str = "days",
    ) -> Tuple[vbt.Portfolio, Dict[str, float]]:
        # TODO: подумать над бэктестом для трансформера, доработать этот?
        """
        Бэктест стратегий, основанных на глубинном обучении.

        Args:
            df: Исходный датафрейм.
            test_preds: Предсказания модели глубинного обучения.
            signal_threshold: Коэффициент, определяющий сигнал.
            init_cash: Начальный капитал.
            freq: Гранулярность свечей. В моем случае - "день".

        Returns:
            Объект Portfolio и основные метрики.
        """
        close_prices = df.pivot(index="date", columns="ticker", values="close").sort_index().ffill().bfill()
        test_prices = close_prices.iloc[-len(test_preds):]
        test_prices.index = pd.to_datetime(test_prices.index)
        preds_df = pd.DataFrame(test_preds, index=test_prices.index, columns=close_prices.columns)

        entries = (preds_df > signal_threshold) & (preds_df.shift(1).fillna(0) <= signal_threshold)
        exits = (preds_df < signal_threshold)

        portfolio = vbt.Portfolio.from_signals(
            close=test_prices,
            entries=entries,
            exits=exits,
            init_cash=init_cash,
            size=1/len(close_prices.columns),
            size_type="percent",
            cash_sharing=True,
            group_by=True,
            freq=freq,
            direction="longonly"
        )

        portfolio_weights = vbt.Portfolio.from_signals(
            close=test_prices,
            entries=entries,
            exits=exits,
            init_cash=init_cash,
            size=1/len(close_prices.columns),
            size_type="percent",
            cash_sharing=True,
            group_by=False,
            freq=freq,
            direction="longonly"
        )

        weights_df = portfolio_weights.asset_value().divide(portfolio_weights.value(), axis=0).fillna(0)

        nn_metrics = portfolio.stats(metrics=["sharpe_ratio", "sortino_ratio", "max_dd", "total_return"])

        return (portfolio, weights_df, dict(nn_metrics))
