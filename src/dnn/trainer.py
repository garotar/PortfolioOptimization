import os
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler=None,
        device="cpu"
    ):
        """
        Класс для обучения, валидации и тестирования DL моделей.

        Args:
            model: Pytorch модель.
            criterion: Функция потерь.
            optimizer: Оптимизатор для обновления параметров модели.
            scheduler: Планировщик скорости обучения.
            device: Устройство для выполнения вычислений.
        """
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
        patience=20,
        model_name="model",
        model_path=None
    ):
        if self.scheduler is None:
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)

        best_val_loss = np.inf
        counter = 0

        train_losses = []
        val_losses = []

        if model_path is not None:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            best_model_file = model_path
        else:
            best_model_file = f"./models/{model_name}_best.pth"

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
                torch.save(self.model.state_dict(), best_model_file)
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Ранняя остановка на эпохе {epoch+1}")
                    break

            if (epoch != 0) and (epoch % 10 == 0):
                print(f"Эпоха [{epoch}/{epochs}] | Train loss: {avg_train_loss:.6f} | Val loss: {avg_val_loss:.6f}")

        self.model.load_state_dict(torch.load(best_model_file))

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
