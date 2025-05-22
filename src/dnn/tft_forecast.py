import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


class TFTForecaster:
    def __init__(
        self,
        df: pd.DataFrame,
        training_cutoff: int
    ):
        self.df = df
        self.training_cutoff = training_cutoff
        self.training_set = None
        self.validation_set = None
        self.train_loader = None
        self.val_loader = None
        self.best_params = None
        self.trained_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def prepare_data(self, forecast_horizon=14):
        data = self.df[["ticker", "date", "close", "volume"]].copy()
        grouped = data.groupby("ticker")

        data["return"] = grouped["close"].pct_change().fillna(0)
        for period in [5, 10, 20]:
            data[f"return_{period}d"] = grouped["close"].pct_change(periods=period)

        for window in [5, 10, 20]:
            rolling_mean = grouped["close"].transform(lambda x: x.rolling(window).mean())
            data[f"ma_return_{window}d"] = data["close"] / rolling_mean - 1

        for window in [5, 10, 20]:
            data[f"volatility_{window}d"] = grouped["close"].pct_change().rolling(window).std().reset_index(level=0, drop=True)

        data["volume_ma5"] = grouped["volume"].rolling(window=5).mean().reset_index(level=0, drop=True)
        data["volume_ma20"] = grouped["volume"].rolling(window=20).mean().reset_index(level=0, drop=True)
        data["volume_ratio"] = data["volume_ma5"] / data["volume_ma20"]

        # data["target_return"] = grouped["close"].pct_change(periods=-forecast_horizon).shift(forecast_horizon) ###

        data.fillna(method="ffill", inplace=True)
        data.fillna(0, inplace=True)

        data["time_idx"] = data.groupby("ticker").cumcount()
        data["day_of_week"] = data["date"].dt.weekday
        data["month"] = data["date"].dt.month
        data["day_of_month"] = data["date"].dt.day
        # print(data.corr(numeric_only=True))

        unknown_reals = [
            "volume", "close", "return",
            "return_5d", "return_10d", "return_20d",
            "ma_return_5d", "ma_return_10d", "ma_return_20d",
            "volatility_5d", "volatility_10d", "volatility_20d",
            "volume_ma5", "volume_ma20", "volume_ratio"
        ]

        training_cutoff = data["time_idx"].max() - self.training_cutoff

        self.training_set = TimeSeriesDataSet(
            data[lambda x: x.time_idx < training_cutoff],
            time_idx="time_idx",
            target="return",
            group_ids=["ticker"],
            min_encoder_length=90,
            max_encoder_length=150,
            min_prediction_length=1,
            max_prediction_length=142,
            static_categoricals=["ticker"],
            time_varying_known_reals=["day_of_week", "month", "day_of_month"],
            time_varying_unknown_reals=unknown_reals,
            target_normalizer=GroupNormalizer(groups=["ticker"], transformation=None),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True
        )

        self.validation_set = TimeSeriesDataSet.from_dataset(
            dataset=self.training_set,
            data=data,
            predict=True,
            stop_randomization=True
        )

        self.train_loader = self.training_set.to_dataloader(
            train=True,
            batch_size=512,
            num_workers=0
        )
        self.val_loader = self.validation_set.to_dataloader(
            train=False,
            batch_size=512,
            num_workers=0
        )

    def evaluate_baseline(self):
        actuals = torch.cat([y for x, (y, _) in iter(self.val_loader)]).to(self.device)
        baseline_preds = Baseline().predict(self.val_loader).to(self.device)
        baseline_mae = (actuals - baseline_preds).abs().mean().item()
        # smape_metric = SMAPE()
        # baseline_smape = smape_metric(actuals, baseline_preds).item()
        print(f"Бейзлайн MAE: {baseline_mae:.4f}")
        # print(f"Бейзлайн SMAPE: {baseline_smape:.4f}")
        return baseline_mae

    def tune_hyperparameters(
        self,
        n_trials: int = 150,
        max_epochs: int = 50
    ):
        study = optimize_hyperparameters(
            self.train_loader,
            self.val_loader,
            model_path="optuna_tft",
            n_trials=n_trials,
            max_epochs=max_epochs,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(64, 512),
            hidden_continuous_size_range=(32, 512),
            attention_head_size_range=(1, 8),
            dropout_range=(0.05, 0.3),
            learning_rate_range=(0.0001, 0.01),
            trainer_kwargs=dict(accelerator="gpu", devices=1, gradient_clip_val=0.1),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
            verbose=False
        )
        self.best_params = study.best_trial.params
        print("Лучшие параметры:", self.best_params)

    def train_final_model(
        self,
        max_epochs: int = 100,
        params=None
    ):
        if self.best_params is not None:
            cfg = self.best_params
            print("Используются параметры из Optuna.")
        elif params is not None:
            cfg = params
            self.best_params = params
            print("Используются кастомные параметры.")
        else:
            raise ValueError(
                "Нет параметров обучения: задайте self.best_params или передайте params"
            )

        self.trained_model = TemporalFusionTransformer.from_dataset(
            self.training_set,
            learning_rate=cfg["learning_rate"],
            hidden_size=cfg["hidden_size"],
            attention_head_size=cfg["attention_head_size"],
            hidden_continuous_size=cfg["hidden_continuous_size"],
            dropout=cfg["dropout"],
            loss=QuantileLoss(),
            output_size=7,
            log_interval=10,
            reduce_on_plateau_patience=4
        )

        early_stop_callback = EarlyStopping(monitor="val_loss", patience=50)
        lr_logger = LearningRateMonitor()
        logger = TensorBoardLogger("lightning_logs")

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=1,
            gradient_clip_val=cfg["gradient_clip_val"],
            callbacks=[early_stop_callback, lr_logger],
            logger=logger,
            log_every_n_steps=10
        )

        trainer.fit(
            self.trained_model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader
        )

    def evaluate_model(self):
        predictions = self.trained_model.predict(self.val_loader)
        actuals = torch.cat([y[0] for x, y in iter(self.val_loader)]).to(self.device)
        raw_predictions = self.trained_model.predict(self.val_loader, mode="raw", return_x=True)

        mae = (actuals - predictions).abs().mean().item()
        smape_metric = SMAPE()
        smape = smape_metric(actuals, predictions).item()

        print(f"MAE после Optuna: {mae:.4f}")
        # print(f"SMAPE после Optuna: {smape:.4f}")

        return (predictions.cpu().numpy(), actuals.cpu().numpy(), raw_predictions)

    def plot_predictions(
        self,
        num_tickers: int = 8
    ):
        raw_preds = self.trained_model.predict(self.val_loader, mode="raw", return_x=True)
        tickers = self.val_loader.dataset.categorical_encoders["ticker"].inverse_transform(
            raw_preds.x["groups"].cpu().numpy().flatten()
        )

        for idx in range(num_tickers):
            ticker = tickers[idx]
            fig = self.trained_model.plot_prediction(
                raw_preds.x, raw_preds.output, idx=idx, add_loss_to_title=True
            )
            fig.suptitle(f"Ticker: {ticker}")
            fig.tight_layout()
            fig.show()

    def get_quantile_predictions(
        self,
        quantile: float = 0.5
    ):
        raw_preds = self.trained_model.predict(self.val_loader, mode="raw", return_x=True)
        quantile_preds = raw_preds.output.prediction[..., self.trained_model.loss.quantiles.index(quantile)]
        return quantile_preds.cpu().numpy()
