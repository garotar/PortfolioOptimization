import torch
import torch.nn as nn
from src.dnn.cnn_embedding import CNNEmbedding


class GRUForecaster(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_tickers: int,
        gru_hidden: int = 64,
        gru_layers: int = 2,
        dropout: float = 0.3,
        cnn_embed_dim: int = 32
    ):

        super().__init__()

        self.cnn_embedding = CNNEmbedding(
            num_tickers=num_tickers,
            embed_dim=cnn_embed_dim
        )

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(gru_hidden + cnn_embed_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(64, num_tickers)
        )

    def forward(self, x_features, x_corr):
        # x_features (batch_size, window, num_tickers*num_features)
        # x_corr (batch_size, window, num_tickers, num_tickers)

        gru_out, _ = self.gru(x_features)

        # (batch_size, lstm_hidden)
        gru_emb = gru_out[:, -1, :]

        # (batch_size, cnn_embed_dim)
        cnn_emb = self.cnn_embedding(x_corr)

        # объединение эмбеддингов (batch_size, lstm_hidden+cnn_embed_dim)
        combined_emb = torch.cat([gru_emb, cnn_emb], dim=1)

        forecast = self.fc(combined_emb)

        return forecast
