import torch
import torch.nn as nn
from cnn_embedding import CNNEmbedding


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_tickers: int = 8,
        lstm_hidden: int = 64,
        cnn_embed_dim: int = 32
    ):

        super().__init__()

        self.cnn_embedding = CNNEmbedding(num_tickers=num_tickers,
                                          embed_dim=cnn_embed_dim)

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden+cnn_embed_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(64, num_tickers)
        )

    def forward(self, x_features, x_corr):
        # x_features (batch_size, window, num_tickers*num_features)
        # x_corr (batch_size, window, num_tickers, num_tickers)

        lstm_out, _ = self.lstm(x_features)

        # (batch_size, lstm_hidden)
        lstm_emb = lstm_out[:, -1, :]

        # (batch_size, cnn_embed_dim)
        cnn_emb = self.cnn_embedding(x_corr)

        # объединение эмбеддингов (batch_size, lstm_hidden+cnn_embed_dim)
        combined_emb = torch.cat([lstm_emb, cnn_emb], dim=1)

        forecast = self.fc(combined_emb)

        return forecast
