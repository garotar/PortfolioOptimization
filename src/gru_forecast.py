import torch
import torch.nn as nn
from cnn_embedding import CNNEmbedding


class GRUForecaster(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_tickers: int = 8,
        gru_hidden: int = 64,
        cnn_embed_dim: int = 32
    ):

        super().__init__()

        self.cnn_embedding = CNNEmbedding(num_tickers=num_tickers,
                                          embed_dim=cnn_embed_dim)

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(gru_hidden + cnn_embed_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(64, num_tickers)
        )

    def forward(self, x_features, x_corr):
        gru_out, _ = self.gru(x_features)

        gru_emb = gru_out[:, -1, :]
        cnn_emb = self.cnn_embedding(x_corr)

        combined_emb = torch.cat([gru_emb, cnn_emb], dim=1)

        forecast = self.fc(combined_emb)

        return forecast
