import torch
import torch.nn as nn
from src.dnn.cnn_embedding import CNNEmbedding


class AttentionLSTMForecaster(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_tickers: int = 8,
        lstm_hidden: int = 64,
        cnn_embed_dim: int = 32,
        attention_heads: int = 4,
        lstm_layers: int = 2,
        dropout: float = 0.4
    ):

        super().__init__()

        self.cnn_embedding = CNNEmbedding(
            num_tickers=num_tickers,
            embed_dim=cnn_embed_dim
        )

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=attention_heads,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(lstm_hidden)

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + cnn_embed_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),
            nn.Linear(64, num_tickers)
        )

    def forward(self, x_features, x_corr):
        lstm_out, _ = self.lstm(x_features)

        # (batch_size, 1, hidden_size)
        query = lstm_out[:, -1:, :]

        # (batch_size, seq_len, hidden_size)
        key = value = lstm_out

        attn_output, attn_weights = self.attention(
            query=query,
            key=key,
            value=value
        )
        attn_output = attn_output.squeeze(1)
        attn_output = self.layer_norm(attn_output + query.squeeze(1))

        cnn_emb = self.cnn_embedding(x_corr)

        combined_emb = torch.cat([attn_output, cnn_emb], dim=1)

        forecast = self.fc(combined_emb)

        return forecast
