import torch
import torch.nn as nn


class CNNEmbedding(nn.Module):
    def __init__(self, num_tickers, embed_dim=32):

        super().__init__()

        self.cnn = nn.Sequential(

            # вход - одна матрица корреляций
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),

            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(16*2*2, embed_dim),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x):
        batch_size, window, num_tickers, num_tickers = x.shape
        x = x.view(batch_size * window, 1, num_tickers, num_tickers)
        emb = self.cnn(x)
        emb = emb.view(batch_size, window, -1)

        # усредняем по окну
        emb = emb.mean(dim=1)

        return emb
