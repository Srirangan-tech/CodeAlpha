import torch
import torch.nn as nn


class EmotionConvGRUAttention(nn.Module):
    """Compact Conv1D + BiGRU network with learned temporal attention pooling."""

    def __init__(self, n_classes, n_mfcc=40):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mfcc, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.20),
        )
        self.sequence = nn.GRU(
            input_size=128,
            hidden_size=80,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(160, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(160, 96),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(96, n_classes),
        )

    def forward(self, x):
        x = self.encoder(x).transpose(1, 2)
        x, _ = self.sequence(x)
        weights = torch.softmax(self.attention(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return self.head(pooled)
