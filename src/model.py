"""
CNN-классификатор для Voice Anti-Spoofing.
Вход: log-Mel спектрограмма (1, n_mels, time).
Выход: logits по классам (Real=0, Fake=1).
Архитектуру можно расширять (больше блоков, residual) для unseen generator.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2D + BatchNorm + ReLU + MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, pool_size: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.pool(x)
        return x


class CNNClassifier(nn.Module):
    """
    3 Conv2D блока (BatchNorm + ReLU + MaxPool),
    Flatten → Dense 128 ReLU → Dropout 0.5 → Dense 2 (logits).
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_mels: int = 80,
        n_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_classes = n_classes

        self.conv1 = ConvBlock(in_channels, 32, kernel_size=3, pool_size=2)   # 32
        self.conv2 = ConvBlock(32, 64, kernel_size=3, pool_size=2)           # 64
        self.conv3 = ConvBlock(64, 128, kernel_size=3, pool_size=2)          # 128

        # Global pooling — не зависит от длины сегмента (удобно для разного segment_length)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, n_mels, time)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)   # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
