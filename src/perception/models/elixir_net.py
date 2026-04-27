from __future__ import annotations

import torch
import torch.nn as nn


class ElixirDigitNet(nn.Module):
    """Tiny CNN for multiclass elixir digit classification (0..10)."""

    def __init__(self, in_ch: int = 3, base: int = 20, num_classes: int = 11) -> None:
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 3
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dropout(self.features(x).flatten(1))
        return self.head(z)
