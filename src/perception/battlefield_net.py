from __future__ import annotations

import torch
import torch.nn as nn


class BattlefieldScreenNet(nn.Module):
    """Tiny CNN for binary battlefield vs non-battlefield on a small RGB square."""

    def __init__(self, in_ch: int = 3, base: int = 24) -> None:
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 3
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.25)
        self.head = nn.Linear(c3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dropout(self.features(x).flatten(1))
        return self.head(z).squeeze(-1)
