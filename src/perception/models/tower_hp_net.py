from __future__ import annotations

import torch
import torch.nn as nn


class TowerHpNet(nn.Module):
    """
    CRNN-style model for tower HP recognition.

    - Presence head predicts whether text is present (`HAS_HP`) or absent (`EMPTY`).
    - CTC head predicts per-timestep digit logits for text-present crops.
    """

    def __init__(self, in_ch: int = 3, base: int = 32, digit_classes: int = 11) -> None:
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 3
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.temporal = nn.GRU(
            input_size=c3,
            hidden_size=c3,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.ctc_head = nn.Linear(c3 * 2, digit_classes)
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(c3 * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
        - ctc_logits: [B, T, C] where C includes CTC blank class
        - presence_logits: [B] raw logits for HAS_HP (1) vs EMPTY (0)
        """
        z = self.features(x).squeeze(2)  # [B, C, T]
        t = z.permute(0, 2, 1)  # [B, T, C]
        t, _ = self.temporal(t)
        ctc_logits = self.ctc_head(t)
        presence_logits = self.presence_head(t.transpose(1, 2)).squeeze(1)
        return ctc_logits, presence_logits

