from __future__ import annotations

import torch
from torch import nn


class HDCQNetwork(nn.Module):
    def __init__(self, hdc_dim: int, action_size: int, hidden_sizes: tuple[int, int]) -> None:
        super().__init__()
        first_hidden, second_hidden = hidden_sizes
        self.layers = nn.Sequential(
            nn.Linear(hdc_dim, first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, action_size),
        )

    def forward(self, encoded_state: torch.Tensor) -> torch.Tensor:
        return self.layers(encoded_state)
