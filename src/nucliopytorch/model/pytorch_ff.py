"""Torch FF model."""
import torch
from torch import nn


class TorchFF(nn.Module):
    """Main Torch FF class."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 128,
        output_size: int = 10,
        dropout: float = 0.0,
        batch_norm: bool = False,
        l2: float = 0.0,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
            self.bn3 = nn.BatchNorm1d(hidden_size)
        self.hl1 = nn.Linear(hidden_size, hidden_size)
        self.hl2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.input(x)
        x = self.relu(x)
        x = self.dropout(x)
        if hasattr(self, "bn1"):
            x = self.bn1(x)
        x = self.hl1(x)
        x = self.relu(x)
        x = self.dropout(x)
        if hasattr(self, "bn2"):
            x = self.bn2(x)
        x = self.hl2(x)
        x = self.relu(x)
        x = self.dropout(x)
        if hasattr(self, "bn3"):
            x = self.bn3(x)
        x = self.output(x)
        return x
