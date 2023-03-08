"""Torch FF model."""
import torch
from torch import nn


class TorchFF(nn.Module):
    """Main Torch FF class."""

    def __init__(
        self, input_size: int = 784, hidden_size: int = 128, output_size: int = 10
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.hl1 = nn.Linear(hidden_size, hidden_size)
        self.hl2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.input(x)
        x = self.relu(x)
        x = self.hl1(x)
        x = self.relu(x)
        x = self.hl2(x)
        x = self.relu(x)
        x = self.output(x)
        return x
