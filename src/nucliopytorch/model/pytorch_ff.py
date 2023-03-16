"""Torch FF model."""
import torch
from torch import nn


class TorchFF(nn.Module):
    """Main Torch FF class.

    input_size
        Size of the input = number of features.
    hidden_size
        Size of the hidden layer. We have used 2 hidden layers.
    output_size
        Size of the output = number of classes.
    dropout
        Dropout rate.
    batch_norm
        Whether to use batch normalization.
    l2
        L2 regularization.
    """

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
        self.hl1 = nn.Linear(hidden_size, hidden_size)
        self.hl2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.input(x)
        x = self.relu(x)
        if self.dropout.p > 0:
            x = self.dropout(x)
        x = self.hl1(x)
        x = self.relu(x)
        if self.dropout.p > 0:
            x = self.dropout(x)
        x = self.hl2(x)
        x = self.relu(x)
        if self.dropout.p > 0:
            x = self.dropout(x)
        x = self.output(x)
        return x
