"""CNN models."""
from typing import Any

import torch
from torch import nn


class TorchCNN(nn.Module):
    """Main Torch CNN class.

    input_size
        Size of the input in the vertical or horizontal dimmension.
    input_channels
        Number of input channels for the first CNN layer.
    output_channels
        Number of output channels for the first CNN layer = number of filters.
    kernel_size
        Size of the kernel.
    stride
        Stride.
    dropout
        Dropout rate.
    batch_norm
        Whether to use batch normalization.
    activation
        Activation function for the CNN layers.
    output_size
        Size of the output = number of classes.
    """

    def __init__(
        self,
        input_size: int = 28,
        input_channels: int = 1,
        output_channels: int = 10,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: str = "relu",
        output_size: int = 10,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv2d(
            in_channels=output_channels,
            out_channels=output_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = nn.ReLU()
        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        self.fc = nn.Linear(output_channels * 2 * input_size**2, output_size)

    def _cnn(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone CNN network."""
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Uses the backbone CNN network and adds a fully connected layer for classification.
        """
        x = self._cnn(x)
        x = self.fc(x)
        return x


class TorchCNNMultitask(TorchCNN):
    """Torch CNN multitask.

    Uses the backbone CNN network and adds two fully connected layers for classification.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fc2 = nn.Linear(
            kwargs["output_channels"] * 2 * kwargs["input_size"] ** 2, 10
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = self._cnn(x)
        x1 = self.fc(x)
        x2 = self.fc2(x)
        return x1, x2


def get_efficientnet() -> torch.nn.Module:
    """Get efficientnet.

    Download the pretrained EfficientNet_V2_S model
    Replace the first layer with a 1-channel.
    The last layer is replaced with a 10-class classifier.
    """
    from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.features[0] = nn.Sequential(
        nn.Conv2d(
            in_channels=1,
            out_channels=24,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        ),
        nn.BatchNorm2d(
            24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
        ),
        nn.SiLU(),
    )

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, 10),
    )

    return model
