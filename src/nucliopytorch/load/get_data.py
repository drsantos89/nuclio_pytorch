"""Get data."""
from typing import Any

import torch
import torchvision


def download_data() -> tuple[Any, Any]:
    """Download data."""
    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    return train_dataset, test_dataset


def get_data_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    """Get data loader."""
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
