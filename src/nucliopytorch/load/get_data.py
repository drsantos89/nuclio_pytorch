"""Get data."""
from typing import Any

import torch
import torchvision


def download_data_fashion() -> tuple[Any, Any]:
    """Download MNIST Fashion data."""
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


def download_data_numbers() -> tuple[Any, Any]:
    """Download MNIST data."""
    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    return train_dataset


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
