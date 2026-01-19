from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torchvision import datasets, transforms


# CIFAR-10 channel stats (commonly used)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class DataConfig:
    data_dir: str
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True


def get_dataloaders(cfg: DataConfig) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Return train/test DataLoaders for CIFAR-10."""

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_ds = datasets.CIFAR10(root=cfg.data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=cfg.data_dir, train=False, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_loader, test_loader
