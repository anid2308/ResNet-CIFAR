from __future__ import annotations

import argparse
import os
import sys

# Allow running as: `python scripts/eval.py ...`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from src.data import DataConfig, get_dataloaders
from src.model import build_resnet18
from src.utils import evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a ResNet-18 checkpoint on CIFAR-10")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    _, test_loader = get_dataloaders(
        DataConfig(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    )

    model = build_resnet18(num_classes=10)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss, acc = evaluate(model, test_loader, device, criterion)

    print(f"checkpoint={args.checkpoint}")
    print(f"test_loss={loss:.4f} test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
