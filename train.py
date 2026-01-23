from __future__ import annotations

import argparse
import os
import sys

# Allow running as: `python scripts/train.py ...`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data import DataConfig, get_dataloaders
from src.model import build_resnet18
from src.utils import TrainConfig, evaluate, export_torchscript, save_checkpoint, set_seed, train_one_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10")
    p.add_argument("--data_dir", type=str, default="./data", help="Dataset directory")
    p.add_argument("--out_dir", type=str, default="./checkpoints", help="Where to write checkpoints")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true", help="Disable AMP even if CUDA is available")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(
        DataConfig(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    )

    model = build_resnet18(num_classes=10).to(device)

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        use_amp=(not args.no_amp),
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    scaler = amp.GradScaler(enabled=(device.type == "cuda" and cfg.use_amp))

    best_acc = 0.0
    best_ckpt = None

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler)
        va_loss, va_acc = evaluate(model, test_loader, device, criterion)
        scheduler.step()

        if va_acc > best_acc:
            best_acc = va_acc
            best_ckpt = save_checkpoint(model, args.out_dir)
            print(f"Saved BEST checkpoint: {best_ckpt}")

            # Also export TorchScript alongside it
            ts_path = best_ckpt.replace(".pth", ".ts.pt")
            export_torchscript(model, ts_path)

        print(
            f"Epoch {epoch:03d} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val_loss {va_loss:.4f} acc {va_acc:.4f} | best {best_acc:.4f}"
        )

    print("Best val_acc:", best_acc)
    print("Best checkpoint:", best_ckpt)


if __name__ == "__main__":
    main()
