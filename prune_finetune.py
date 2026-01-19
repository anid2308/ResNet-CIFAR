from __future__ import annotations

import argparse
import copy
import os
import sys

# Allow running as: `python scripts/prune_finetune.py ...`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch import amp
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data import DataConfig, get_dataloaders
from src.model import build_resnet18
from src.utils import evaluate, set_seed, train_one_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Structured pruning + fine-tuning for ResNet-18 (mask-based)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prune_ratio", type=float, default=0.30)
    p.add_argument("--finetune_epochs", type=int, default=20)
    p.add_argument("--finetune_lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./outputs")
    return p.parse_args()


def apply_structured_pruning(model: nn.Module, amount: float) -> nn.Module:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prune.ln_structured(m, name="weight", amount=amount, n=2, dim=0)
        elif isinstance(m, nn.Linear):
            prune.ln_structured(m, name="weight", amount=amount, n=2, dim=0)
    return model


def effective_nonzero_params(model: nn.Module) -> tuple[int, int]:
    total, nonzero = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if hasattr(m, "weight_mask") and hasattr(m, "weight_orig"):
                w = (m.weight_orig.detach() * m.weight_mask.detach())
            else:
                w = m.weight.detach()
            total += w.numel()
            nonzero += torch.count_nonzero(w).item()
            if m.bias is not None:
                b = m.bias.detach()
                total += b.numel()
                nonzero += torch.count_nonzero(b).item()
    return nonzero, total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(
        DataConfig(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    )

    # Load checkpoint
    model = build_resnet18(num_classes=10)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device)

    # Prune
    apply_structured_pruning(model, amount=args.prune_ratio)
    nz, tot = effective_nonzero_params(model)
    print(f"Effective sparsity after pruning: {(1 - nz / tot) * 100:.2f}%")

    # Fine-tune
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=args.finetune_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)
    scaler = amp.GradScaler(device_type="cuda", enabled=(device.type == "cuda"))

    best_acc = 0.0
    best_sd = None

    for epoch in range(1, args.finetune_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler)
        va_loss, va_acc = evaluate(model, test_loader, device, criterion)
        scheduler.step()

        if va_acc > best_acc:
            best_acc = va_acc
            best_sd = copy.deepcopy(model.state_dict())

        print(
            f"[FT {epoch:02d}/{args.finetune_epochs}] train_acc={tr_acc:.4f} val_acc={va_acc:.4f} best_val_acc={best_acc:.4f}"
        )

    if best_sd is not None:
        model.load_state_dict(best_sd)

    # Make pruning permanent (keeps zeros, removes reparam)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight_mask"):
            prune.remove(m, "weight")

    # Eval on CPU for apples-to-apples
    model_cpu = copy.deepcopy(model).eval().cpu()
    loss, acc = evaluate(model_cpu, test_loader, torch.device("cpu"), criterion)

    nz2, tot2 = effective_nonzero_params(model_cpu)
    print(f"[PRUNED+FT] acc={acc:.4f} loss={loss:.4f} | effective sparsity={(1 - nz2 / tot2) * 100:.2f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "resnet18_cifar10_pruned_ft_state_dict.pth")
    torch.save(model_cpu.state_dict(), out_path)
    print("Saved pruned+FT state_dict to:", out_path)


if __name__ == "__main__":
    main()
