from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import amp
from tqdm import tqdm


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    preds = output.argmax(dim=-1)
    return (preds == target).float().mean().item()


@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    use_amp: bool = True


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[amp.GradScaler] = None,
) -> Tuple[float, float]:
    model.train()
    loss_sum, acc_sum, n = 0.0, 0.0, 0

    use_cuda_amp = (device.type == "cuda") and (scaler is not None)

    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type="cuda", enabled=use_cuda_amp):
            out = model(x)
            loss = criterion(out, y)

        if use_cuda_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = y.size(0)
        n += bs
        loss_sum += loss.item() * bs
        acc_sum += accuracy(out, y) * bs

        pbar.set_postfix(loss=loss_sum / n, acc=acc_sum / n)

    return loss_sum / n, acc_sum / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        loss = criterion(out, y)

        bs = y.size(0)
        n += bs
        loss_sum += loss.item() * bs
        acc_sum += accuracy(out, y) * bs

    return loss_sum / n, acc_sum / n


def save_checkpoint(model: nn.Module, out_dir: str, prefix: str = "resnet18_cifar10") -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{prefix}_{stamp}.pth")
    torch.save(model.state_dict(), path)
    return path


def export_torchscript(model: nn.Module, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    example = torch.randn(1, 3, 32, 32, device=next(model.parameters()).device)
    ts = torch.jit.trace(model.eval(), example)
    ts.save(out_path)
    return out_path


def model_size_mb_state_dict(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getbuffer()) / (1024 ** 2)


@torch.inference_mode()
def benchmark_cpu_latency_ms(
    model: nn.Module,
    iters: int = 200,
    warmup: int = 50,
    batch_size: int = 1,
    num_threads: Optional[int] = 1,
) -> float:
    model = model.eval().cpu()
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))

    x = torch.randn(batch_size, 3, 32, 32)
    for _ in range(warmup):
        _ = model(x)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters
