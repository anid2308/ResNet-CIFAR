from __future__ import annotations

import argparse
import os
import sys

# Allow running as: `python scripts/benchmark_cpu.py ...`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.model import build_resnet18
from src.utils import benchmark_cpu_latency_ms, model_size_mb_state_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark CPU latency for a checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_threads", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = build_resnet18(num_classes=10)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval().cpu()

    size_mb = model_size_mb_state_dict(model)
    lat_ms = benchmark_cpu_latency_ms(
        model,
        iters=args.iters,
        warmup=args.warmup,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )

    print(f"checkpoint={args.checkpoint}")
    print(f"state_dict_size_mb={size_mb:.2f}")
    print(f"cpu_latency_ms={lat_ms:.2f} (batch_size={args.batch_size}, threads={args.num_threads})")


if __name__ == "__main__":
    main()
