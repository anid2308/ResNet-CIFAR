from __future__ import annotations

import argparse
import os
import sys

# Allow running as: `python scripts/ptq_fx.py ...`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from src.data import DataConfig, get_dataloaders
from src.model import build_resnet18
from src.utils import benchmark_cpu_latency_ms, evaluate, model_size_mb_state_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-training quantization (INT8) using FX graph mode")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--calib_batches", type=int, default=100)
    p.add_argument("--out_dir", type=str, default="./outputs")
    return p.parse_args()


def _quant_imports():
    try:
        from torch.ao.quantization import QConfigMapping, get_default_qconfig
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
        return QConfigMapping, get_default_qconfig, prepare_fx, convert_fx
    except Exception:
        # Compatibility fallback (older torch)
        from torch.ao.quantization import QConfigMapping, get_default_qconfig
        from torch.quantization.quantize_fx import prepare_fx, convert_fx  # type: ignore
        return QConfigMapping, get_default_qconfig, prepare_fx, convert_fx


@torch.inference_mode()
def calibrate(prepared_model: torch.nn.Module, loader, num_batches: int) -> None:
    prepared_model.eval()
    for i, (x, _) in enumerate(loader):
        x = x.to("cpu")
        _ = prepared_model(x)
        if i + 1 >= num_batches:
            break


def main() -> None:
    args = parse_args()

    # Load FP32
    model_fp32 = build_resnet18(num_classes=10)
    model_fp32.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model_fp32 = model_fp32.eval().cpu()

    # Dataloaders for calibration + eval
    _, test_loader = get_dataloaders(
        DataConfig(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Baseline (CPU)
    base_loss, base_acc = evaluate(model_fp32, test_loader, torch.device("cpu"), criterion)
    base_size = model_size_mb_state_dict(model_fp32)
    base_lat = benchmark_cpu_latency_ms(model_fp32, num_threads=1)

    # Quantization engine
    supported = torch.backends.quantized.supported_engines
    engine = "fbgemm" if "fbgemm" in supported else ("qnnpack" if "qnnpack" in supported else supported[0])
    torch.backends.quantized.engine = engine
    print("Quant engine:", engine)

    QConfigMapping, get_default_qconfig, prepare_fx, convert_fx = _quant_imports()

    qconfig = get_default_qconfig(engine)
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    example_inputs = (torch.randn(1, 3, 32, 32),)

    prepared = prepare_fx(model_fp32, qconfig_mapping, example_inputs)
    calibrate(prepared, test_loader, num_batches=args.calib_batches)

    model_int8 = convert_fx(prepared).eval().cpu()

    int8_loss, int8_acc = evaluate(model_int8, test_loader, torch.device("cpu"), criterion)
    int8_size = model_size_mb_state_dict(model_int8)
    int8_lat = benchmark_cpu_latency_ms(model_int8, num_threads=1)

    print(f"[BASE FP32]  acc={base_acc:.4f} loss={base_loss:.4f} size={base_size:.1f}MB latency={base_lat:.2f}ms")
    print(f"[PTQ INT8]   acc={int8_acc:.4f} loss={int8_loss:.4f} size={int8_size:.1f}MB latency={int8_lat:.2f}ms")
    print(f"Size shrink: {base_size/int8_size:.2f}x | Latency speedup: {base_lat/int8_lat:.2f}x | Acc drop: {(base_acc-int8_acc)*100:.2f} pts")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "resnet18_cifar10_int8_fx_state_dict.pth")
    torch.save(model_int8.state_dict(), out_path)
    print("Saved INT8 state_dict to:", out_path)


if __name__ == "__main__":
    main()
