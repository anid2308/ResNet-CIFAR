
# ResNet-18 on CIFAR-10 (PyTorch) 🧠📷

A script-based repo for training a **ResNet-18** CNN on **CIFAR-10**, plus two compression experiments:

- **Post-Training Quantization (PTQ, INT8, FX graph mode)** for smaller models + faster **CPU** inference
- **Structured pruning + fine-tuning (mask-based)** to reduce the effective number of nonzero weights

This repo was refactored from the original notebook: `notebooks/ResNet_CIFAR10_pt_2.ipynb`.

---

## What’s implemented

- **Model**: `torchvision.models.resnet18(weights=None)` with the final `fc` layer replaced for **10 classes** (see `src/model.py`).
- **Data**: CIFAR-10 auto-download via TorchVision + standard augmentation (see `src/data.py`):
  - Train: `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` + Normalize
  - Test: Normalize
  - Mean/Std: `(0.4914, 0.4822, 0.4465)` / `(0.2470, 0.2435, 0.2616)`
- **Training**: SGD + CosineAnnealingLR + optional AMP on CUDA (see `scripts/train.py`).
- **Evaluation**: test loss/accuracy from a saved `state_dict` checkpoint (see `scripts/eval.py`).
- **INT8 PTQ**: FX graph mode quantization using `torch.ao.quantization.quantize_fx` (see `scripts/ptq_fx.py`).
- **Pruning**: structured pruning with masks (`torch.nn.utils.prune.ln_structured`) + fine-tuning, then `prune.remove` to make zeros permanent (see `scripts/prune_finetune.py`).
- **CPU benchmarking**: state_dict size + CPU latency (see `scripts/benchmark_cpu.py`).

---

## Results (from the original notebook run)

CPU latency measured with **batch_size=1**, **1 thread**, averaged over **200 iters** (50 warmup).

| Model | Test Acc | Size (state_dict) | CPU Latency | Notes |
|---|---:|---:|---:|---|
| FP32 baseline | **0.7866** | **42.7 MB** | **8.69 ms** | 60 epochs |
| INT8 PTQ (FX) | **0.7891** | **10.8 MB** | **2.59 ms** | ~**3.95×** smaller, ~**3.36×** faster |
| Pruned + FT | **0.7180** | (mask-based) | (not measured) | **30.05%** effective sparsity, **-6.86 pts** acc drop |

> Pruning here creates **masks/zeros** but does **not** shrink tensor shapes. Big speedups usually require channel-slimming surgery (physically removing channels) or sparse-aware kernels.

---

## Repo structure

```text
resnet18-cifar10/
  scripts/
    train.py
    eval.py
    ptq_fx.py
    prune_finetune.py
    benchmark_cpu.py
  src/
    __init__.py
    data.py
    model.py
    utils.py
  notebooks/
    ResNet_CIFAR10_pt_2.ipynb
  checkpoints/        # created automatically (gitignored)
  outputs/            # created automatically (gitignored)
  requirements.txt
  LICENSE
  README.md
  .gitignore
```

Notes:
- `data/` is also gitignored by default.
- The scripts add the repo root to `sys.path`, so you can run them directly with `python scripts/<name>.py ...`.

---

## Setup

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install **PyTorch + TorchVision** for your machine (CPU or CUDA). Make sure the versions match.

Then install the remaining small dependencies:

```bash
pip install -r requirements.txt
```

If you see `operator torchvision::nms does not exist`, it’s almost always a Torch/TorchVision version mismatch.

---

## Train (FP32)

Downloads CIFAR-10 automatically into `--data_dir`.

```bash
python scripts/train.py \
  --data_dir ./data \
  --out_dir ./checkpoints \
  --epochs 60 \
  --batch_size 128 \
  --lr 0.1 \
  --weight_decay 5e-4
```

Common extra flags:
- `--seed 42` (default)
- `--no_amp` to disable AMP even if CUDA is available
- `--num_workers 2` (default)

Outputs:
- best checkpoint (state_dict): `checkpoints/resnet18_cifar10_<timestamp>.pth`
- TorchScript export: `checkpoints/resnet18_cifar10_<timestamp>.ts.pt`

---

## Evaluate

```bash
python scripts/eval.py \
  --checkpoint checkpoints/resnet18_cifar10_<timestamp>.pth \
  --data_dir ./data \
  --device cpu
```

---

## Benchmark CPU latency

```bash
python scripts/benchmark_cpu.py \
  --checkpoint checkpoints/resnet18_cifar10_<timestamp>.pth \
  --batch_size 1 \
  --num_threads 1
```

This prints:
- `state_dict_size_mb`
- `cpu_latency_ms` (average over `--iters`, with `--warmup` ignored from timing)

---

## PTQ INT8 (FX graph mode)

Runs baseline FP32 evaluation/benchmark on CPU, then quantizes via FX graph mode and repeats evaluation/benchmark.

```bash
python scripts/ptq_fx.py \
  --checkpoint checkpoints/resnet18_cifar10_<timestamp>.pth \
  --data_dir ./data \
  --calib_batches 100 \
  --out_dir ./outputs
```

Outputs:
- INT8 state_dict: `outputs/resnet18_cifar10_int8_fx_state_dict.pth`

Notes:
- The script auto-selects a quantization engine (`fbgemm` if available, else `qnnpack`).
- Calibration is performed by running `--calib_batches` batches through the prepared model on CPU.

---

## Structured pruning + fine-tuning (mask-based)

Applies structured pruning masks to Conv2d/Linear layers (output channels/features), then fine-tunes and makes pruning permanent.

```bash
python scripts/prune_finetune.py \
  --checkpoint checkpoints/resnet18_cifar10_<timestamp>.pth \
  --data_dir ./data \
  --prune_ratio 0.30 \
  --finetune_epochs 20 \
  --finetune_lr 0.01 \
  --out_dir ./outputs
```

Outputs:
- pruned+fine-tuned state_dict: `outputs/resnet18_cifar10_pruned_ft_state_dict.pth`

Notes:
- This pruning method creates zeros in weights but does not change tensor shapes.
- The script prints **effective sparsity** before and after fine-tuning.

---

## Notes (vs the notebook)

The original notebook’s baseline training loop didn’t explicitly call `optimizer.zero_grad()` inside `train_epoch` (the fine-tune loop did). This repo uses the standard pattern (`zero_grad` every batch), so exact numbers may shift slightly, but training is more correct/stable.

---

## License

MIT — see `LICENSE`.
