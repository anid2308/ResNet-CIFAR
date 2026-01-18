# ResNet-18 on CIFAR-10 (PyTorch) 

This repo trains a **ResNet-18** CNN on the **CIFAR-10** dataset and includes two compression experiments:

- **Post-Training Quantization (PTQ, INT8, FX graph mode)** for ~4× smaller model size and faster CPU inference
- **Structured pruning + fine-tuning** (mask-based) targeting ~30% sparsity

> Primary entrypoint: **`ResNet_CIFAR10_pt_2.ipynb`** (Google Colab-style paths like `/content/...`).

---

## What’s inside
- ResNet-18 (from `torchvision.models`) with a CIFAR-10 head (`fc → 10 classes`)
- Standard CIFAR-10 augmentation + normalization
- SGD + CosineAnnealingLR training loop
- Checkpoint + TorchScript export helper
- CPU latency benchmarking helper
- INT8 PTQ via **`torch.ao.quantization.quantize_fx`**
- Structured pruning via **`torch.nn.utils.prune.ln_structured`** + fine-tuning

---

## Results (from the notebook)
All latency numbers are **CPU, batch_size=1, 1 thread**, averaged over 200 iters (50 warmup).

| Model | Test Acc | Size (state_dict) | CPU Latency | Notes |
|---|---:|---:|---:|---|
| FP32 baseline | **0.7866** | **42.7 MB** | **8.69 ms** | Trained 60 epochs |
| INT8 PTQ (FX) | **0.7891** | **10.8 MB** | **2.59 ms** | ~**3.95×** smaller, ~**3.36×** faster |
| Pruned + FT | **0.7180** | (mask-based) | (not measured) | **30.05%** effective sparsity, **-6.86 pts** acc drop |

⚠️ **Important pruning note:** this pruning approach creates **masks / zeros** but **does not shrink tensor shapes**. You usually won’t see big latency wins unless you do channel-slimming surgery (physically removing channels) or use sparse-aware kernels.

---

## Setup

### Option A: Run in Google Colab (recommended)
1. Upload `ResNet_CIFAR10_pt_2.ipynb` to Colab
2. Runtime → Change runtime type → set **GPU** (training is much faster)
3. Run all cells top-to-bottom

### Option B: Run locally
Create a venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install torch torchvision
```
## Training (baseline FP32)

The notebook trains with:
Epochs: 60
Batch size: 128
Optimizer: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
LR schedule: CosineAnnealingLR
Loss: CrossEntropy with label_smoothing=0.1
AMP: enabled on CUDA (torch.amp)

Data pipeline:
Train: RandomCrop(32, padding=4) + RandomHorizontalFlip() + Normalize
Test: Normalize
Normalize stats (CIFAR-10):

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

Artifacts saved (Colab):
FP32 checkpoint: /content/artifacts/resnet18_cifar10_<timestamp>.pth
TorchScript export: /content/artifacts/resnet18_cifar10_<timestamp>.ts.pt

## Evaluate baseline + CPU benchmark

After training, the notebook rebuilds the model and loads a saved checkpoint.
You must update this line in the notebook to your actual saved checkpoint:

```python
BEST_CKPT_PATH = '/content/artifacts/resnet18_cifar10_<timestamp>.pth'
```

Then it reports:
test loss / accuracy
serialized state_dict size in MB
CPU latency in ms

## PTQ INT8 (FX graph mode)
The notebook performs static PTQ using prepare_fx → calibration → convert_fx.

Key details:
Engine: chooses fbgemm if available (common on x86), else qnnpack
Calibration: runs 100 batches through the prepared model (no gradients)
Evaluation + size + CPU latency measured for the INT8 model
Saved artifact (Colab):

INT8 state dict: /content/artifacts_ptq/resnet18_cifar10_int8_fx_state_dict.pth

## Structured Pruning + fine-tuning
Pruning method:
  Applies torch.nn.utils.prune.ln_structured(..., amount=0.30, dim=0) to:
    nn.Conv2d (output channels)
    nn.Linear (output features)

Then fine-tunes:
  FT epochs: 20
  FT lr: 0.01
  SGD (momentum=0.9, weight_decay=5e-4) + CosineAnnealingLR
  Keeps best fine-tuned weights by validation accuracy

Finally:
  prune.remove(...) is called to make pruning permanent (keeps zeros)
  Evaluates pruned model on CPU

Saved artifact (Colab):
  pruned+FT state dict: /content/artifacts_pruned/resnet18_cifar10_pruned_ft_state_dict.pth
  
## Reproducing the benchmark numbers
  The notebook’s CPU benchmarking uses:
    torch.set_num_threads(1) (when num_threads=1 is passed)
    random input tensor of shape (batch_size, 3, 32, 32)
    warmup + timed loop

  For more stable results:
    run on an idle machine
    repeat several trials and report mean/std

## Known Limitations/ next improvements
Pruning accuracy drop is large in the current notebook run (~6.86 pts). Likely improvements:
prune less aggressively (e.g., 10–20%) or layer-wise pruning
longer fine-tuning or better LR schedule
avoid pruning early layers heavily
try channel pruning with actual model surgery to realize latency gains

PTQ currently looks great (very small/no acc drop). If you see drops on other machines:
increase calibration batches
ensure preprocessing is identical (Normalize)
confirm model is in eval() during quantization

## License
MIT

## Credits
CIFAR-10 dataset (TorchVision)
ResNet architecture (TorchVision)
