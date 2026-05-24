# AGENTS.md

## Quick commands

```bash
# Install (uv)
uv sync

# CLI entry point
lkcellpose train --backbone convnextv2_base --epochs 10
lkcellpose predict <path> --weights best.pt --backbone convnextv2_base
lkcellpose val --weights best.pt --backbone convnextv2_base
lkcellpose info --backbone convnextv2_base --flops

# Python API
from lkcellpose import LKCellposeModel
model = LKCellposeModel("convnextv2_base")
model.train(epochs=10)
```

No test suite exists yet. Verify changes by importing and running a forward+backward pass:
```bash
uv run python -c "
import torch; from lkcellpose.nn.tasks import PanopticCellposeModel
m = PanopticCellposeModel('convnextv2_nano', pretrained=False, panoptic=True)
out = m(torch.randn(1,3,256,256)); out.sum().backward()
print(f'OK: {out.shape}')
"
```

## Architecture

Two-layer facade: `LKCellposeModel` (engine/model.py) → `PanopticCellposeModel` (nn/tasks.py). The facade resolves trainer/validator/predictor by dotted class path from `task_map`.

### Model pipeline
`encoder → UNetDecoder → PanopticHead → (B, 8, H, W)`

Output layout: `[Y-flow, X-flow, cellprob, class0..class4]` (panoptic) or `[Y-flow, X-flow, cellprob]` (instance-only).

### Backbones (22 variants)
- **convnextv2**: nano/tiny/small/base/large/huge/atto/femto/pico (timm)
- **fastvit**: t8/t12/s12/s24/s36/sa12/sa24/sa36 (timm)
- **unireplknet**: t/s/b/l (vendored, not timm — see below)
- **cellpose_unet**: original Cellpose U-Net encoder
- **cellpose_sam**: stub only (not implemented)

Registry: `lkcellpose/nn/backbones/__init__.py` — `BACKBONE_REGISTRY` maps name→builder, `VARIANT_CHANNELS` maps name→4-stage channel list.

### Decoder (Cellpose-faithful)
`lkcellpose/nn/neck.py:UNetDecoder` matches original Cellpose v3 `resnet_torch.py`:
- Skip connections: **addition** (not concatenation) — `self.concatenation = False`
- Upsampling: `nn.Upsample(nearest)` (no learned transposed conv)
- Style vector: GAP + L2 norm from deepest encoder features, FiLM-like broadcast
- Pre-activation: BN→ReLU→Conv (not Conv→BN→GELU)
- BN momentum: 0.05 (not PyTorch default 0.1)
- 4 convs per residual block (2 residual pairs), matching `resup`

### Key modules
- `lkcellpose/nn/modules/conv.py`: PreActConv, StyleConv, ResUpBlock, ConvBlock (legacy), Conv, DWConv
- `lkcellpose/nn/heads.py`: PanopticHead (1x1 conv, no BN/act)
- `lkcellpose/nn/backbones/cellpose_unet.py`: ResDownBlock (pre-act, 4 convs, BN momentum 0.05)

## Critical gotchas

- **UniRepLKNet is vendored** at `lkcellpose/nn/backbones/unireplknet.py` (584 lines, Apache 2.0). It is NOT from timm. Must set `attempt_use_lk_impl=False` to avoid CUTLASS/iGEMM CUDA dependency.
- **class_map dtype is int16**, with 255 as ignore index. `PanopticLoss._focal_loss` casts to `.long()` internally for `cross_entropy`.
- **bfloat16 backward fails on CPU** (avx2_vnni_2). AMP bfloat16 only works on CUDA.
- **No GradScaler** — bfloat16 AMP uses `torch.amp.autocast` only, no GradScaler.
- **Gradient checkpointing** is on by default (`grad_checkpoint: true`). Gradient accum steps=32 (effective batch=256).
- **`main.py` is a stub** — not the entry point. Use `lkcellpose` CLI or `LKCellposeModel`.
- **Flow fields** are computed via `cellpose.dynamics` and cached as `.npz`. Set `cleanup_cache: true` to delete after training.

## Config system

YAML-based, Ultralytics-style override chain:
`default.yaml` ← experiment YAML ← CLI overrides ← Python kwargs

- Default config: `lkcellpose/cfg/default.yaml`
- Experiment configs: `configs/*.yaml`
- Access: `get_cfg(cfg_path, overrides_dict)` returns `IterableSimpleNamespace`

## Data

PanNuke 3-fold CV loaded from HuggingFace (`RationAI/PanNuke`) or local `.npy`. 5 nucleus classes, 19 tissues. Folds: 1/2/3.

## Code style

- No comments unless explicitly requested
- Follow existing patterns in the file being edited
- Ultralytics OOP conventions: BaseTrainer/BaseValidator/BasePredictor, callbacks, dotted-path class resolution
