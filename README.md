# LK-Cellpose

Large-Kernel ConvNet backbones for Cellpose panoptic nuclei segmentation.

Replaces the SAM ViT-L encoder in Cellpose-SAM with hierarchical ConvNet backbones (ConvNeXt V2, UniRepLKNet, FastViT) connected via a U-Net decoder with skip connections. The panoptic head predicts joint flow fields + class logits for simultaneous instance segmentation and nuclei classification on the PanNuke dataset.

## Architecture

```
Input (3ch, 256x256)
    │
    ▼
┌─────────────────────┐
│   Backbone (4-stage) │─── stage1 [4x] ──┐
│   ConvNeXt V2        │─── stage2 [8x] ──┤
│   UniRepLKNet        │─── stage3 [16x] ─┤  skip
│   FastViT             │─── stage4 [32x] ─┤  connections
└─────────────────────┘                    │
    │                                      │
    ▼                                      ▼
┌─────────────────────┐        ┌─────────────────────┐
│   U-Net Decoder     │◄───────│  Skip Concatenation │
│   [512,256,128,64,32]│       └─────────────────────┘
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   PanopticHead      │  1x1 conv → 8 channels
└─────────────────────┘
    │
    ├── Y-flow (ch 0)
    ├── X-flow (ch 1)
    ├── cellprob  (ch 2)
    └── class logits (ch 3-7, 5 nuclei classes)
    │
    ▼
┌─────────────────────┐
│   Cellpose dynamics  │  Euler integration → instance masks
│   + argmax            │  → per-pixel class labels
└─────────────────────┘
```

## Supported Backbones

### ConvNeXt V2 (7x7 DW-Conv, FCMAE pretrained)

| Variant | Kernel | Params | Pretrained | License |
|---------|--------|--------|------------|---------|
| convnextv2_atto | 7x7 | 3.7M | IN-1K FCMAE | CC BY-NC-ND 4.0 |
| convnextv2_femto | 7x7 | 5.2M | IN-1K FCMAE | CC BY-NC-ND 4.0 |
| convnextv2_pico | 7x7 | 9.1M | IN-1K FCMAE | CC BY-NC-ND 4.0 |
| convnextv2_nano | 7x7 | 15.6M | IN-1K FCMAE | CC BY-NC-ND 4.0 |
| convnextv2_tiny | 7x7 | 28.6M | IN-22K FCMAE | CC BY-NC-ND 4.0 |
| convnextv2_small | 7x7 | 38M | IN-22K FCMAE | CC BY-NC-ND 4.0 |
| convnextv2_base | 7x7 | 89M | IN-22K FCMAE | CC BY-NC-ND 4.0 |
| convnextv2_large | 7x7 | 198M | IN-22K FCMAE | CC BY-NC-ND 4.0 |
| convnextv2_huge | 7x7 | 660M | IN-22K FCMAE | CC BY-NC-ND 4.0 |

### UniRepLKNet (13-17x17 DW-Conv, Dilated Reparam Block)

| Variant | Kernel | Params | Pretrained | License |
|---------|--------|--------|------------|---------|
| unireplknet_t | mixed | 31M | IN-1K | Apache 2.0 |
| unireplknet_s | mixed | 56M | IN-1K | Apache 2.0 |
| unireplknet_b | mixed | 97M | IN-22K | Apache 2.0 |
| unireplknet_l | mixed | 255M | IN-22K | Apache 2.0 |

### FastViT (9-11x11 DW-Conv, Hybrid CNN-Transformer)

| Variant | Kernel | Params | Pretrained | License |
|---------|--------|--------|------------|---------|
| fastvit_t8 | 9-11x11 | 4M | IN-1K | Apple |
| fastvit_t12 | 9-11x11 | 7M | IN-1K | Apple |
| fastvit_s12 | 9-11x11 | 8M | IN-1K | Apple |
| fastvit_sa12 | 9-11x11 | 10M | IN-1K | Apple |
| fastvit_sa24 | 9-11x11 | 19M | IN-1K | Apple |
| fastvit_sa36 | 9-11x11 | 26M | IN-1K | Apple |
| fastvit_ma36 | 9-11x11 | 34M | IN-1K | Apple |

### Baselines

| Variant | Architecture | Params | Notes |
|---------|-------------|--------|-------|
| cellpose_unet | Residual U-Net encoder | ~21M | Custom reimplementation |
| cellpose_sam | SAM ViT-L encoder | ~305M | Uses cellpose.vit_sam.Transformer |

## Quick Start

### Install

```bash
pip install -e .
```

### Train

```bash
# ConvNeXt V2 Base on PanNuke
lkcellpose train --backbone convnextv2_base --epochs 2000 --batch 8 --lr0 5e-5

# FastViT SA36
lkcellpose train --backbone fastvit_sa36 --name fastvit_sa36

# UniRepLKNet Small
lkcellpose train --backbone unireplknet_s --name unireplknet_s

# With config file
lkcellpose train --config configs/convnextv2_base.yaml
```

### Predict

```bash
lkcellpose predict image.png --backbone convnextv2_base --weights runs/exp/weights/best.pt
```

### Validate

```bash
lkcellpose val --backbone convnextv2_base --weights runs/exp/weights/best.pt --val-folds 2
```

### Model Info

```bash
lkcellpose info --backbone convnextv2_base --detailed --flops
```

### Python API

```python
from lkcellpose import LKCellposeModel

# Train
model = LKCellposeModel("convnextv2_base")
model.train(epochs=2000, batch=8)

# Predict
results = model.predict("image.png")
for r in results:
    print(f"Detected {r.n_instances} nuclei")
    r.save("output.png")

# Model info
model_info = LKCellposeModel("convnextv2_base")
nn_model = model_info._resolve_class(
    model_info.task_map["panoptic"]["trainer"]
)().get_model()
info = nn_model.info(detailed=True)
flops = nn_model.flops()
```

## Configuration

All defaults are in `lkcellpose/cfg/default.yaml`. Override via CLI flags, config YAML, or Python kwargs:

```bash
# CLI overrides
lkcellpose train --backbone convnextv2_base --lr0 1e-4 --batch 4

# Config file
lkcellpose train --config configs/convnextv2_base.yaml
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| lr0 | 5e-5 | Peak learning rate |
| weight_decay | 0.1 | AdamW weight decay |
| warmup_epochs | 10 | Linear LR warmup |
| lr_decay_milestones | [-100, -50] | Step decay epochs (relative to total) |
| lr_decay_factor | 10 | LR reduction factor at each milestone |
| grad_accum_steps | 32 | Gradient accumulation (effective batch = 256) |
| batch | 8 | Per-GPU batch size |
| amp_dtype | bfloat16 | AMP dtype (bfloat16, no GradScaler needed) |
| focal_gamma | 2.0 | Focal loss focusing parameter |
| flow_weight | 5.0 | Flow MSE loss scaling |

## Project Structure

```
lkcellpose/
├── cfg/
│   ├── __init__.py           # get_cfg(), cfg2dict(), check_cfg()
│   └── default.yaml          # 87 config keys
├── engine/
│   ├── model.py              # LKCellposeModel facade
│   ├── trainer.py            # BaseTrainer (AMP, grad accum, EMA, callbacks)
│   ├── validator.py          # BaseValidator (callback-driven)
│   ├── predictor.py          # BasePredictor (tiling, overlap averaging)
│   └── results.py            # Results container
├── nn/
│   ├── tasks.py              # BaseModel, PanopticCellposeModel, CellposeSAMModel
│   ├── neck.py               # UNetDecoder (skip connections, grad checkpointing)
│   ├── heads.py              # PanopticHead
│   ├── modules/conv.py       # Conv, DWConv, ConvTranspose2x, DropBlock, ConvBlock
│   └── backbones/
│       ├── convnext.py        # ConvNeXt V2 via timm
│       ├── fastvit.py         # FastViT via timm
│       ├── unireplknet.py     # UniRepLKNet (vendored)
│       └── cellpose_unet.py  # Cellpose U-Net baseline
├── models/panoptic/
│   ├── train.py              # PanopticTrainer
│   ├── val.py                # PanopticValidator
│   └── predict.py            # PanopticPredictor
├── data/
│   ├── pannuke.py            # PanNukeDataset (HuggingFace + .npy)
│   ├── augment.py            # CellposeAugment
│   └── transforms.py         # Flow computation, class maps, caching
├── utils/
│   ├── loss.py               # PanopticLoss (flow MSE + cellprob BCE + focal)
│   ├── metrics.py            # PanopticQuality (bPQ, mPQ, per-class PQ)
│   ├── torch_utils.py        # select_device, ModelEMA, EarlyStopping
│   ├── model_info.py         # model_info(), flops_estimate(), parameter_count()
│   └── callbacks/
│       ├── base.py           # 16 callback events
│       └── tensorboard.py    # TensorBoard logging
├── dynamics.py               # compute_masks(), labels_to_flows() (cellpose wrapper)
├── cli.py                    # CLI: train/predict/val/info
└── __init__.py
```

## Key Design Decisions

- **8-channel output**: [Y-flow, X-flow, cellprob, 5x class logits] enables joint instance segmentation + classification in a single forward pass
- **Focal Loss** with inverse-sqrt-frequency alpha weighting handles PanNuke class imbalance (Dead nuclei: 2,908 vs Neoplastic: 77,403)
- **bfloat16 AMP**: No GradScaler needed; 2x memory savings vs float32
- **Gradient checkpointing**: Enabled by default in U-Net decoder for memory efficiency
- **Flow caching**: Precomputed flow fields stored as .npz, cleaned up after training
- **Cellpose dynamics**: Uses `cellpose.dynamics` for gradient vector field integration (Euler method) and mask recovery
- **ignore_index=255**: Background pixels in class_map use 255 to exclude from focal loss

## Evaluation

PanNuke uses Panoptic Quality (PQ = DQ x SQ) with IoU > 0.5 matching:

| Metric | Description |
|--------|-------------|
| bPQ | Binary PQ — all nuclei as one class, averaged across tissues |
| mPQ | Multi-class PQ — per-class PQ averaged across 5 nucleus classes |
| Per-class PQ | Individual PQ for Neoplastic, Inflammatory, Connective, Dead, Epithelial |

3-fold cross-validation on PanNuke (7,901 images, 189,744 nuclei, 19 tissue types).

Baseline: HoVer-Net mPQ=0.463, bPQ=0.660.

## Dependencies

- Python >= 3.11
- PyTorch >= 2.0
- timm >= 1.0.27
- cellpose >= 3.1
- datasets (HuggingFace)
- tensorboard
- numpy, scipy, pillow, pyyaml, tqdm, opencv-python-headless

## Citation

```bibtex
@software{lkcellpose2025,
  title = {LK-Cellpose: Large-Kernel ConvNets for Cellpose Panoptic Nuclei Segmentation},
  year = {2025}
}
```
