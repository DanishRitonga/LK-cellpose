import argparse
import sys
import warnings

warnings.filterwarnings("ignore", message="Importing from timm.models.* is deprecated")

from lkcellpose import LKCellposeModel, __version__


def _build_train_parser(subparsers):
    p = subparsers.add_parser("train", help="Train a model")
    p.add_argument("--backbone", default="convnextv2_base")
    p.add_argument("--config", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="exp")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--pretrained-tag", default="fcmae_ft_in22k_in1k")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--train-folds", type=int, nargs="+", default=[1])
    p.add_argument("--val-folds", type=int, nargs="+", default=[2])
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--cache-flows", action="store_true", default=True)
    p.add_argument("--no-cache-flows", dest="cache_flows", action="store_false")
    p.add_argument("--grad-checkpoint", action="store_true", default=True)
    p.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    p.add_argument("--panoptic", action="store_true", default=True)
    p.add_argument("--no-panoptic", dest="panoptic", action="store_false")
    p.add_argument("--ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="ema", action="store_false")
    p.add_argument("--freeze-encoder", action="store_true", default=False)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    return p


def _build_predict_parser(subparsers):
    p = subparsers.add_parser("predict", help="Run inference")
    p.add_argument("source", help="Image path or directory")
    p.add_argument("--backbone", default="convnextv2_base")
    p.add_argument("--weights", default=None, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--cellprob-threshold", type=float, default=0.0)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--min-size", type=int, default=15)
    p.add_argument("--tile-overlap", type=float, default=0.1)
    p.add_argument("--save-dir", default=None)
    return p


def _build_val_parser(subparsers):
    p = subparsers.add_parser("val", help="Validate a model")
    p.add_argument("--backbone", default="convnextv2_base")
    p.add_argument("--weights", default=None, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--val-folds", type=int, nargs="+", default=[2])
    p.add_argument("--batch", type=int, default=16)
    return p


def _build_info_parser(subparsers):
    p = subparsers.add_parser("info", help="Print model info")
    p.add_argument("--backbone", default="convnextv2_base")
    p.add_argument("--detailed", action="store_true", default=False)
    p.add_argument("--flops", action="store_true", default=False)
    p.add_argument("--device", default="cpu")
    return p


def _run_train(args):
    overrides = {
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "pretrained": args.pretrained,
        "pretrained_tag": args.pretrained_tag,
        "train_folds": args.train_folds,
        "val_folds": args.val_folds,
        "augment": args.augment,
        "cache_flows": args.cache_flows,
        "grad_checkpoint": args.grad_checkpoint,
        "panoptic": args.panoptic,
        "ema": args.ema,
        "freeze_encoder": args.freeze_encoder,
    }
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.lr0 is not None:
        overrides["lr0"] = args.lr0
    if args.weight_decay is not None:
        overrides["weight_decay"] = args.weight_decay
    if args.warmup_epochs is not None:
        overrides["warmup_epochs"] = args.warmup_epochs
    if args.grad_accum_steps is not None:
        overrides["grad_accum_steps"] = args.grad_accum_steps
    if args.workers is not None:
        overrides["workers"] = args.workers
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.patience is not None:
        overrides["patience"] = args.patience

    model = LKCellposeModel(backbone=args.backbone)
    model.train(cfg=args.config or "default", **overrides)


def _run_predict(args):
    import torch
    from lkcellpose.nn.tasks import PanopticCellposeModel
    from lkcellpose.utils.torch_utils import select_device

    device = select_device(args.device)
    nn_model = PanopticCellposeModel(
        backbone=args.backbone,
        pretrained=False,
        panoptic=True,
    )
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model_state = nn_model.state_dict()
    from lkcellpose.utils.torch_utils import intersect_dicts
    csd = intersect_dicts(state, model_state)
    model_state.update(csd)
    nn_model.load_state_dict(model_state, strict=False)
    nn_model = nn_model.to(device).eval()

    model = LKCellposeModel(backbone=args.backbone)
    model.model = nn_model
    overrides = {
        "cellprob_threshold": args.cellprob_threshold,
        "flow_threshold": args.flow_threshold,
        "min_size": args.min_size,
        "tile_overlap": args.tile_overlap,
    }
    if args.save_dir:
        overrides["save_dir"] = args.save_dir
    results = model.predict(args.source, **overrides)

    for i, r in enumerate(results):
        if args.save_dir:
            from pathlib import Path
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            r.save(str(Path(args.save_dir) / f"result_{i:04d}.png"))
        print(f"Image {i}: {r.n_instances} instances detected")

    print(f"Predicted {len(results)} images")


def _run_val(args):
    import torch
    from lkcellpose.nn.tasks import PanopticCellposeModel
    from lkcellpose.utils.torch_utils import select_device

    device = select_device(args.device)
    nn_model = PanopticCellposeModel(
        backbone=args.backbone,
        pretrained=False,
        panoptic=True,
    )
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model_state = nn_model.state_dict()
    from lkcellpose.utils.torch_utils import intersect_dicts
    csd = intersect_dicts(state, model_state)
    model_state.update(csd)
    nn_model.load_state_dict(model_state, strict=False)
    nn_model = nn_model.to(device).eval()

    model = LKCellposeModel(backbone=args.backbone)
    model.model = nn_model
    overrides = {
        "device": args.device,
        "val_folds": args.val_folds,
    }
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.batch:
        overrides["batch"] = args.batch
    model.val(**overrides)


def _run_info(args):
    from lkcellpose.nn.tasks import PanopticCellposeModel
    from lkcellpose.utils import LOGGER

    model = PanopticCellposeModel(
        backbone=args.backbone,
        pretrained=False,
        panoptic=True,
    )
    info = model.info(detailed=args.detailed, verbose=True)
    print(f"\nBackbone: {args.backbone}")
    print(f"Total params: {info['total_params']:,}")
    print(f"Trainable params: {info['trainable_params']:,}")
    print(f"Frozen params: {info['frozen_params']:,}")
    print(f"Layers: {info['layers']}")

    if args.detailed:
        print(f"\n  Backbone: {info.get('backbone', 'N/A'):,}")
        print(f"  Decoder: {info.get('decoder', 'N/A'):,}")
        print(f"  Head: {info.get('head', 'N/A'):,}")

    if args.flops:
        from lkcellpose.utils.torch_utils import select_device
        device = select_device(args.device)
        flops_info = model.flops(device=str(device))
        print(f"\nGFLOPs (256x256): {flops_info['gflops']:.2f}")
        if flops_info["by_module"]:
            print("By module:")
            for mod, gf in sorted(flops_info["by_module"].items()):
                print(f"  {mod}: {gf:.2f} GFLOPs")


def main():
    parser = argparse.ArgumentParser(
        description="LK-Cellpose: Large-kernel Cellpose for nuclei panoptic segmentation"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    _build_train_parser(subparsers)
    _build_predict_parser(subparsers)
    _build_val_parser(subparsers)
    _build_info_parser(subparsers)

    args = parser.parse_args()

    if args.mode == "train":
        _run_train(args)
    elif args.mode == "predict":
        _run_predict(args)
    elif args.mode == "val":
        _run_val(args)
    elif args.mode == "info":
        _run_info(args)


if __name__ == "__main__":
    main()
