import torch
import torch.nn as nn

from lkcellpose.utils import LOGGER


def model_info(model, detailed=False, verbose=True, imgsz=256):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    n_layers = sum(1 for _ in model.modules())
    backbone_name = getattr(model, "backbone_name", None)

    info = {
        "total_params": n_p,
        "trainable_params": n_g,
        "frozen_params": n_p - n_g,
        "layers": n_layers,
        "backbone_name": backbone_name,
    }

    if detailed:
        info.update(parameter_count(model))

    if verbose:
        tag = f"({backbone_name})" if backbone_name else ""
        LOGGER.info(
            f"Model{tag}: params={n_p:,}, grads={n_g:,}, "
            f"frozen={n_p - n_g:,}, layers={n_layers}"
        )
        if detailed:
            pc = parameter_count(model)
            for k in ("backbone", "decoder", "head"):
                LOGGER.info(f"  {k}: {pc[k]:,}")

    return info


def flops_estimate(model, imgsz=256, device="cpu"):
    hooks = []
    flops = {}

    def _conv_hook(mod, inp, out):
        cout = mod.out_channels
        cin = mod.in_channels // mod.groups
        kh, kw = mod.kernel_size
        oh, ow = out.shape[2], out.shape[3]
        ops = 2 * cout * cin * kh * kw * oh * ow
        flops[mod._flops_name] = flops.get(mod._flops_name, 0) + ops

    def _linear_hook(mod, inp, out):
        ops = 2 * mod.in_features * mod.out_features
        flops[mod._flops_name] = flops.get(mod._flops_name, 0) + ops

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            mod._flops_name = name
            hooks.append(mod.register_forward_hook(_conv_hook))
        elif isinstance(mod, nn.Linear):
            mod._flops_name = name
            hooks.append(mod.register_forward_hook(_linear_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
            model(dummy)
    except Exception:
        LOGGER.warning("FLOPs estimate: forward pass failed")
    finally:
        model.train(was_training)

    for h in hooks:
        h.remove()
    for name, mod in model.named_modules():
        if hasattr(mod, "_flops_name"):
            del mod._flops_name

    total = sum(flops.values())
    gflops = total / 1e9

    # group by top-level module
    grouped = {}
    for name, ops in flops.items():
        top = name.split(".")[0] if "." in name else name
        grouped[top] = grouped.get(top, 0) + ops

    return {
        "gflops": gflops,
        "total_flops": total,
        "by_module": {k: v / 1e9 for k, v in grouped.items()},
    }


def parameter_count(model):
    total = sum(x.numel() for x in model.parameters())
    trainable = sum(x.numel() for x in model.parameters() if x.requires_grad)

    by_module = {}
    for name, param in model.named_parameters():
        top = name.split(".")[0] if "." in name else name
        by_module[top] = by_module.get(top, 0) + param.numel()

    backbone = by_module.get("encoder", by_module.get("net", 0))
    head = by_module.get("head", by_module.get("class_head", 0))

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "backbone": backbone,
        "decoder": by_module.get("decoder", 0),
        "head": head,
    }
