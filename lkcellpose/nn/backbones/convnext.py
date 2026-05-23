import timm
import torch.nn as nn


CONVNEXT_V2_VARIANTS = {
    "convnextv2_atto": [40, 80, 160, 320],
    "convnextv2_femto": [48, 96, 192, 384],
    "convnextv2_pico": [64, 128, 256, 512],
    "convnextv2_nano": [80, 160, 320, 640],
    "convnextv2_tiny": [96, 192, 384, 768],
    "convnextv2_small": [96, 192, 384, 768],
    "convnextv2_base": [128, 256, 512, 1024],
    "convnextv2_large": [192, 384, 768, 1536],
    "convnextv2_huge": [352, 704, 1408, 2816],
}


def convnextv2_backbone(name: str = "convnextv2_base", pretrained: bool = True,
                        pretrained_tag: str | None = "fcmae_ft_in22k_in1k", **kwargs):
    """Create ConvNeXt V2 backbone with features_only mode.

    Returns:
        model: nn.Module that returns list of 4 feature maps when called
        feature_info: dict with "channels" (list of ints) and "reductions" (list of ints)
    """
    checkpoint_name = pretrained_tag if pretrained else None
    model = timm.create_model(
        name,
        pretrained=pretrained,
        checkpoint_name=checkpoint_name,
        features_only=True,
        **kwargs,
    )
    feature_info = {
        "channels": [f["num_chs"] for f in model.feature_info],
        "reductions": [f["reduction"] for f in model.feature_info],
    }
    return model, feature_info
