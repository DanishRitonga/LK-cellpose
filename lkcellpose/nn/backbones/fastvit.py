import timm
import torch.nn as nn


FASTVIT_VARIANTS = {
    "fastvit_t8": [48, 96, 192, 384],
    "fastvit_t12": [64, 128, 256, 512],
    "fastvit_s12": [64, 128, 256, 512],
    "fastvit_sa12": [64, 128, 256, 512],
    "fastvit_sa24": [64, 128, 256, 512],
    "fastvit_sa36": [64, 128, 256, 512],
    "fastvit_ma36": [76, 152, 304, 608],
}


def fastvit_backbone(name: str = "fastvit_sa36", pretrained: bool = True,
                     pretrained_tag: str | None = "apple_in1k", **kwargs):
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
