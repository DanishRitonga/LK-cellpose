from lkcellpose.nn.backbones.convnext import convnextv2_backbone, CONVNEXT_V2_VARIANTS
from lkcellpose.nn.backbones.fastvit import fastvit_backbone, FASTVIT_VARIANTS
from lkcellpose.nn.backbones.unireplknet import unireplknet_backbone, UNIREPLKNET_VARIANTS
from lkcellpose.nn.backbones.cellpose_unet import cellpose_unet_backbone, CELLPOSE_UNET_VARIANTS

BACKBONE_REGISTRY = {}
for _v in CONVNEXT_V2_VARIANTS:
    BACKBONE_REGISTRY[_v] = convnextv2_backbone
for _v in FASTVIT_VARIANTS:
    BACKBONE_REGISTRY[_v] = fastvit_backbone
for _v in UNIREPLKNET_VARIANTS:
    BACKBONE_REGISTRY[_v] = unireplknet_backbone
for _v in CELLPOSE_UNET_VARIANTS:
    BACKBONE_REGISTRY[_v] = cellpose_unet_backbone
BACKBONE_REGISTRY["cellpose_sam"] = None

VARIANT_CHANNELS = {}
VARIANT_CHANNELS.update(CONVNEXT_V2_VARIANTS)
VARIANT_CHANNELS.update(FASTVIT_VARIANTS)
VARIANT_CHANNELS.update(UNIREPLKNET_VARIANTS)
VARIANT_CHANNELS.update(CELLPOSE_UNET_VARIANTS)
VARIANT_CHANNELS["cellpose_sam"] = [256]


def build_backbone(name, pretrained=True, pretrained_tag=None, **kwargs):
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. Available: {sorted(BACKBONE_REGISTRY.keys())}"
        )
    builder = BACKBONE_REGISTRY[name]
    if builder is None:
        raise NotImplementedError(f"Backbone '{name}' is not yet implemented")
    return builder(name, pretrained=pretrained, pretrained_tag=pretrained_tag, **kwargs)
