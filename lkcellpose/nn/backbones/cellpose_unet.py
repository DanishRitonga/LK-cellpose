import torch
import torch.nn as nn
import torch.nn.functional as F

from lkcellpose.nn.modules.conv import PreActConv

_BN_KW = dict(eps=1e-5, momentum=0.05)


class ResDownBlock(nn.Module):
    """Encoder residual block matching Cellpose's resdown.

    2 residual pairs with pre-activation (BN->ReLU->Conv):
      pair 1: proj(x) + conv1(conv0(x))
      pair 2: x       + conv3(conv2(x))
    """

    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.BatchNorm2d(in_ch, **_BN_KW),
            nn.Conv2d(in_ch, out_ch, 1),
        )
        self.conv0 = PreActConv(in_ch, out_ch, k)
        self.conv1 = PreActConv(out_ch, out_ch, k)
        self.conv2 = PreActConv(out_ch, out_ch, k)
        self.conv3 = PreActConv(out_ch, out_ch, k)

    def forward(self, x):
        x = self.proj(x) + self.conv1(self.conv0(x))
        x = x + self.conv3(self.conv2(x))
        return x


class CellposeUNetEncoder(nn.Module):
    """Encoder from the original Cellpose Residual U-Net (v1-v3).

    Faithful re-implementation using Cellpose's resdown blocks
    (pre-activation BN->ReLU->Conv, 4 convs per stage, 2 residual pairs)
    adapted for our framework with a stride-4 stem to produce features
    at reductions [4, 8, 16, 32], matching ConvNeXt/UniRepLKNet output
    spacing so the same UNetDecoder can be attached for fair comparison.

    Reference: Stringer et al., Nature Methods 2021.
    """

    def __init__(self, in_chans=3, nbase=None):
        super().__init__()
        if nbase is None:
            nbase = [64, 128, 256, 512]
        self.nbase = nbase

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, nbase[0], 4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(nbase[0], **_BN_KW),
            nn.ReLU(inplace=True),
        )

        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(4):
            in_ch = nbase[i - 1] if i > 0 else nbase[0]
            out_ch = nbase[i]
            self.stages.append(ResDownBlock(in_ch, out_ch))
            self.norms.append(nn.BatchNorm2d(out_ch, **_BN_KW))

    def forward(self, x):
        x = self.stem(x)
        features = []
        for i in range(4):
            if i > 0:
                x = F.max_pool2d(x, 2, 2)
            x = self.stages[i](x)
            features.append(self.norms[i](x))
        return features


CELLPOSE_UNET_VARIANTS = {
    "cellpose_unet": [64, 128, 256, 512],
}


def cellpose_unet_backbone(name="cellpose_unet", pretrained=True,
                           pretrained_tag=None, **kwargs):
    """Create Cellpose Residual U-Net encoder as a backbone.

    Returns:
        model: nn.Module whose forward() returns list of 4 feature maps
        feature_info: dict with "channels" and "reductions" keys
    """
    model = CellposeUNetEncoder(**kwargs)
    feature_info = {
        "channels": [64, 128, 256, 512],
        "reductions": [4, 8, 16, 32],
    }
    return model, feature_info
