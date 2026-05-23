import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class CellposeUNetEncoder(nn.Module):
    """Encoder from the original Cellpose Residual U-Net (v1-v3).

    Re-implementation of the residual-block encoder adapted for our
    framework: uses a stride-4 stem to produce features at reductions
    [4, 8, 16, 32], matching ConvNeXt/UniRepLKNet output spacing so
    the same UNetDecoder can be attached for fair comparison.

    The residual blocks (2x Conv+BN+ReLU with skip connection) are
    faithful to the original Cellpose architecture.

    Reference: Stringer et al., Nature Methods 2021.
    """

    def __init__(self, in_chans=3, nbase=None, n_blocks_per_stage=2):
        super().__init__()
        if nbase is None:
            nbase = [64, 128, 256, 512]
        self.nbase = nbase

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, nbase[0], 4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(nbase[0]),
            nn.ReLU(inplace=True),
        )

        self.transitions = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(4):
            in_ch = nbase[i - 1] if i > 0 else nbase[0]
            out_ch = nbase[i]
            if in_ch != out_ch:
                self.transitions.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ))
            else:
                self.transitions.append(nn.Identity())

            stage = nn.Sequential(
                *[ResidualBlock(out_ch) for _ in range(n_blocks_per_stage)]
            )
            self.stages.append(stage)
            self.norms.append(nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.stem(x)
        features = []
        for i in range(4):
            if i > 0:
                x = F.max_pool2d(x, 2, 2)
            x = self.transitions[i](x)
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
