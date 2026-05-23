import torch
import torch.nn as nn
import torch.nn.functional as F
from lkcellpose.nn.modules.conv import ResUpBlock


class UNetUpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_skip=True, style_ch=None):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.use_skip = use_skip
        if use_skip and skip_ch is not None and skip_ch > 0:
            self.skip_proj = (
                nn.Identity() if skip_ch == out_ch
                else nn.Conv2d(skip_ch, out_ch, 1, bias=False)
            )
        else:
            self.skip_proj = None
        self.resup = ResUpBlock(in_ch, out_ch, style_ch)

    def forward(self, x, skip=None, style=None):
        x = self.upsample(x)
        skip_p = (
            self.skip_proj(skip)
            if self.skip_proj is not None and skip is not None
            else None
        )
        return self.resup(x, style, skip=skip_p)


class UNetDecoder(nn.Module):
    """
    Hierarchical U-Net decoder matching original Cellpose's design.

    Key design choices (matching Cellpose v3 resnet_torch.py):
    - Skip connections: element-wise addition (concatenation=False)
    - Upsampling: nearest-neighbor (no learned transposed conv)
    - Style vector: GAP + L2-norm from deepest encoder features,
      broadcast and added before BN->ReLU->Conv in each decoder conv
      (FiLM-like, matching Cellpose's make_style + batchconvstyle)
    - Pre-activation: BN->ReLU->Conv (not Conv->BN->Act)
    - BN momentum: 0.05 (not PyTorch default 0.1)
    - 2 residual pairs per block (4 convs, matching resup)

    Takes encoder features [f0, f1, f2, f3] at reductions [4, 8, 16, 32]
    and produces a feature map at full input resolution.

    Args:
        encoder_channels: list of 4 ints, feature channels at each encoder stage
        decoder_channels: list of 5 ints, channels for each decoder stage
        grad_checkpoint: bool, use gradient checkpointing
        style_channels: int or None, dimension of style vector.
            If None, defaults to encoder_channels[-1] (deepest stage).
            Set to 0 to disable style conditioning.
    """

    def __init__(self, encoder_channels, decoder_channels, grad_checkpoint=True,
                 style_channels=None):
        super().__init__()
        enc = list(reversed(encoder_channels))  # [1024, 512, 256, 128]
        dec = decoder_channels  # [512, 256, 128, 64, 32]

        if style_channels is None:
            style_channels = encoder_channels[-1]
        self.use_style = style_channels > 0
        style_ch = style_channels if self.use_style else 0

        self.up3 = UNetUpBlock(enc[0], enc[1], dec[0], use_skip=True, style_ch=style_ch)
        self.up2 = UNetUpBlock(dec[0], enc[2], dec[1], use_skip=True, style_ch=style_ch)
        self.up1 = UNetUpBlock(dec[1], enc[3], dec[2], use_skip=True, style_ch=style_ch)

        self.up0a = UNetUpBlock(dec[2], 0, dec[3], use_skip=False, style_ch=style_ch)
        self.up0b = UNetUpBlock(dec[3], 0, dec[4], use_skip=False, style_ch=style_ch)

        self.grad_checkpoint = grad_checkpoint

    @staticmethod
    def _compute_style(x):
        """L2-normalized global average pooling (Cellpose's make_style)."""
        s = F.adaptive_avg_pool2d(x, 1).flatten(1)
        s = s / (s.pow(2).sum(1, keepdim=True).pow(0.5) + 1e-8)
        return s

    def forward(self, encoder_feats):
        f0, f1, f2, f3 = encoder_feats

        style = self._compute_style(f3) if self.use_style else None

        x = f3
        x = self._run_block(self.up3, x, f2, style)
        x = self._run_block(self.up2, x, f1, style)
        x = self._run_block(self.up1, x, f0, style)
        x = self._run_block(self.up0a, x, style=style)
        x = self._run_block(self.up0b, x, style=style)

        return x

    def _run_block(self, block, x, skip=None, style=None):
        if self.grad_checkpoint and self.training:
            if skip is not None:
                return torch.utils.checkpoint.checkpoint(
                    lambda x_, s_, st_: block(x_, s_, st_), x, skip, style,
                    use_reentrant=False,
                )
            else:
                return torch.utils.checkpoint.checkpoint(
                    lambda x_, st_: block(x_, style=st_), x, style,
                    use_reentrant=False,
                )
        return block(x, skip, style=style)
