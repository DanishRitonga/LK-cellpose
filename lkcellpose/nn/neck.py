import torch
import torch.nn as nn
from lkcellpose.nn.modules.conv import Conv, ConvTranspose2x, ConvBlock


class UNetUpBlock(nn.Module):
    """Upsample + concat skip + 2x ConvBlock."""
    def __init__(self, in_ch, skip_ch, out_ch, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.up = ConvTranspose2x(in_ch, out_ch)
        cat_ch = out_ch + (skip_ch if use_skip else 0)
        self.conv = ConvBlock(cat_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if self.use_skip and skip is not None:
            x = torch.cat([x, skip], dim=1)
        elif not self.use_skip:
            pass
        return self.conv(x)


class UNetDecoder(nn.Module):
    """
    Hierarchical U-Net decoder with skip connections.
    
    Takes encoder features [f0, f1, f2, f3] at reductions [4, 8, 16, 32]
    and produces a feature map at full input resolution.
    
    For ConvNeXt V2 Base (encoder_channels=[128, 256, 512, 1024]):
      Stage 3→2: upsample f3 (1024@8x8) → cat f2 (512@16x16) → 512@16x16
      Stage 2→1: upsample (512@16x16) → cat f1 (256@32x32) → 256@32x32
      Stage 1→0: upsample (256@32x32) → cat f0 (128@64x64) → 128@64x64
      Stage 0→up: upsample (128@64x64) → 64@128x128
      Final up: upsample (64@128x128) → 32@256x256
    
    Args:
        encoder_channels: list of 4 ints, feature channels at each encoder stage
        decoder_channels: list of 5 ints, channels for each decoder stage
        grad_checkpoint: bool, use gradient checkpointing on ConvBlocks
    """
    def __init__(self, encoder_channels, decoder_channels, grad_checkpoint=True):
        super().__init__()
        enc = list(reversed(encoder_channels))  # [1024, 512, 256, 128]
        dec = decoder_channels  # [512, 256, 128, 64, 32]
        
        self.up3 = UNetUpBlock(enc[0], enc[1], dec[0], use_skip=True)
        self.up2 = UNetUpBlock(dec[0], enc[2], dec[1], use_skip=True)
        self.up1 = UNetUpBlock(dec[1], enc[3], dec[2], use_skip=True)
        
        self.up0a = UNetUpBlock(dec[2], 0, dec[3], use_skip=False)
        self.up0b = UNetUpBlock(dec[3], 0, dec[4], use_skip=False)
        
        self.grad_checkpoint = grad_checkpoint

    def forward(self, encoder_feats):
        f0, f1, f2, f3 = encoder_feats
        
        x = f3
        x = self._run_block(self.up3, x, f2)
        x = self._run_block(self.up2, x, f1)
        x = self._run_block(self.up1, x, f0)
        x = self._run_block(self.up0a, x)
        x = self._run_block(self.up0b, x)
        
        return x

    def _run_block(self, block, x, skip=None):
        if self.grad_checkpoint and self.training:
            if skip is not None:
                return torch.utils.checkpoint.checkpoint(
                    lambda x_, s_: block(x_, s_), x, skip, use_reentrant=False
                )
            else:
                return torch.utils.checkpoint.checkpoint(
                    lambda x_: block(x_), x, use_reentrant=False
                )
        return block(x, skip)
