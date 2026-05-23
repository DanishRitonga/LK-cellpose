import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 * d
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, ch, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, k, s, k // 2, groups=ch, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvTranspose2x(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(in_ch, out_ch, 2, 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv_t(x)))


class DropBlock(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.drop = nn.Dropout2d(p)

    def forward(self, x):
        return self.drop(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, style_ch=None):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch)
        self.conv2 = Conv(out_ch, out_ch)
        self.shortcut = (
            nn.Identity() if in_ch == out_ch else Conv(in_ch, out_ch, k=1, s=1)
        )
        self.style_proj = nn.Linear(style_ch, out_ch) if style_ch else None

    def forward(self, x, style=None):
        h = self.conv1(x)
        if self.style_proj is not None and style is not None:
            s = self.style_proj(style).unsqueeze(-1).unsqueeze(-1)
            h = h + s
        h = self.conv2(h)
        return h + self.shortcut(x)


_BN_KW = dict(eps=1e-5, momentum=0.05)


class PreActConv(nn.Module):
    """BN -> ReLU -> Conv (pre-activation, matching Cellpose's batchconv)."""

    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch, **_BN_KW)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k // 2)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class StyleConv(nn.Module):
    """Add skip + style, then BN -> ReLU -> Conv (matching Cellpose's batchconvstyle)."""

    def __init__(self, in_ch, out_ch, style_ch, k=3):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch, **_BN_KW)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k // 2)
        self.style_proj = nn.Linear(style_ch, in_ch)

    def forward(self, x, style, skip=None):
        if skip is not None:
            x = x + skip
        s = self.style_proj(style).unsqueeze(-1).unsqueeze(-1)
        x = x + s
        return self.conv(self.act(self.bn(x)))


class ResUpBlock(nn.Module):
    """Decoder residual block matching Cellpose's resup.

    2 residual pairs with style conditioning:
      pair 1: proj(x) + conv1(style, conv0(x), skip=y)
      pair 2: x    + conv3(style, conv2(style, x))
    """

    def __init__(self, in_ch, out_ch, style_ch, k=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.BatchNorm2d(in_ch, **_BN_KW),
            nn.Conv2d(in_ch, out_ch, 1),
        )
        self.conv0 = PreActConv(in_ch, out_ch, k)
        self.conv1 = StyleConv(out_ch, out_ch, style_ch, k)
        self.conv2 = StyleConv(out_ch, out_ch, style_ch, k)
        self.conv3 = StyleConv(out_ch, out_ch, style_ch, k)

    def forward(self, x, style, skip=None):
        h = self.conv1(self.conv0(x), style, skip=skip)
        x = self.proj(x) + h
        h = self.conv2(x, style)
        h = self.conv3(h, style)
        x = x + h
        return x
