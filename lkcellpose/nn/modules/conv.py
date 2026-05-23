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
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch)
        self.conv2 = Conv(out_ch, out_ch)
        self.shortcut = (
            nn.Identity() if in_ch == out_ch else Conv(in_ch, out_ch, k=1, s=1)
        )

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.shortcut(x)
