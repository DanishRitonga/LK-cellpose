import torch.nn as nn


class PanopticHead(nn.Module):
    """
    Output head for panoptic nuclei segmentation.
    
    Input: decoder features (B, C, H, W)
    Output: (B, nout, H, W) where nout = 3 + n_classes
    
    Channels: [Y-flow, X-flow, cellprob, class0_logit, ..., classN_logit]
    """
    def __init__(self, in_channels=32, nout=8):
        super().__init__()
        self.head = nn.Conv2d(in_channels, nout, kernel_size=1, bias=True)
        self.nout = nout

    def forward(self, x):
        return self.head(x)
