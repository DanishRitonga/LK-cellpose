import torch
import torch.nn as nn

from lkcellpose.nn.backbones import build_backbone, VARIANT_CHANNELS
from lkcellpose.nn.neck import UNetDecoder
from lkcellpose.nn.heads import PanopticHead


class BaseModel(nn.Module):
    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, **kwargs):
        return self._predict_once(x, **kwargs)

    def _predict_once(self, x, **kwargs):
        raise NotImplementedError

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)

    def init_criterion(self):
        raise NotImplementedError

    def load(self, weights, verbose=True):
        from lkcellpose.utils.torch_utils import intersect_dicts
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        model_state = self.state_dict()
        csd = intersect_dicts(state, model_state)
        model_state.update(csd)
        self.load_state_dict(model_state, strict=False)
        if verbose:
            from lkcellpose.utils import LOGGER
            LOGGER.info(f"Transferred {len(csd)}/{len(model_state)} items from {weights}")

    def info(self, detailed=False, verbose=True, imgsz=256):
        from lkcellpose.utils.model_info import model_info
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def flops(self, imgsz=256, device="cpu"):
        from lkcellpose.utils.model_info import flops_estimate
        return flops_estimate(self, imgsz=imgsz, device=device)


class PanopticCellposeModel(BaseModel):
    """
    Panoptic Cellpose model: encoder → U-Net decoder → panoptic head.
    
    Output channels: [Y-flow, X-flow, cellprob, class0, ..., class4]
    """

    def __init__(self, backbone="convnextv2_base", pretrained=True,
                 pretrained_tag="fcmae_ft_in22k_in1k",
                 decoder_channels=None, nout=8, n_classes=5,
                 grad_checkpoint=True, panoptic=True, **kwargs):
        super().__init__()
        self.backbone_name = backbone
        self.n_classes = n_classes
        self.panoptic = panoptic
        self.nout = nout if panoptic else 3

        self.encoder, feat_info = build_backbone(
            backbone, pretrained=pretrained, pretrained_tag=pretrained_tag, **kwargs
        )
        encoder_channels = feat_info["channels"]

        if decoder_channels is None:
            decoder_channels = _default_decoder_channels(encoder_channels)

        self.decoder = UNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            grad_checkpoint=grad_checkpoint,
        )
        self.head = PanopticHead(
            in_channels=decoder_channels[-1],
            nout=self.nout,
        )
        self.criterion = None
        self._args = {}

    def _predict_once(self, x):
        feats = self.encoder(x)
        dec = self.decoder(feats)
        out = self.head(dec)
        return out

    def init_criterion(self):
        from lkcellpose.utils.loss import PanopticLoss
        return PanopticLoss(
            flow_weight=self._args.get("flow_weight", 5.0),
            cellprob_weight=self._args.get("cellprob_weight", 1.0),
            class_weight=self._args.get("class_weight", 1.0),
            focal_gamma=self._args.get("focal_gamma", 2.0),
            focal_alpha=self._args.get("focal_alpha", None),
            n_classes=self.n_classes,
            panoptic=self.panoptic,
        )

    def set_model_args(self, **kwargs):
        self._args.update(kwargs)


class CellposeSAMModel(BaseModel):
    """
    Cellpose-SAM baseline: SAM ViT-L encoder + readout head.
    NOT a U-Net — uses simple 1x1 conv + transposed conv readout.

    For panoptic mode, the class head operates on the 256-ch neck
    features (before readout), then class logits are upsampled to
    match the flow/cellprob output resolution.
    """

    def __init__(self, pretrained=True, nout=3, panoptic=False,
                 n_classes=5, **kwargs):
        super().__init__()
        self.panoptic = panoptic
        self.n_classes = n_classes
        self.nout = 3

        from cellpose.vit_sam import Transformer as CPTransformer
        self.net = CPTransformer(backbone="vit_l", nout=self.nout)

        if panoptic:
            self.class_head = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, n_classes, 1),
            )
        else:
            self.class_head = None

        self.criterion = None
        self._args = {}

    def _predict_once(self, x):
        out, _ = self.net(x)
        if self.class_head is not None and self.panoptic:
            neck_feat = self._neck_features(x)
            class_logits = self.class_head(neck_feat)
            class_logits = nn.functional.interpolate(
                class_logits, size=out.shape[2:], mode="bilinear", align_corners=False
            )
            out = torch.cat([out, class_logits], dim=1)
        return out

    def _neck_features(self, x):
        """Run encoder + neck, return 256-ch feature map before readout."""
        import torch.nn.functional as F
        net = self.net
        x = net.encoder.patch_embed(x)
        if net.encoder.pos_embed is not None:
            x = x + net.encoder.pos_embed
        for blk in net.encoder.blocks:
            x = blk(x)
        x = net.encoder.neck(x.permute(0, 3, 1, 2))
        return x

    def init_criterion(self):
        from lkcellpose.utils.loss import PanopticLoss
        return PanopticLoss(
            flow_weight=self._args.get("flow_weight", 5.0),
            cellprob_weight=self._args.get("cellprob_weight", 1.0),
            class_weight=self._args.get("class_weight", 1.0),
            focal_gamma=self._args.get("focal_gamma", 2.0),
            focal_alpha=self._args.get("focal_alpha", None),
            n_classes=self.n_classes,
            panoptic=self.panoptic,
        )

    def set_model_args(self, **kwargs):
        self._args.update(kwargs)


def _default_decoder_channels(encoder_channels):
    enc = list(reversed(encoder_channels))
    dec = [min(enc[0], 512)]
    for i in range(1, len(enc)):
        dec.append(min(enc[i] // 2, dec[-1]))
    dec.extend([dec[-1] // 2, dec[-1] // 4])
    while len(dec) < 5:
        dec.append(dec[-1] // 2)
    return dec[:5]
