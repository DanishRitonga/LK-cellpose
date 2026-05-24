import torch
import torch.nn as nn
import torch.nn.functional as F


class PanopticLoss(nn.Module):
    """
    Combined loss for panoptic Cellpose: flow MSE + cellprob BCE + class focal loss.
    
    Matches Cellpose v4 training loss exactly:
      - Flow loss:  MSE(pred_flow, 5 * gt_flow) / 2    (GT flows scaled by 5, divided by 2)
      - Cellprob loss: BCEWithLogitsLoss (no pos_weight, no class weighting)
      - Class loss: focal loss (panoptic mode only)
    
    Output tensor layout: (B, 8, H, W)
      [0] = Y-flow
      [1] = X-flow
      [2] = cell probability (logits)
      [3:8] = class logits (5 classes)
    
    Targets:
      flows[0:2] = ground-truth Y/X flow (unit vectors, magnitude ~1 on fg, 0 on bg)
      flows[2] = cell probability (0 or 1)
      class_map = per-pixel class label (0-4), 255=ignore
    """

    # Cellpose v4 scales GT flows by 5 in the loss so the network learns
    # to output magnitude-~5 flows, matching what compute_masks expects
    # (it divides by 5 internally before Euler integration).
    FLOW_SCALE = 5.0

    def __init__(self, cellprob_weight=1.0, class_weight=1.0,
                 focal_gamma=2.0, focal_alpha=None, n_classes=5, panoptic=True):
        super().__init__()
        self.cellprob_weight = cellprob_weight
        self.class_weight = class_weight
        self.focal_gamma = focal_gamma
        self.panoptic = panoptic
        self.n_classes = n_classes

        if focal_alpha is not None and focal_alpha != "auto":
            self.register_buffer("focal_alpha", torch.tensor(focal_alpha, dtype=torch.float32))
        else:
            self.focal_alpha = None

    def forward(self, preds, batch):
        # --- Flow loss (matching cellpose v4 exactly) ---
        # Cellpose: veci = 5. * lbl[:,-2:];  loss = MSE(y[:,-3:-1], veci) / 2.
        # MSE over ALL pixels (including background, which has zero-flow targets)
        # to force the network to predict zero flow for background pixels.
        pred_flow = preds[:, :2]
        gt_flow_scaled = batch["flows"][:, :2] * self.FLOW_SCALE
        flow_loss = F.mse_loss(pred_flow, gt_flow_scaled, reduction="mean") / 2.0

        # --- Cellprob loss (matching cellpose v4: no pos_weight) ---
        pred_cellprob = preds[:, 2]
        gt_cellprob = batch["cellprob"]
        cellprob_loss = F.binary_cross_entropy_with_logits(pred_cellprob, gt_cellprob)

        total_loss = flow_loss + self.cellprob_weight * cellprob_loss
        loss_items = {
            "flow_loss": flow_loss.detach(),
            "cellprob_loss": cellprob_loss.detach(),
        }

        if self.panoptic and preds.shape[1] > 3:
            pred_class = preds[:, 3:]
            gt_class = batch["class_map"]

            class_loss = self._focal_loss(pred_class, gt_class)
            total_loss = total_loss + self.class_weight * class_loss
            loss_items["class_loss"] = class_loss.detach()

        loss_items["total_loss"] = total_loss.detach()
        return total_loss, loss_items

    def _focal_loss(self, logits, targets):
        B, C, H, W = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)

        valid = targets_flat != 255
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        logits_valid = logits_flat[valid]
        targets_valid = targets_flat[valid]

        ce = F.cross_entropy(logits_valid, targets_valid.long(), reduction='none')
        pt = torch.exp(-ce)

        focal_weight = (1 - pt) ** self.focal_gamma

        if self.focal_alpha is not None:
            alpha_t = self.focal_alpha[targets_valid]
            focal_loss = alpha_t * focal_weight * ce
        else:
            focal_loss = focal_weight * ce

        return focal_loss.mean()

    @staticmethod
    def compute_alpha_from_counts(class_counts, n_classes=5):
        if isinstance(class_counts, dict):
            counts = [class_counts.get(i, 1) for i in range(n_classes)]
        else:
            counts = list(class_counts)

        inv_sqrt = [1.0 / max(c, 1) ** 0.5 for c in counts]
        total = sum(inv_sqrt)
        alpha = [a * n_classes / total for a in inv_sqrt]
        return alpha
