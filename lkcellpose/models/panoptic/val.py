import torch
import numpy as np
from lkcellpose.engine.validator import BaseValidator
from lkcellpose.utils.metrics import PanopticQuality
from lkcellpose.utils import LOGGER, NUCLEUS_CLASSES
from lkcellpose.dynamics import compute_masks
from lkcellpose.data.pannuke import PanNukeDataset


class PanopticValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.pq_metric = None
        self.n_classes = 5

    def init_metrics(self, model):
        n_classes = self.args.get("n_classes", 5) if hasattr(self.args, "get") else 5
        self.n_classes = n_classes
        self.pq_metric = PanopticQuality(n_classes=n_classes)
        self.loss_sum = 0.0
        self.n_batches = 0

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        return batch

    def update_metrics(self, preds, batch):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=False):
            preds_float = preds.float()

        for b in range(preds.shape[0]):
            p = preds_float[b].cpu().numpy()
            flow_y, flow_x = p[0], p[1]
            cellprob = 1.0 / (1.0 + np.exp(-p[2]))
            cellprob_threshold = self.args.get("cellprob_threshold", 0.0) if hasattr(self.args, "get") else 0.0
            flow_threshold = self.args.get("flow_threshold", 0.4) if hasattr(self.args, "get") else 0.4
            min_size = self.args.get("min_size", 15) if hasattr(self.args, "get") else 15

            flows = np.stack([flow_y, flow_x], axis=0)
            pred_labels = compute_masks(flows, cellprob,
                                        cellprob_threshold=cellprob_threshold,
                                        flow_threshold=flow_threshold,
                                        min_size=min_size,
                                        device=self.device)

            gt_labels = batch["labels"][b].cpu().numpy()
            n_pred = len(np.unique(pred_labels[pred_labels > 0]))
            n_gt = len(np.unique(gt_labels[gt_labels > 0]))
            if not hasattr(self, "_n_pred_total"):
                self._n_pred_total = 0
                self._n_gt_total = 0
                self._n_samples = 0
                self._flow_mag_sum = 0.0
                self._cellprob_pos_sum = 0
            self._n_pred_total += n_pred
            self._n_gt_total += n_gt
            self._n_samples += 1
            flow_mag = np.sqrt(p[0]**2 + p[1]**2)
            self._flow_mag_sum += flow_mag.mean()
            self._cellprob_pos_sum += (cellprob > 0.5).sum()

            if p.shape[0] > 3:
                pred_class_map = np.argmax(p[3:], axis=0)
            else:
                pred_class_map = None

            gt_class_map = batch["class_map"][b].cpu().numpy()
            gt_class_map_clean = gt_class_map.copy().astype(np.int8)
            gt_class_map_clean[gt_class_map_clean == 255] = -1

            self.pq_metric.update(
                pred_labels, gt_labels,
                gt_class_map=gt_class_map_clean if gt_class_map_clean is not None else None,
                pred_class_map=pred_class_map,
            )

        if self.training and hasattr(self, "_loss_fn"):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=False):
                loss, _ = self.model.loss(batch, preds.float())
            self.loss_sum += loss.item()
            self.n_batches += 1

    def get_stats(self):
        if self.pq_metric is not None:
            return self.pq_metric.compute()
        return {"bPQ": 0.0, "mPQ": 0.0, "per_class_pq": {}}

    def get_dataloader(self):
        folds = self.args.get("val_folds", [2]) if hasattr(self.args, "get") else [2]
        batch_size = self.args.get("batch", 8) if hasattr(self.args, "get") else 8
        workers = self.args.get("workers", 8) if hasattr(self.args, "get") else 8
        datasets = []
        for fold in folds:
            ds = PanNukeDataset(
                fold=fold,
                split="val",
                data_dir=self.args.get("data_dir") if hasattr(self.args, "get") else None,
                augment=None,
                n_classes=self.args.get("n_classes", 5) if hasattr(self.args, "get") else 5,
                min_masks=self.args.get("min_masks", 5) if hasattr(self.args, "get") else 5,
                device=None,
            )
            datasets.append(ds)
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            from torch.utils.data import ConcatDataset
            dataset = ConcatDataset(datasets)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )

    def print_results(self, results):
        LOGGER.info(f"Validation: bPQ={results.get('bPQ', 0):.4f}, mPQ={results.get('mPQ', 0):.4f}")
        per_class = results.get("per_class_pq", {})
        for cls_id, pq in per_class.items():
            name = NUCLEUS_CLASSES[cls_id] if cls_id < len(NUCLEUS_CLASSES) else f"Class{cls_id}"
            LOGGER.info(f"  {name}: PQ={pq:.4f}")
        if hasattr(self, "_n_samples") and self._n_samples > 0:
            avg_pred = self._n_pred_total / self._n_samples
            avg_gt = self._n_gt_total / self._n_samples
            LOGGER.info(f"  Avg instances/sample: pred={avg_pred:.1f}, gt={avg_gt:.1f}")
            avg_flow_mag = self._flow_mag_sum / self._n_samples
            avg_cellprob_pos = self._cellprob_pos_sum / (self._n_samples * 256 * 256) * 100
            LOGGER.info(f"  Avg flow magnitude: {avg_flow_mag:.3f}, cellprob>0.5: {avg_cellprob_pos:.1f}%")
