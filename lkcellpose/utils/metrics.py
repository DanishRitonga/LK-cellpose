import numpy as np
from collections import defaultdict


class PanopticQuality:
    """
    Panoptic Quality (PQ) computation for nuclei segmentation.
    
    PQ = DQ × SQ where:
      DQ (Detection Quality) = TP / (TP + 0.5*FP + 0.5*FN)
      SQ (Segmentation Quality) = mean IoU of matched pairs
    
    Matching: IoU > 0.5
    """

    def __init__(self, n_classes=5, iou_threshold=0.5):
        self.n_classes = n_classes
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)
        self.iou_sum = defaultdict(float)

    def update(self, pred_labels, gt_labels, gt_class_map=None, pred_class_map=None):
        pred_ids = np.unique(pred_labels[pred_labels > 0])
        gt_ids = np.unique(gt_labels[gt_labels > 0])

        self._update_binary(pred_labels, gt_labels, pred_ids, gt_ids)
        if gt_class_map is not None and pred_class_map is not None:
            self._update_per_class(pred_labels, gt_labels, pred_ids, gt_ids,
                                   pred_class_map, gt_class_map)

    def _update_binary(self, pred_labels, gt_labels, pred_ids, gt_ids):
        iou_matrix = self._compute_iou_matrix(pred_labels, gt_labels, pred_ids, gt_ids)
        matched_gt = set()
        matched_pred = set()

        for _ in range(min(len(pred_ids), len(gt_ids))):
            if iou_matrix.size == 0:
                break
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[idx] > self.iou_threshold:
                pi, gi = idx
                matched_gt.add(gi)
                matched_pred.add(pi)
                self.iou_sum["all"] += iou_matrix[idx]
                self.tp["all"] += 1
                iou_matrix[pi, :] = 0
                iou_matrix[:, gi] = 0
            else:
                break

        self.fp["all"] += len(pred_ids) - len(matched_pred)
        self.fn["all"] += len(gt_ids) - len(matched_gt)

    def _update_per_class(self, pred_labels, gt_labels, pred_ids, gt_ids,
                          pred_class_map, gt_class_map):
        iou_matrix = self._compute_iou_matrix(pred_labels, gt_labels, pred_ids, gt_ids)

        for pi, pred_id in enumerate(pred_ids):
            pred_mask = pred_labels == pred_id
            pred_cls = int(np.median(pred_class_map[pred_mask]))
            best_iou = 0.0
            best_gi = -1

            for gi, gt_id in enumerate(gt_ids):
                if iou_matrix[pi, gi] > best_iou:
                    gt_mask = gt_labels == gt_id
                    gt_cls = int(np.median(gt_class_map[gt_mask]))
                    if gt_cls == pred_cls and iou_matrix[pi, gi] > self.iou_threshold:
                        best_iou = iou_matrix[pi, gi]
                        best_gi = gi

            if best_gi >= 0:
                self.tp[pred_cls] += 1
                self.iou_sum[pred_cls] += best_iou
            else:
                self.fp[pred_cls] += 1

        for gi, gt_id in enumerate(gt_ids):
            gt_mask = gt_labels == gt_id
            gt_cls = int(np.median(gt_class_map[gt_mask]))
            matched = False
            for pi in range(len(pred_ids)):
                if iou_matrix[pi, gi] > self.iou_threshold:
                    pred_mask = pred_labels == pred_ids[pi]
                    pred_cls = int(np.median(pred_class_map[pred_mask]))
                    if pred_cls == gt_cls:
                        matched = True
                        break
            if not matched:
                self.fn[gt_cls] += 1

    def _compute_iou_matrix(self, pred_labels, gt_labels, pred_ids, gt_ids):
        if len(pred_ids) == 0 or len(gt_ids) == 0:
            return np.zeros((len(pred_ids), len(gt_ids)))

        iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))
        for pi, pid in enumerate(pred_ids):
            pred_mask = pred_labels == pid
            for gi, gid in enumerate(gt_ids):
                gt_mask = gt_labels == gid
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                if union > 0:
                    iou_matrix[pi, gi] = intersection / union
        return iou_matrix

    def compute(self):
        results = {}

        tp = self.tp.get("all", 0)
        fp = self.fp.get("all", 0)
        fn = self.fn.get("all", 0)
        iou = self.iou_sum.get("all", 0)

        if tp > 0:
            dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-8)
            sq = iou / tp
            results["bPQ"] = dq * sq
        else:
            results["bPQ"] = 0.0

        per_class_pq = {}
        for cls in range(self.n_classes):
            tp_c = self.tp.get(cls, 0)
            fp_c = self.fp.get(cls, 0)
            fn_c = self.fn.get(cls, 0)
            iou_c = self.iou_sum.get(cls, 0)
            if tp_c > 0:
                dq_c = tp_c / (tp_c + 0.5 * fp_c + 0.5 * fn_c + 1e-8)
                sq_c = iou_c / tp_c
                per_class_pq[cls] = dq_c * sq_c
            else:
                per_class_pq[cls] = 0.0

        results["mPQ"] = np.mean(list(per_class_pq.values())) if per_class_pq else 0.0
        results["per_class_pq"] = per_class_pq

        return results
