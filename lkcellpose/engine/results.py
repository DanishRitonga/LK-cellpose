import numpy as np
from pathlib import Path


class Results:
    """Container for prediction results."""

    def __init__(self, orig_img=None, masks=None, classes=None, pred=None, path=None):
        self.orig_img = orig_img
        self.masks = masks
        self.classes = classes
        self.pred = pred
        self.path = path

    @property
    def n_instances(self):
        return len(np.unique(self.masks)) - 1 if self.masks is not None else 0

    def plot(self, show_classes=True, alpha=0.5, return_rgb=False):
        import matplotlib.cm as cm
        img = self.orig_img.copy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        overlay = img.copy()
        instance_ids = np.unique(self.masks)
        instance_ids = instance_ids[instance_ids > 0]
        colors = cm.tab20(np.linspace(0, 1, max(len(instance_ids), 1)))
        for i, inst_id in enumerate(instance_ids):
            mask = self.masks == inst_id
            c = colors[i % len(colors)][:3]
            if show_classes and self.classes is not None:
                c = self._class_color(self.classes[mask][0]) if mask.any() else c
            overlay[mask] = (overlay[mask].astype(float) * (1 - alpha) + np.array(c) * 255 * alpha).astype(np.uint8)
        return overlay if return_rgb else overlay

    def _class_color(self, cls_id):
        from lkcellpose.utils import NUCLEUS_CLASSES
        colors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
        ]
        return colors[cls_id % len(colors)]

    def save(self, filename):
        from PIL import Image
        rgb = self.plot(return_rgb=True)
        Image.fromarray(rgb).save(filename)

    def summary(self):
        return {
            "n_instances": self.n_instances,
            "shape": self.masks.shape if self.masks is not None else None,
        }

    def __repr__(self):
        return f"Results(n_instances={self.n_instances})"
