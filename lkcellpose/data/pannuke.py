import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from lkcellpose.data.transforms import normalize_img, cache_flows, compute_class_map
from lkcellpose.data.augment import CellposeAugment
from lkcellpose.utils import NUCLEUS_CLASSES


class PanNukeDataset(Dataset):
    """
    PanNuke dataset for panoptic nuclei segmentation.

    Loads from HuggingFace datasets (RationAI/PanNuke) or local .npy files.
    Precomputes and caches Cellpose flow fields.
    """

    def __init__(self, fold, split="train", data_dir=None, cache_dir=None,
                 augment=None, n_classes=5, min_masks=5, device=None):
        """
        Args:
            fold: int, PanNuke fold number (1, 2, or 3)
            split: str, "train" or "val" or "test" (used for augmentation toggle)
            data_dir: Path or None, if set, load from local .npy files instead of HuggingFace
            cache_dir: Path or None, directory for flow field cache
            augment: CellposeAugment instance or None
            n_classes: int, number of nucleus classes
            min_masks: int, minimum number of masks per image
            device: torch device for flow computation
        """
        super().__init__()
        self.fold = fold
        self.split = split
        self.augment = augment if (augment is not None and split == "train") else None
        self.n_classes = n_classes
        self.min_masks = min_masks
        self.device = device

        if cache_dir is None:
            cache_dir = Path("cache") / "pannuke_flows"
        self.cache_dir = cache_dir

        self.samples = []
        if data_dir is not None:
            self._load_from_npy(data_dir, fold)
        else:
            self._load_from_huggingface(fold)

    def _load_from_huggingface(self, fold):
        from datasets import load_dataset
        dataset = load_dataset("RationAI/PanNuke", split=f"fold{fold}")
        for idx, sample in enumerate(dataset):
            n_masks = len(sample["categories"])
            if n_masks < self.min_masks:
                continue
            self.samples.append({
                "idx": idx,
                "fold": fold,
                "image": sample["image"],
                "instances": sample["instances"],
                "categories": sample["categories"],
                "tissue": sample["tissue"],
            })

    def _load_from_npy(self, data_dir, fold):
        import numpy as np
        data_dir = Path(data_dir)
        images = np.load(data_dir / f"fold{fold}" / "images" / f"fold{fold}" / "images.npy", mmap_mode="r")
        masks = np.load(data_dir / f"fold{fold}" / "masks" / f"fold{fold}" / "masks.npy", mmap_mode="r")
        for idx in range(images.shape[0]):
            instance_map = np.zeros(images.shape[1:3], dtype=np.int32)
            categories = []
            counter = 1
            for ch in range(5):
                inst_ids = np.unique(masks[idx, :, :, ch])
                inst_ids = inst_ids[inst_ids > 0]
                for inst_id in inst_ids:
                    instance_map[masks[idx, :, :, ch] == inst_id] = counter
                    categories.append(ch)
                    counter += 1
            if counter - 1 < self.min_masks:
                continue
            self.samples.append({
                "idx": idx,
                "fold": fold,
                "image_np": images[idx],
                "instance_map": instance_map,
                "categories": categories,
                "tissue": 0,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if "image" in sample and not isinstance(sample.get("image"), np.ndarray):
            img = np.array(sample["image"])
        else:
            img = sample.get("image_np", np.zeros((256, 256, 3), dtype=np.uint8))

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        img = normalize_img(img)

        if "instance_map" in sample:
            labels = sample["instance_map"]
            categories = sample["categories"]
        else:
            labels, categories = self._build_instance_map(sample)

        cache_result = cache_flows(
            self.cache_dir, sample["idx"], sample["fold"],
            labels, categories, device=self.device
        )
        flows = cache_result["flows"]
        class_map = cache_result["class_map"]

        data = {
            "img": img,
            "flows": flows,
            "class_map": class_map,
            "labels": labels,
        }

        if self.augment is not None:
            data = self.augment(data)

        img_t = torch.from_numpy(data["img"]).permute(2, 0, 1).float()
        flows_t = torch.from_numpy(data["flows"]).float()
        class_map_t = torch.from_numpy(data["class_map"]).long()
        labels_t = torch.from_numpy(data["labels"]).long()

        return {
            "img": img_t,
            "flows": flows_t,
            "class_map": class_map_t,
            "labels": labels_t,
            "idx": idx,
        }

    def _build_instance_map(self, sample):
        h, w = 256, 256
        inst_map = np.zeros((h, w), dtype=np.int32)
        categories = []
        counter = 1
        for inst_mask, cat in zip(sample["instances"], sample["categories"]):
            mask = np.array(inst_mask) > 0
            inst_map[mask] = counter
            categories.append(cat)
            counter += 1
        return inst_map, categories
