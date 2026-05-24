import torch
from pathlib import Path
from lkcellpose.engine.trainer import BaseTrainer
from lkcellpose.nn.tasks import PanopticCellposeModel
from lkcellpose.data.pannuke import PanNukeDataset
from lkcellpose.data.augment import CellposeAugment
from lkcellpose.utils import LOGGER


class PanopticTrainer(BaseTrainer):
    loss_names = ("flow_loss", "cellprob_loss", "class_loss", "total_loss")

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "panoptic"
        from lkcellpose.cfg import DEFAULT_CFG_PATH
        super().__init__(cfg or DEFAULT_CFG_PATH, overrides, _callbacks)

    def get_model(self):
        model = PanopticCellposeModel(
            backbone=self.args.get("backbone", "convnextv2_base"),
            pretrained=self.args.get("pretrained", True),
            pretrained_tag=self.args.get("pretrained_tag", "fcmae_ft_in22k_in1k"),
            nout=8 if self.args.get("panoptic", True) else 3,
            n_classes=self.args.get("n_classes", 5),
            grad_checkpoint=self.args.get("grad_checkpoint", True),
            panoptic=self.args.get("panoptic", True),
        )
        return model

    def get_validator(self):
        from lkcellpose.models.panoptic.val import PanopticValidator
        return PanopticValidator(save_dir=self.save_dir, args=dict(vars(self.args)))

    def get_dataloader(self, split):
        folds_key = f"{split}_folds"
        folds = self.args.get(folds_key, [1] if split == "train" else [2])
        augment = CellposeAugment(
            scale_range=self.args.get("scale_range", 0.5),
            grayscale_prob=self.args.get("grayscale_prob", 0.1),
            invert_prob=self.args.get("invert_prob", 0.25),
            channel_dropout_prob=self.args.get("channel_dropout_prob", 0.1),
            brightness_std=self.args.get("brightness_std", 0.2),
            contrast_range=self.args.get("contrast_range", (0.5, 1.5)),
            degradation_prob=self.args.get("degradation_prob", 0.5),
            hflip=self.args.get("hflip", True),
            vflip=self.args.get("vflip", True),
            rotation=self.args.get("rotation", True),
        ) if split == "train" and self.args.get("augment", True) else None

        datasets = []
        for fold in folds:
            ds = PanNukeDataset(
                fold=fold,
                split=split,
                data_dir=self.args.get("data_dir"),
                augment=augment,
                n_classes=self.args.get("n_classes", 5),
                min_masks=self.args.get("min_masks", 5),
                device=self.device if split == "train" else None,
            )
            datasets.append(ds)

        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            from torch.utils.data import ConcatDataset
            dataset = ConcatDataset(datasets)

        shuffle = split == "train"
        batch_size = self.batch_size if split == "train" else self.batch_size * 2
        workers = self.args.get("workers", 8)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            drop_last=split == "train",
        )

    def label_loss_items(self, loss_items):
        if isinstance(loss_items, dict):
            return tuple(loss_items.get(k, 0.0) for k in self.loss_names)
        return loss_items

    def progress_string(self):
        return f"Epoch {self.epoch}/{self.epochs}, loss={self.loss:.4f}"
