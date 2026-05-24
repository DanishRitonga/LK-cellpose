from types import SimpleNamespace
from typing import Any
from pathlib import Path


class IterableSimpleNamespace(SimpleNamespace):
    def __iter__(self):
        return iter(vars(self).items())

    def __str__(self):
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(
            f"'{name}' has no attribute '{attr}'. Check lkcellpose/cfg/default.yaml."
        )

    def get(self, key, default=None):
        return vars(self).get(key, default)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


import logging

LOGGER = logging.getLogger("lkcellpose")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(_handler)
    LOGGER.setLevel(logging.INFO)

RANK = -1
ROOT = Path(__file__).parent.parent

NUM_NUCLEUS_CLASSES = 5
NUCLEUS_CLASSES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
TISSUE_TYPES = [
    "Adrenal Gland", "Bile Duct", "Bladder", "Breast", "Cervix",
    "Colon", "Esophagus", "Head & Neck", "Kidney", "Liver",
    "Lung", "Ovarian", "Pancreatic", "Prostate", "Skin",
    "Stomach", "Testis", "Thyroid", "Uterus",
]
