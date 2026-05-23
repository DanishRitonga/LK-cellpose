import copy
import random

import numpy as np
import torch

from lkcellpose.utils import LOGGER


def select_device(device: str | int | None = "auto") -> torch.device:
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "0"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if isinstance(device, int):
        device = str(device)

    if device.isdigit():
        d = torch.device(f"cuda:{device}")
        LOGGER.info(f"Using device: cuda:{device} ({torch.cuda.get_device_name(d)})")
        return d

    if "," in device:
        devices = device.split(",")
        LOGGER.info(f"Using devices: {[d.strip() for d in devices]}")
        return torch.device(f"cuda:{devices[0].strip()}")

    if device == "mps":
        LOGGER.info("Using device: mps")
        return torch.device("mps")

    LOGGER.info("Using device: cpu")
    return torch.device("cpu")


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.mul_(self.decay).add_(model_p, alpha=1 - self.decay)

    def to(self, device):
        self.ema.to(device)
        return self


class EarlyStopping:
    def __init__(self, patience=100):
        self.patience = patience
        self.best = None
        self.counter = 0

    def __call__(self, metric):
        if self.best is None or metric < self.best:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def intersect_dicts(da: dict, db: dict, exclude: tuple = ()) -> dict:
    return {
        k: v
        for k, v in da.items()
        if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape
    }


def init_seeds(seed: int = 0, deterministic: bool = True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def smart_inference_mode():
    return torch.inference_mode
