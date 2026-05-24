import torch
import numpy as np
from tqdm import tqdm
from lkcellpose.cfg import get_cfg, DEFAULT_CFG_PATH
from lkcellpose.utils import LOGGER, NUCLEUS_CLASSES
from lkcellpose.utils.metrics import PanopticQuality
from lkcellpose.utils.torch_utils import smart_inference_mode
from lkcellpose.utils.callbacks import get_default_callbacks


class BaseValidator:
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        self.args = get_cfg(overrides=args) if args else get_cfg()
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.training = True
        self.callbacks = _callbacks or get_default_callbacks()
        self.metrics = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
        from lkcellpose.utils.torch_utils import select_device
        self.device = select_device(self.args.get("device", "auto"))

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            model = trainer.ema.ema if trainer.ema else trainer.model
            self.dataloader = trainer.val_loader
            self.save_dir = trainer.save_dir
        else:
            model = model.to(self.device)
            self.dataloader = self.get_dataloader()

        self.run_callbacks("on_val_start")
        self.init_metrics(model)
        results = {}
        for batch in tqdm(self.dataloader, desc="Validating"):
            self.run_callbacks("on_val_batch_start")
            batch = self.preprocess(batch)
            preds = model(batch["img"])
            self.update_metrics(preds, batch)
            self.run_callbacks("on_val_batch_end")
        results = self.get_stats()
        self.print_results(results)
        self.run_callbacks("on_val_end")
        return results

    def run_callbacks(self, event):
        for cb in self.callbacks.get(event, []):
            cb(self)

    def init_metrics(self, model):
        raise NotImplementedError

    def preprocess(self, batch):
        if self.training:
            batch["img"] = batch["img"].to(self.device)
            batch["flows"] = batch["flows"].to(self.device)
            batch["class_map"] = batch["class_map"].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        raise NotImplementedError

    def get_stats(self):
        raise NotImplementedError

    def print_results(self, results):
        raise NotImplementedError

    def get_dataloader(self):
        raise NotImplementedError
