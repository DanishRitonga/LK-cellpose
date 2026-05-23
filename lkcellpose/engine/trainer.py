import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from contextlib import nullcontext
from pathlib import Path
from tqdm import tqdm

mp.set_start_method("spawn", force=True)

from lkcellpose.cfg import get_cfg, DEFAULT_CFG_PATH
from lkcellpose.utils import LOGGER
from lkcellpose.utils.torch_utils import select_device, ModelEMA, EarlyStopping, init_seeds
from lkcellpose.utils.callbacks import get_default_callbacks, add_integration_callbacks


class BaseTrainer:
    def __init__(self, cfg=DEFAULT_CFG_PATH, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides.setdefault("mode", "train")
        self.args = get_cfg(cfg, overrides)
        self.device = select_device(self.args.get("device", "auto"))
        self.validator = None
        self.metrics = None
        self.plots = {}
        self.save_dir = Path(self.args.project) / self.args.name
        self.wdir = self.save_dir / "weights"
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"
        self.batch_size = self.args.get("batch", 8)
        self.epochs = self.args.get("epochs", 2000)
        self.start_epoch = 0
        self.best_metric = float("inf")
        self.global_step = 0
        self.loss = 0.0
        self.loss_items = None
        self.epoch = 0

        init_seeds(self.args.get("seed", 0), deterministic=True)
        self.callbacks = _callbacks or get_default_callbacks()
        add_integration_callbacks(self.callbacks)

    def add_callback(self, event, callback):
        self.callbacks.setdefault(event, []).append(callback)

    def run_callbacks(self, event):
        for cb in self.callbacks.get(event, []):
            cb(self)

    def train(self):
        self._setup_train()
        self.run_callbacks("on_train_start")
        self._do_train()
        self.run_callbacks("on_train_end")

    def _setup_train(self):
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.model = self.get_model()
        self.model.to(self.device)
        self.model.set_model_args(**{k: v for k, v in vars(self.args).items()
                                     if k in ("flow_weight", "cellprob_weight", "class_weight",
                                              "focal_gamma", "focal_alpha")})
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.train_loader = self.get_dataloader("train")
        self.val_loader = self.get_dataloader("val")
        self.scaler = None
        use_amp = self.args.get("amp", True)
        amp_dtype = getattr(torch, self.args.get("amp_dtype", "bfloat16"), torch.bfloat16)
        self.amp_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
        self.amp_dtype = amp_dtype
        if self.args.get("ema", True):
            self.ema = ModelEMA(self.model, decay=self.args.get("ema_decay", 0.9999))
        else:
            self.ema = None
        self.early_stopping = EarlyStopping(patience=self.args.get("patience", 100))
        if self.args.get("freeze_encoder", False):
            self._freeze_encoder()

    def _do_train(self):
        grad_accum = self.args.get("grad_accum_steps", 32)
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
            optimizer_step = 0
            for i, batch in enumerate(pbar):
                self.run_callbacks("on_train_batch_start")
                batch = self.preprocess_batch(batch)
                with self.amp_ctx:
                    loss, loss_items = self.model(batch)
                    loss = loss / grad_accum
                loss.backward()
                self.loss = loss.item() * grad_accum
                self.loss_items = {k: v.item() for k, v in loss_items.items()}
                if (i + 1) % grad_accum == 0 or (i + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.ema:
                        self.ema.update(self.model)
                    optimizer_step += 1
                self.global_step += 1
                pbar.set_postfix(loss=self.loss, lr=self.optimizer.param_groups[0]["lr"])
                self.run_callbacks("on_train_batch_end")
            if self.scheduler:
                self.scheduler.step()
            self.run_callbacks("on_train_epoch_end")
            if (epoch + 1) % self.args.get("val_interval", 1) == 0:
                metrics = self.validate()
                self.metrics = metrics
                val_metric = metrics.get("mPQ", metrics.get("bPQ", 0.0))
                if val_metric > self.best_metric:
                    self.best_metric = val_metric
                    self.save_checkpoint("best")
                self.save_checkpoint("last")
                if self.early_stopping(-val_metric):
                    LOGGER.info(f"Early stopping at epoch {epoch}")
                    break
            self.run_callbacks("on_fit_epoch_end")

    def validate(self):
        validator = self.get_validator()
        return validator(trainer=self)

    def save_checkpoint(self, name="last"):
        ckpt = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "args": dict(vars(self.args)),
        }
        if self.ema:
            ckpt["ema"] = self.ema.ema.state_dict()
        torch.save(ckpt, self.wdir / f"{name}.pt")

    def build_optimizer(self):
        lr = self.args.get("lr0", 5e-5)
        wd = self.args.get("weight_decay", 0.1)
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

    def build_scheduler(self):
        warmup = self.args.get("warmup_epochs", 10)
        return LRWarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=warmup,
            total_epochs=self.epochs,
            lr_decay_milestones=self.args.get("lr_decay_milestones", [-100, -50]),
            lr_decay_factor=self.args.get("lr_decay_factor", 10),
        )

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device)
        batch["flows"] = batch["flows"].to(self.device)
        if "cellprob" in batch:
            batch["cellprob"] = batch["cellprob"].to(self.device)
        batch["class_map"] = batch["class_map"].to(self.device)
        return batch

    def _freeze_encoder(self):
        if hasattr(self.model, "encoder"):
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            LOGGER.info("Encoder frozen")

    def get_model(self):
        raise NotImplementedError

    def get_validator(self):
        raise NotImplementedError

    def get_dataloader(self, split):
        raise NotImplementedError


class LRWarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs=10, total_epochs=2000,
                 lr_decay_milestones=None, lr_decay_factor=10):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_decay_milestones = lr_decay_milestones or []
        self.lr_decay_factor = lr_decay_factor
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_lr = 0.0
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            self.current_lr = self.base_lr * self.epoch / self.warmup_epochs
        else:
            self.current_lr = self.base_lr
            for milestone in self.lr_decay_milestones:
                abs_milestone = self.total_epochs + milestone if milestone < 0 else milestone
                if self.epoch >= abs_milestone:
                    self.current_lr = self.base_lr / self.lr_decay_factor
                    self.base_lr = self.current_lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.current_lr
