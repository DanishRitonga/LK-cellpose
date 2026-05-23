from torch.utils.tensorboard import SummaryWriter


def on_train_start(trainer):
    trainer.writer = SummaryWriter(trainer.save_dir / "tensorboard")


def on_train_batch_end(trainer):
    if not hasattr(trainer, "writer") or trainer.writer is None:
        return
    if trainer.global_step % 100 != 0:
        return
    trainer.writer.add_scalar("train/loss", trainer.loss, trainer.global_step)
    if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
        for i, v in enumerate(trainer.loss_items):
            trainer.writer.add_scalar(f"train/loss_item_{i}", v, trainer.global_step)
    lr = trainer.optimizer.param_groups[0]["lr"]
    trainer.writer.add_scalar("train/lr", lr, trainer.global_step)


def on_fit_epoch_end(trainer):
    if not hasattr(trainer, "writer") or trainer.writer is None:
        return
    lr = trainer.optimizer.param_groups[0]["lr"]
    trainer.writer.add_scalar("train/lr", lr, trainer.epoch)
    if hasattr(trainer, "metrics") and trainer.metrics:
        for k, v in trainer.metrics.items():
            trainer.writer.add_scalar(f"val/{k}", v, trainer.epoch)


def on_train_end(trainer):
    if hasattr(trainer, "writer") and trainer.writer is not None:
        trainer.writer.close()
        trainer.writer = None


callbacks = {
    "on_train_start": on_train_start,
    "on_train_batch_end": on_train_batch_end,
    "on_fit_epoch_end": on_fit_epoch_end,
    "on_train_end": on_train_end,
}
