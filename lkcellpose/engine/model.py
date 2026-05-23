from pathlib import Path
from lkcellpose.cfg import get_cfg, DEFAULT_CFG_PATH, MODES
from lkcellpose.utils import LOGGER


class LKCellposeModel:
    """
    Central facade class for LK-Cellpose.
    
    Usage:
        model = LKCellposeModel("convnextv2_base")
        model.train(data="pannuke", epochs=100)
        results = model.predict("image.png")
    """

    task_map = {
        "panoptic": {
            "trainer": "lkcellpose.models.panoptic.train.PanopticTrainer",
            "validator": "lkcellpose.models.panoptic.val.PanopticValidator",
            "predictor": "lkcellpose.models.panoptic.predict.PanopticPredictor",
        }
    }

    def __init__(self, backbone="convnextv2_base", task="panoptic", verbose=True):
        self.backbone = backbone
        self.task = task
        self.verbose = verbose
        self.model = None
        self.trainer = None

    def train(self, cfg=DEFAULT_CFG_PATH, **overrides):
        overrides.setdefault("backbone", self.backbone)
        overrides.setdefault("task", self.task)
        overrides["mode"] = "train"
        trainer_cls = self._resolve_class(self.task_map[self.task]["trainer"])
        self.trainer = trainer_cls(cfg=cfg, overrides=overrides)
        self.trainer.train()
        self.model = self.trainer.model
        return self.trainer.metrics

    def val(self, cfg=DEFAULT_CFG_PATH, **overrides):
        overrides.setdefault("backbone", self.backbone)
        overrides.setdefault("task", self.task)
        overrides["mode"] = "val"
        validator_cls = self._resolve_class(self.task_map[self.task]["validator"])
        validator = validator_cls(args=overrides)
        return validator(model=self.model)

    def predict(self, source, cfg=DEFAULT_CFG_PATH, **overrides):
        overrides.setdefault("backbone", self.backbone)
        overrides.setdefault("task", self.task)
        overrides["mode"] = "predict"
        predictor_cls = self._resolve_class(self.task_map[self.task]["predictor"])
        predictor = predictor_cls(cfg=cfg, overrides=overrides)
        return predictor(source, model=self.model)

    @staticmethod
    def _resolve_class(path):
        module_path, class_name = path.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
