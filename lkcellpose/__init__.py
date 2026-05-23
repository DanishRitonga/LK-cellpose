from lkcellpose.cfg import get_cfg, cfg2dict, DEFAULT_CFG_PATH
from lkcellpose.engine.model import LKCellposeModel
from lkcellpose.utils import NUM_NUCLEUS_CLASSES, NUCLEUS_CLASSES, TISSUE_TYPES, LOGGER

__version__ = "0.1.0"


def train(backbone="convnextv2_base", **kwargs):
    model = LKCellposeModel(backbone=backbone)
    return model.train(**kwargs)


def predict(source, backbone="convnextv2_base", **kwargs):
    model = LKCellposeModel(backbone=backbone)
    return model.predict(source, **kwargs)
