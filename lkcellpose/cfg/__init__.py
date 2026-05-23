from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from lkcellpose.utils import IterableSimpleNamespace, LOGGER

DEFAULT_CFG_PATH = Path(__file__).parent / "default.yaml"


def cfg2dict(cfg: str | Path | dict | SimpleNamespace) -> dict:
    if isinstance(cfg, (str, Path)):
        p = Path(cfg)
        if not p.suffix:
            p = Path(__file__).parent / f"{cfg}.yaml"
        cfg = yaml.safe_load(p.read_text())
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)
    return cfg


def check_cfg(cfg: dict) -> None:
    for k, v in cfg.items():
        if v is None:
            continue
        if not isinstance(k, str):
            raise TypeError(f"Config key must be str, got {type(k)}: {k}")


def get_cfg(
    cfg: str | Path | dict | SimpleNamespace = DEFAULT_CFG_PATH,
    overrides: dict | None = None,
) -> IterableSimpleNamespace:
    cfg = cfg2dict(cfg)
    if overrides:
        overrides = cfg2dict(overrides)
        cfg = {**cfg, **overrides}
    check_cfg(cfg)
    return IterableSimpleNamespace(**cfg)


MODES = {"train", "val", "predict"}
TASKS = {"panoptic"}
