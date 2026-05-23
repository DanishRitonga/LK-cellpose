from copy import deepcopy
from lkcellpose.utils.callbacks.base import default_callbacks


def get_default_callbacks():
    return deepcopy(default_callbacks)


def add_integration_callbacks(callbacks):
    from lkcellpose.utils.callbacks.tensorboard import callbacks as tb_cbs
    for k, v in tb_cbs.items():
        if k in callbacks:
            callbacks[k].append(v)
    return callbacks
