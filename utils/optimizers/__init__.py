from common.importModule import import_all_modules
from pathlib import Path


OPTIMIZER_REGISTRY = {}


def register_optimizer(name):
    """
        Registers Optimizer.

        To use it, apply this decorator to an optimizer function, like this:

        .. code-block:: python

            @register_optimizer('my_optimizer_name')
            def MyOptimizerName():
                ...

    """

    def register_optimizer_class(func):
        if name in OPTIMIZER_REGISTRY:
            raise ValueError("Cannot register duplicate optimizer ({})".format(name))

        OPTIMIZER_REGISTRY[name] = func
        return func

    return register_optimizer_class


def get_optimizer(optimizer_name: str):
    assert optimizer_name in OPTIMIZER_REGISTRY, "Unknown optimizer"
    return OPTIMIZER_REGISTRY[optimizer_name]


def build_optimizer(cfg: dict):
    optimizer = get_optimizer(cfg["name"])
    return optimizer(cfg)


import_all_modules(str(Path(__file__).parent), "utils.optimizers")
