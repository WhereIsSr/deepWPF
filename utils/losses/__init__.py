from common.importModule import import_all_modules
from pathlib import Path


LOSS_REGISTRY = {}


def register_loss(name):
    """
        Registers Loss.

        To use it, apply this decorator to a loss function, like this:

        .. code-block:: python

            @register_loss('my_loss_name')
            def MyLossName():
                ...

    """

    def register_loss_class(func):
        if name in LOSS_REGISTRY:
            raise ValueError("Cannot register duplicate loss ({})".format(name))

        LOSS_REGISTRY[name] = func
        return func

    return register_loss_class


def get_loss(loss_name: str):
    assert loss_name in LOSS_REGISTRY, "Unknown loss"
    return LOSS_REGISTRY[loss_name]


def build_loss(cfg: dict):
    loss = get_loss(cfg["loss"]["name"])
    return loss(cfg)


import_all_modules(str(Path(__file__).parent), "utils.losses")
