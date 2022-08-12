from common.importModule import import_all_modules
from pathlib import Path
import torch


MODEL_REGISTRY = {}


def register_model(name):
    """
    Registers Model.

    To use it, apply this decorator to a model class, like this:

    .. code-block:: python

        @register_model('my_model_name')
        class MyModelName():
            ...

    """

    def register_model_class(func):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))

        MODEL_REGISTRY[name] = func
        return func

    return register_model_class


def get_model(model_name: str):
    """
    Lookup the model_name in the model registry and return.
    If the model is not implemented, asserts will be thrown and workflow will exit.
    """
    assert model_name in MODEL_REGISTRY, "Unknown model"
    return MODEL_REGISTRY[model_name]


def build_model(model_cfg: dict):
    """
    Given the model config, construct the model and return it.

    return:
        model
    """
    model = get_model(model_cfg["name"])
    m = model(model_cfg)
    if model_cfg["task"] == "test":
        resume_path = model_cfg["resume_path"]
        m.load_state_dict(torch.load(resume_path))
    return m


import_all_modules(str(Path(__file__).parent), "utils.models")
