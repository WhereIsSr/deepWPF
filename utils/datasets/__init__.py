from common.importModule import import_all_modules
from pathlib import Path
from torch.utils.data import DataLoader


DATASET_REGISTRY = {}


def register_dataset(name):
    """
        Registers Dataset.

        To use it, apply this decorator to a dataset class, like this:

        .. code-block:: python

            @register_dataset('my_dataset_name')
            class MyDatasetName():
                ...

    """

    def register_dataset_class(func):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))

        DATASET_REGISTRY[name] = func
        return func

    return register_dataset_class


def get_dataset(dataset_name: str):
    assert dataset_name in DATASET_REGISTRY, "Unknown dataset"
    return DATASET_REGISTRY[dataset_name]


def build_dataset(dataset_cfg: dict):
    dataset = get_dataset(dataset_cfg["name"])
    dataset_object = dataset(dataset_cfg)
    if dataset_cfg["task"] == "train":
        dataset_loader = DataLoader(dataset_object, batch_size=dataset_cfg["batch_size"], shuffle=True, drop_last=True,
                                    num_workers=2)
    else:
        dataset_loader = DataLoader(dataset_object, batch_size=dataset_cfg["batch_size"], shuffle=False, drop_last=True,
                                    num_workers=2)
    return dataset_object, dataset_loader


import_all_modules(str(Path(__file__).parent), "utils.datasets")




