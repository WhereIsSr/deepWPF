import os
import sys
import importlib


def import_all_modules(root: str, base_module: str) -> None:
    """
    import all files that end with [".py", ".pyc"] in the root folder as modules
    :param root: path of the folder
    :param base_module: name of the module
    """
    for file in os.listdir(root):
        if file.endswith((".py", ".pyc")) and not file.startswith("_"):
            module = file[: file.find(".py")]
            if module not in sys.modules:
                module_name = ".".join([base_module, module])
                importlib.import_module(module_name)
