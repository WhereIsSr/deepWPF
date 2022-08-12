from common.importModule import import_all_modules
from pathlib import Path


SCHEDULE_REGISTRY = {}


def register_schedule(name):
    """
        Registers Schedule.

        To use it, apply this decorator to a schedule class, like this:

        .. code-block:: python

            @register_schedule('my_schedule_name')
            class MyScheduleName():
                ...

    """

    def register_schedule_class(func):
        if name in SCHEDULE_REGISTRY:
            raise ValueError("Cannot register duplicate schedule ({})".format(name))

        SCHEDULE_REGISTRY[name] = func
        return func

    return register_schedule_class


def get_schedule(schedule_name: str):
    assert schedule_name in SCHEDULE_REGISTRY, "Unknown schedule"
    return SCHEDULE_REGISTRY[schedule_name]


def build_schedule(cfg: dict):
    schedule = get_schedule(cfg["schedule"]["name"])
    return schedule(cfg)


import_all_modules(str(Path(__file__).parent), "utils.schedules")
