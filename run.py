from config.config import get_config
from utils.tasks.task import Task
from utils.schedules import build_schedule
from common.logger import init_logger
import logging


def run():
    cfg = get_config("config/file/baseline_GRU_train.yaml")
    init_logger("WPF", cfg, is_test=cfg["is_debug"])
    logging.info(cfg)

    task = Task(cfg)

    sche = build_schedule(cfg)
    sche.run(task)


if __name__ == '__main__':
    run()
