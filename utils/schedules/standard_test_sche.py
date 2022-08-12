import torch
import numpy as np
from utils.tasks.task import Task
from utils.datasets import build_dataset
from utils.schedules import register_schedule
from common.evaluation import evaluate, test
from common.logger import shutdown_logging
import logging


@register_schedule("standard_test_sche")
class StandardTestSche:
    def __init__(self, cfg: dict):
        self.device = cfg["device"]["name"]

    def run(self, task: Task):
        logging.info("prepare to test")
        model = task.model.to(self.device)

        predictions = []
        true_list = []
        raw_data = []
        logging.info("begin testing")
        # every turbine
        for i in range(task.num_turbines):
            logging.info(f"ID of the current turbine: {i}")

            logging.info("prepare dataset")
            dataset_cfg = task.dataset_cfg
            dataset_cfg["turbine_id"] = i
            dataset_cfg["task"] = "test"
            # tuple. eg: (dataset_object, dataset_loader)
            test_datasets = build_dataset(dataset_cfg)

            # test
            logging.info("testing...")
            pre, true_ls, raw = test(model, test_datasets, self.device)
            predictions.append(pre)
            true_list.append(true_ls)
            raw_data.append(raw)
            logging.info(f"turbine{i} testing completely")

        logging.info("All turbines testing completely")
        logging.info("begin evaluating")
        evaluate(predictions, true_list, raw_data, task.cfg)
        logging.info("evaluating completely")
        shutdown_logging()
        logging.info("done!")
