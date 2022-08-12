from utils.models import build_model
from utils.optimizers import build_optimizer
from utils.losses import build_loss
import logging
import pandas as pd
import os


class Task:
    """
    Desc:
        A task stores all information of training, validating and  testing
        ie: model, dataset, loss, optimizer, hook, cfg...

        The schedule class and runners class do some specify operations 
        based on the task. 
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # build model
        model_cfg = cfg["model"]
        self.model = build_model(model_cfg)
        logging.info("build model successfully")

        # dataset
        self.dataset_cfg = cfg["dataset"]
        self.num_turbines = self.dataset_cfg["num_turbines"]
        logging.info("loading dataset...")
        df_raw = pd.read_csv(os.path.join(self.dataset_cfg["path"], self.dataset_cfg["file_name"]))
        if self.dataset_cfg["position"]:
            pos = pd.read_csv(os.path.join(self.dataset_cfg["path"], self.dataset_cfg["position_name"]))
            self.dataset_cfg["pos_data"] = pos
        self.dataset_cfg["raw_data"] = df_raw
        self.dataset_cfg["input_len"] = cfg["model"]["input_len"]
        self.dataset_cfg["output_len"] = cfg["model"]["output_len"]
        logging.info("load successfully")

        # loss
        if cfg["model"]["task"] == "train":
            self.loss = build_loss(cfg)
            logging.info("build loss successfully")

        # optimizer
        if cfg["model"]["task"] == "train":
            optimizer_cfg = cfg["optimizer"]
            optimizer_cfg["model"] = self.model
            self.optimizer = build_optimizer(optimizer_cfg)
            logging.info("build optimizer successfully")

        # checkpoints
        self.save_path = cfg["checkpoint"]["save_path"]
