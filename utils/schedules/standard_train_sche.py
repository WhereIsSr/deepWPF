import torch
import numpy as np
from utils.tasks.task import Task
from utils.datasets import build_dataset
from utils.schedules import register_schedule
from common.learningRate import adjust_learning_rate
from common.earlyStopping import EarlyStopping
from common.evaluation import evaluate, val, test
from common.logger import shutdown_logging
import logging
import time
from time import strftime
from time import gmtime


@register_schedule("standard_train_sche")
class StandardTrainSche:
    def __init__(self, cfg: dict):
        self.num_turbines = cfg["dataset"]["num_turbines"]
        self.epochs = cfg["schedule"]["train_epochs"]
        self.patience = cfg["schedule"]["patience"]
        self.device = cfg["device"]["name"]
        self.lr_type = cfg["optimizer"]["lr_adjust"]
        self.lr_begin = cfg["optimizer"]["learning_rate"]
        self.input_size = cfg["model"]["input_size"]

    def run(self, task: Task):
        start_time = time.time()

        logging.info("prepare to train")
        model = task.model.to(self.device)
        optimizer = task.optimizer
        loss = task.loss

        predictions = []
        true_list = []
        raw_data = []
        logging.info("begin training")
        # every turbine
        for i in range(task.num_turbines):
            logging.info(f"ID of the current turbine: {i}")
            early_stopping = EarlyStopping(patience=self.patience)

            logging.info("prepare dataset")
            dataset_cfg = task.dataset_cfg
            dataset_cfg["turbine_id"] = i
            dataset_cfg["task"] = "train"
            # tuple. eg: (dataset_object, dataset_loader)
            train_datasets = build_dataset(dataset_cfg)
            val_datasets = build_dataset(dataset_cfg)
            test_datasets = build_dataset(dataset_cfg)

            # train
            logging.info("training...")
            for ep in range(self.epochs):
                train_loss = []
                dataloader = train_datasets[1]
                begin_time = time.time()
                for idx, (x, y) in enumerate(dataloader):

                    x = x.to(self.device).type(torch.float32)
                    y = y.to(self.device).type(torch.float32)[:, :, self.input_size-1:self.input_size]
                    y_pre = model(x)
                    ls = loss(y_pre, y)
                    train_loss.append(ls.item())
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()

                    if idx % 49 == 0:
                        end_time = time.time()
                        average_loss = np.array(train_loss).mean()
                        iter_time = (end_time - begin_time)/50
                        eta_second = iter_time*((self.num_turbines-i-1)*self.epochs*len(dataloader) +
                                                (self.epochs-ep-1)*len(dataloader) +
                                                len(dataloader)-idx-1)
                        eta = strftime("%H:%M:%S", gmtime(eta_second))
                        logging.info(f"turbineID:{i} epoch:{ep} iter:{idx + 1} loss:{average_loss} eta:{eta} "
                                     f"iter_time:{round(iter_time, 8)}")
                        train_loss = []
                        begin_time = time.time()

                # val
                logging.info("validating...")
                val_loss = val(model, val_datasets[1], loss, self.device)
                logging.info(f"epoch:{ep} val loss:{val_loss}")

                logging.info("early_stopping checking...")
                early_stopping(val_loss, model, task.save_path, i)
                if early_stopping.early_stop:
                    logging.info("Early stop!")
                    break

                logging.info("adjust learning rete")
                adjust_learning_rate(optimizer, ep, self.lr_type, self.lr_begin)

            # test
            # logging.info("testing...")
            # pre, true_ls, raw = test(model, test_datasets, self.device)
            # predictions.append(pre)
            # true_list.append(true_ls)
            # raw_data.append(raw)
            logging.info(f"turbine{i} training completely")

        logging.info("All turbines training completely")
        # logging.info("begin evaluating")
        # evaluate(predictions, true_list, raw_data, task.cfg)
        # logging.info("evaluating completely")
        shutdown_logging()
        total_time = time.time() - start_time
        total_time = strftime("%H:%M:%S", gmtime(total_time))
        logging.info(f"total_time: {total_time}")
        logging.info("done!")
