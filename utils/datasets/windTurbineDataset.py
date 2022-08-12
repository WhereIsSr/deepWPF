from torch.utils.data import Dataset
from utils.datasets import register_dataset
import pandas as pd
import numpy as np
import torch
import os


class Scaler(object):
    """
    Desc: Normalization utilities
    """

    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        """
        Desc:
            Fit the data
        Args:
            data:
        Returns:
            None
        """
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        """
        Desc:
            Transform the data
        Args:
            data:
        Returns:
            The transformed data
        """
        mean = torch.tensor(self.mean).type_as(data.dtype()).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.tensor(self.std).type_as(data.dtype()).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        """
        Desc:
            Restore to the original data
        Args:
            data: the transformed data
        Returns:
            The original data
        """
        mean = torch.tensor(self.mean) if torch.is_tensor(data) else self.mean
        std = torch.tensor(self.std) if torch.is_tensor(data) else self.std
        return (data * std) + mean


@register_dataset("WindTurbineDataset")
class WindTurbineDataset(Dataset):
    def __init__(self, cfg: dict):
        super().__init__()
        self.path = cfg["path"]
        self.fileName = cfg["file_name"]
        self.startCol = cfg["start_col"]
        self.trainDays = cfg["train_days"]
        self.valDays = cfg["val_days"]
        self.testDays = cfg["test_days"]
        self.inputLen = cfg["input_len"]
        self.outputLen = cfg["output_len"]
        self.task = cfg["task"]
        self.turbine_id = cfg["turbine_id"]
        self.total_days = cfg["total_days"]
        self.day_len = cfg["day_len"]
        self.total_size = self.total_days * self.day_len
        self.position = cfg["position"]
        if self.position:
            self.pos_data = cfg["pos_data"]

        self.__read_data(cfg["raw_data"])

    def __len__(self):
        return len(self.data) - self.inputLen - self.outputLen + 1

    def __getitem__(self, index):
        x = self.data[index:index + self.inputLen]
        y = self.data[index + self.inputLen:index + self.inputLen + self.outputLen]
        return x, y

    def __read_data(self, raw_data):
        # index of the slice border
        train_border1 = self.turbine_id * self.total_size
        train_border2 = train_border1 + self.trainDays * self.day_len
        val_border1 = self.turbine_id * self.total_size + self.trainDays * self.day_len - self.inputLen
        val_border2 = val_border1 + self.valDays * self.day_len
        test_border1 = self.turbine_id * self.total_size + self.trainDays * self.day_len + self.valDays * self.day_len \
                       - self.inputLen
        test_border2 = test_border1 + self.testDays * self.day_len

        cols_data = raw_data[raw_data.columns[self.startCol:]]

        pd.set_option('mode.chained_assignment', None)
        cols_data.replace(to_replace=np.nan, value=0, inplace=True)

        # normalize
        self.scaler = Scaler()
        if self.scaler:
            train_data = cols_data[train_border1:train_border2]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(cols_data.values)
        else:
            data = cols_data.values
        data = torch.tensor(data)

        # position embedding
        if self.position:
            inv_freq = 1 / (10000 ** (torch.arange(0., data.shape[1], 2)/data.shape[1]))
            inp_x = torch.outer(torch.tensor(self.pos_data[self.pos_data.columns[0]]), inv_freq)
            inp_x = torch.cat([inp_x.sin(), inp_x.cos()], dim=-1)
            inp_y = torch.outer(torch.tensor(self.pos_data[self.pos_data.columns[1]]), inv_freq)
            inp_y = torch.cat([inp_y.sin(), inp_y.cos()], dim=-1)
            data = torch.cat((data, torch.repeat_interleave(inp_x, self.total_size, dim=0)), dim=1)
            data = torch.cat((data, torch.repeat_interleave(inp_y, self.total_size, dim=0)), dim=1)

        # slice to get data
        if self.task == "train":
            self.data = data[train_border1:train_border2]
            self.raw_data = cols_data[train_border1 + self.inputLen:train_border2]
        elif self.task == "val":
            self.data = data[val_border1:val_border2]
            self.raw_data = cols_data[val_border1 + self.inputLen:val_border2]
        else:
            self.data = data[test_border1:test_border2]
            self.raw_data = cols_data[test_border1 + self.inputLen:test_border2]

    def get_rawData(self):
        return self.raw_data

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

