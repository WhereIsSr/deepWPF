import torch
import yaml
import os
from datetime import datetime


def get_config(path: str):
    """
    Desc:
        get config from yaml file
    return:
        cfg:dict
    """
    with open(path, 'r') as f:
        cfg = yaml.full_load(f)

    # device
    if cfg["device"]["name"] == "gpu":
        cfg["device"]["name"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        cfg["device"]["name"] = torch.device("cpu")

    # setup path
    if cfg["model"]["task"] == "train":
        cur_setup = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
            cfg["model"]["name"], cfg["model"]["task"], cfg["loss"]["name"], cfg["optimizer"]["name"],
            cfg["schedule"]["name"], cfg["dataset"]["name"], cfg["device"]["name"], datetime.now()
        )
    else:
        cur_setup = '{}_{}_{}_{}_{}_{}'.format(
            cfg["model"]["name"], cfg["model"]["task"], cfg["schedule"]["name"], cfg["dataset"]["name"],
            cfg["device"]["name"], datetime.now()
        )
    save_path = os.path.join(cfg["checkpoint"]["path"], cur_setup)
    cfg["checkpoint"]["save_path"] = save_path
    if not cfg["is_debug"] and not os.path.exists(save_path):
        os.makedirs(save_path)

    return cfg

