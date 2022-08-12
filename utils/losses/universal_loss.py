import torch.nn as nn
from utils.losses import register_loss


@register_loss("MSE")
def get_MSE(cfg):
    return nn.MSELoss()

