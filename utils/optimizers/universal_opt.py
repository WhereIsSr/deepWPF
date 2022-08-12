from utils.optimizers import register_optimizer
import torch


@register_optimizer("Adam")
def adam(cfg: dict):
    return torch.optim.Adam(
        params=cfg["model"].parameters(),
        lr=cfg["learning_rate"]
    )
