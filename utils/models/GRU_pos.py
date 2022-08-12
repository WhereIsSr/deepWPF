import torch.nn as nn
import torch
from utils.models import register_model


@register_model('GRU_pos')
class GRU(nn.Module):
    """
    Desc:
        model: a simple GRU model with position embedding
    """

    def __init__(self, model_cfg: dict):
        super(GRU, self).__init__()
        self.output_len = model_cfg["output_len"]
        self.com_gru = nn.GRU(input_size=model_cfg["input_size"], hidden_size=model_cfg["hidden_size"],
                              num_layers=model_cfg["num_layers"], bias=model_cfg["bias"],
                              batch_first=model_cfg["batch_first"], dropout=model_cfg["dropout"],
                              bidirectional=False)
        self.pos_linear = nn.Linear(model_cfg["input_size"] * 2, model_cfg["input_size"])
        self.proj = nn.Linear(model_cfg["hidden_size"], model_cfg["output_size"])
        self.input_size = model_cfg["input_size"]

    def forward(self, x):
        pos = x[:, :, -self.input_size*2:]
        pos = self.pos_linear(pos)
        x = x[:, :, 0:self.input_size]
        x = x + pos
        x_pre = torch.zeros([x.shape[0], self.output_len, x.shape[2]]).to(x.device)
        x_pre = torch.concat((x, x_pre), 1)
        x_pre = self.proj(self.com_gru(x_pre)[0])
        return x_pre[:, -self.output_len:, :]
