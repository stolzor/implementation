import math

import torch
from torch import nn


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, sent_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.sent_len = sent_len
        self.device = torch.device("cuda:0")

    def forward(self, ids: torch.Tensor) -> None:
        if not isinstance(ids, torch.Tensor):
            raise TypeError("Enter value should consists type torch.Tensor")

        PE = torch.zeros((self.sent_len, self.d_model)).to(self.device)
        pos = torch.arange(self.sent_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model)
        )

        PE[:, 0::2] = torch.sin(pos * div_term)
        PE[:, 1::2] = torch.cos(pos * div_term)

        return ids + PE
