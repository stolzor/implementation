from typing import List
import math

import torch
from torch import nn


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, sent_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.sent_len = sent_len
    
    def forward(self, ids: List[int | float]) -> None:
        if not isinstance(ids, torch.Tensor):
            ids = torch.Tensor(ids)

        if len(ids.shape) == 1:
            ids = ids.unsqueeze(0)

        h, w = ids.shape
        pad_ids = torch.zeros((self.sent_len, self.d_model))
        pad_ids[:h, :w] += pad_ids[:h, :w] + ids

        PE = torch.zeros((self.sent_len, self.d_model))
        pos = torch.arange(self.sent_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model)
        )

        PE[:, 0::2] = torch.sin(pos * div_term)
        PE[:, 1::2] = torch.cos(pos * div_term)

        return pad_ids + PE
