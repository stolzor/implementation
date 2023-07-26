from abc import ABC, abstractmethod
from typing import List, Literal
import math

import torch


class AttentionBase(ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def forward(self, key, query, values):
        ...
    
    def positional_encoding(
            self, 
            ids: List[int | float] | torch.Tensor, 
            d_model: int = 512, 
            max_len: int = 128
        ) -> None:
        if isinstance(ids, list):
            ids = torch.Tensor(ids)
        
        PE = torch.zeros((max_len, d_model))
        pos = torch.arange(max_len)[:, None]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        PE[:, 0::2] = torch.sin(pos * div_term)
        PE[:, 1::2] = torch.cos(pos * div_term)

        return ids + PE
