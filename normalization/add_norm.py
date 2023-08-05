from typing import List

import torch
from torch import nn

from layer_norm import LayerNorm


class AddNorm(nn.Module):
    def __init__(self, enter_shape: List[int], dropout: float = 0.2) -> None:
        super().__init__()
        self.layer_norm = LayerNorm(enter_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, residual_x: torch.Tensor, current_x: torch.Tensor
    ) -> torch.Tensor:
        x = self.layer_norm(residual_x + self.dropout(current_x))
        return x
